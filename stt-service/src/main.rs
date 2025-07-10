use std::collections::HashMap;
use std::sync::Arc;
use std::fs;
use std::io::Write;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, RwLock};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tracing::{info, error, warn};
use base64::{Engine as _, engine::general_purpose};

use anyhow::Result;
use candle::{Device, Tensor};
use kaudio;
use moshi;
use sentencepiece;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "start")]
    Start { session_id: String },
    #[serde(rename = "audio_chunk")]
    AudioChunk { data: String },
    #[serde(rename = "end")]
    End,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "connected")]
    Connected { session_id: String },
    #[serde(rename = "text_chunk")]
    TextChunk { text: String, is_final: bool },
    #[serde(rename = "session_end")]
    SessionEnd { session_id: String },
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Debug, serde::Deserialize)]
struct SttConfig {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    stt_config: SttConfig,
}

impl Config {
    fn model_config(&self, vad: bool) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        let extra_heads = if vad {
            Some(moshi::lm::ExtraHeadsConfig {
                num_heads: 4,
                dim: 6,
            })
        } else {
            None
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads,
        }
    }
}

struct MoshiSTTModel {
    state: moshi::asr::State,
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    config: Config,
    device: Device,
}

impl MoshiSTTModel {
    async fn load_from_hf(hf_repo: &str, use_vad: bool) -> Result<Self> {
        let device = Self::get_device()?;
        let dtype = device.bf16_default_to_f32();

        info!("Loading model from HuggingFace repository: {}", hf_repo);

        // Retrieve the model files from the Hugging Face Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(hf_repo.to_string());
        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&fs::read_to_string(&config_file)?)?;
        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get("model.safetensors")?;
        let mimi_file = repo.get(&config.mimi_name)?;

        info!("Loading tokenizer...");
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;
        
        info!("Loading model weights...");
        let vb_lm = unsafe { 
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &device)? 
        };
        
        info!("Loading audio tokenizer...");
        let audio_tokenizer = moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), &device)?;
        
        info!("Initializing language model...");
        let lm = moshi::lm::LmModel::new(
            &config.model_config(use_vad),
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
        )?;
        
        let asr_delay_in_tokens = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0., audio_tokenizer, lm)?;
        
        info!("✅ Moshi STT model loaded successfully");
        
        Ok(MoshiSTTModel {
            state,
            text_tokenizer,
            config,
            device,
        })
    }

    fn get_device() -> Result<Device> {
        if candle::utils::cuda_is_available() {
            info!("Using CUDA device");
            Ok(Device::new_cuda(0)?)
        } else if candle::utils::metal_is_available() {
            info!("Using Metal device");
            Ok(Device::new_metal(0)?)
        } else {
            info!("Using CPU device");
            Ok(Device::Cpu)
        }
    }

    fn process_audio_chunk(&mut self, audio_data: &[f32]) -> Result<Vec<String>> {
        let mut transcriptions = Vec::new();
        
        // Process audio in chunks of 1920 samples (80ms at 24kHz)
        for chunk in audio_data.chunks(1920) {
            let pcm = Tensor::new(chunk, &self.device)?.reshape((1, 1, ()))?;
            let asr_msgs = self.state.step_pcm(pcm, None, &().into(), |_, _, _| ())?;
            
            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Word { tokens, .. } => {
                        let word = self
                            .text_tokenizer
                            .decode_piece_ids(tokens)
                            .unwrap_or_else(|_| String::new());
                        if !word.trim().is_empty() {
                            transcriptions.push(word.trim().to_string());
                        }
                    }
                    moshi::asr::AsrMsg::Step { .. } => {
                        // VAD information - could be used for end-of-turn detection
                    }
                    moshi::asr::AsrMsg::EndWord { .. } => {
                        // Word boundary information
                    }
                }
            }
        }
        
        Ok(transcriptions)
    }
}

struct AudioSession {
    buffer: Vec<f32>,
    last_processed: std::time::Instant,
    model: MoshiSTTModel,
}

type Sessions = Arc<RwLock<HashMap<String, mpsc::UnboundedSender<Vec<u8>>>>>;
type AudioSessions = Arc<RwLock<HashMap<String, AudioSession>>>;
type WsSender = futures_util::stream::SplitSink<tokio_tungstenite::WebSocketStream<TcpStream>, Message>;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    let sessions: Sessions = Arc::new(RwLock::new(HashMap::new()));
    let audio_sessions: AudioSessions = Arc::new(RwLock::new(HashMap::new()));
    
    info!("Starting Native Rust STT service with Moshi...");
    
    // Pre-load the model (this takes some time)
    let _model = MoshiSTTModel::load_from_hf("kyutai/stt-1b-en_fr-candle", false).await?;
    info!("Model pre-loaded successfully");
    
    let listener = TcpListener::bind("127.0.0.1:3030").await?;
    info!("🚀 Native Rust STT service listening on ws://127.0.0.1:3030");
    
    while let Ok((stream, addr)) = listener.accept().await {
        info!("📡 New connection from: {}", addr);
        let sessions = sessions.clone();
        let audio_sessions = audio_sessions.clone();
        tokio::spawn(handle_connection(stream, sessions, audio_sessions));
    }
    
    Ok(())
}

async fn handle_connection(
    stream: TcpStream, 
    sessions: Sessions, 
    audio_sessions: AudioSessions
) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws_stream) => ws_stream,
        Err(e) => {
            error!("Error during WebSocket handshake: {}", e);
            return;
        }
    };
    
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    let session_id = Uuid::new_v4().to_string();
    
    // Send connection confirmation
    let connected_msg = ServerMessage::Connected { 
        session_id: session_id.clone() 
    };
    
    if let Err(e) = ws_sender.send(Message::Text(
        serde_json::to_string(&connected_msg).unwrap()
    )).await {
        error!("Failed to send connected message: {}", e);
        return;
    }
    
    // Create channel for audio processing
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<u8>>();
    sessions.write().await.insert(session_id.clone(), audio_tx);
    
    // Initialize audio session with model
    match MoshiSTTModel::load_from_hf("kyutai/stt-1b-en_fr-candle", false).await {
        Ok(model) => {
            audio_sessions.write().await.insert(session_id.clone(), AudioSession {
                buffer: Vec::new(),
                last_processed: std::time::Instant::now(),
                model,
            });
        }
        Err(e) => {
            error!("Failed to load model for session {}: {}", session_id, e);
            let error_msg = ServerMessage::Error {
                message: format!("Failed to initialize STT model: {}", e),
            };
            let _ = ws_sender.send(Message::Text(
                serde_json::to_string(&error_msg).unwrap()
            )).await;
            return;
        }
    }
    
    // Spawn audio processing task
    let session_id_clone = session_id.clone();
    let audio_sessions_clone = audio_sessions.clone();
    
    tokio::spawn(async move {
        process_audio_stream(session_id_clone, audio_rx, ws_sender, audio_sessions_clone).await;
    });
    
    // Handle incoming WebSocket messages
    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Err(e) = handle_client_message(text, &session_id, &sessions).await {
                    error!("Error handling client message: {}", e);
                    break;
                }
            }
            Ok(Message::Close(_)) => {
                info!("Client disconnected");
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
    
    // Cleanup
    sessions.write().await.remove(&session_id);
    audio_sessions.write().await.remove(&session_id);
    info!("Session {} ended", session_id);
}

async fn handle_client_message(
    text: String, 
    session_id: &str, 
    sessions: &Sessions
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let message: ClientMessage = serde_json::from_str(&text)?;
    
    match message {
        ClientMessage::Start { .. } => {
            info!("🎙️ Session {} started recording", session_id);
        }
        ClientMessage::AudioChunk { data } => {
            let audio_data = general_purpose::STANDARD.decode(&data)?;
            
            if let Some(sender) = sessions.read().await.get(session_id) {
                if let Err(e) = sender.send(audio_data) {
                    error!("Failed to send audio data: {}", e);
                }
            }
        }
        ClientMessage::End => {
            info!("⏹️ Session {} ended by client", session_id);
        }
    }
    
    Ok(())
}

async fn process_audio_stream(
    session_id: String,
    mut audio_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    mut ws_sender: WsSender,
    audio_sessions: AudioSessions,
) {
    info!("🔄 Starting native audio processing for session {}", session_id);
    
    while let Some(audio_chunk) = audio_rx.recv().await {
        if let Some(session) = audio_sessions.write().await.get_mut(&session_id) {
            // Convert WebM audio to raw PCM
            match convert_webm_to_pcm(&audio_chunk).await {
                Ok(pcm_data) => {
                    session.buffer.extend_from_slice(&pcm_data);
                    
                    // Process audio every 2 seconds or when buffer is large enough
                    let should_process = session.last_processed.elapsed().as_secs() >= 2 
                        || session.buffer.len() > 48000; // ~2 seconds of audio at 24kHz
                    
                    if should_process && !session.buffer.is_empty() {
                        let audio_data = session.buffer.clone();
                        session.buffer.clear();
                        session.last_processed = std::time::Instant::now();
                        
                        // Process audio with native Moshi
                        match session.model.process_audio_chunk(&audio_data) {
                            Ok(transcriptions) => {
                                for text in transcriptions {
                                    if !text.trim().is_empty() {
                                        let text_chunk = ServerMessage::TextChunk {
                                            text: text.trim().to_string(),
                                            is_final: false,
                                        };
                                        
                                        if let Err(e) = ws_sender.send(Message::Text(
                                            serde_json::to_string(&text_chunk).unwrap()
                                        )).await {
                                            error!("Failed to send text chunk: {}", e);
                                            return;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Native STT processing error: {}", e);
                                let error_msg = ServerMessage::Error {
                                    message: format!("STT processing failed: {}", e),
                                };
                                let _ = ws_sender.send(Message::Text(
                                    serde_json::to_string(&error_msg).unwrap()
                                )).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Audio conversion error: {}", e);
                }
            }
        }
    }
    
    // Process any remaining audio
    if let Some(session) = audio_sessions.write().await.get_mut(&session_id) {
        if !session.buffer.is_empty() {
            match session.model.process_audio_chunk(&session.buffer) {
                Ok(transcriptions) => {
                    for text in transcriptions {
                        if !text.trim().is_empty() {
                            let final_chunk = ServerMessage::TextChunk {
                                text: text.trim().to_string(),
                                is_final: true,
                            };
                            let _ = ws_sender.send(Message::Text(
                                serde_json::to_string(&final_chunk).unwrap()
                            )).await;
                        }
                    }
                }
                Err(e) => {
                    error!("Final processing error: {}", e);
                }
            }
        }
    }
    
    // Send session end message
    let final_msg = ServerMessage::SessionEnd { session_id: session_id.clone() };
    let _ = ws_sender.send(Message::Text(
        serde_json::to_string(&final_msg).unwrap()
    )).await;
    
    info!("✅ Native audio processing completed for session {}", session_id);
}

async fn convert_webm_to_pcm(webm_data: &[u8]) -> Result<Vec<f32>> {
    // Write WebM data to temporary file
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!("audio_{}.webm", Uuid::new_v4()));
    
    fs::write(&temp_file, webm_data)?;
    
    // Use kaudio to decode the audio file
    let (pcm_data, sample_rate) = kaudio::pcm_decode(&temp_file)?;
    
    // Clean up temporary file
    let _ = fs::remove_file(&temp_file);
    
    // Resample to 24kHz if needed (Moshi expects 24kHz)
    let pcm_24k = if sample_rate != 24000 {
        kaudio::resample(&pcm_data, sample_rate as usize, 24000)?
    } else {
        pcm_data
    };
    
    Ok(pcm_24k)
}