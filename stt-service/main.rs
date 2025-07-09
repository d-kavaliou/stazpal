use axum::{
    extract::{ws::WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::{info, error, warn};
use serde::{Deserialize, Serialize};

mod kyutai_integration;

#[derive(Debug, Serialize, Deserialize)]
struct STTResponse {
    text: String,
    confidence: f32,
    is_final: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::init();
    
    let app = Router::new()
        .route("/ws/stt", get(stt_websocket_handler))
        .route("/health", get(health_check))
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], 8001));
    info!("STT Service listening on {}", addr);
    
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn stt_websocket_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_stt_socket)
}

async fn handle_stt_socket(mut socket: WebSocket) {
    info!("New STT WebSocket connection");
    
    // Initialize Kyutai processor
    let mut stt = match kyutai_integration::KyutaiSTT::new().await {
        Ok(stt) => stt,
        Err(e) => {
            error!("Failed to initialize STT: {}", e);
            return;
        }
    };
    
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(msg) => {
                if let Ok(audio_data) = msg.into_data() {
                    match stt.transcribe_chunk(&audio_data).await {
                        Ok(text) => {
                            let response = STTResponse {
                                text,
                                confidence: 0.9, // Placeholder
                                is_final: true,
                            };
                            
                            if let Ok(json) = serde_json::to_string(&response) {
                                if let Err(e) = socket.send(axum::extract::ws::Message::Text(json)).await {
                                    error!("Failed to send STT response: {}", e);
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            error!("STT processing error: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                warn!("WebSocket error: {}", e);
                break;
            }
        }
    }
    
    info!("STT WebSocket connection closed");
}

async fn health_check() -> &'static str {
    "STT Service OK"
}
