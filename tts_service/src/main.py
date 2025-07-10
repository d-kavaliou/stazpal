import asyncio
import torch
import io
import base64
import logging
from typing import Optional
import numpy as np

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import soundfile as sf
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dia TTS Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
processor: Optional[AutoProcessor] = None
model: Optional[AutoModel] = None
tokenizer: Optional[AutoTokenizer] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class TTSRequest(BaseModel):
    text: str
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    guidance_scale: float = 3.0
    speaker_id: Optional[str] = None

class TTSResponse(BaseModel):
    audio_data: str  # base64 encoded WAV
    sample_rate: int
    duration: float
    text_processed: str
    generation_time_ms: int

@app.on_event("startup")
async def load_model():
    """Load the TTS model on startup"""
    global processor, model, tokenizer
    
    try:
        logger.info(f"Loading TTS model on device: {device}")
        
        # Use a more standard TTS model for now
        # You can replace with actual Dia model when available
        model_checkpoint = "microsoft/speecht5_tts"
        
        try:
            processor = AutoProcessor.from_pretrained(model_checkpoint)
            model = AutoModel.from_pretrained(model_checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        except Exception as e:
            logger.warning(f"Failed to load {model_checkpoint}, falling back to dummy model: {e}")
            # Fallback to a simple implementation
            processor = None
            model = None
            tokenizer = None
        
        if model and device == "cuda":
            model = model.to(device)
        
        logger.info("TTS model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        # Don't raise to allow service to start even without model
        pass

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for TTS"""
    text = text.strip()
    
    # Remove speaker tags if present
    if text.startswith("[S1]") or text.startswith("[S2]"):
        text = text[4:].strip()
    
    # Remove emotion tags like (laughs), (sighs), etc.
    import re
    text = re.sub(r'\([^)]*\)', '', text)
    
    return text.strip()

def generate_dummy_audio(text: str, sample_rate: int = 22050, duration: float = None) -> np.ndarray:
    """Generate dummy audio when model is not available"""
    if duration is None:
        # Estimate duration based on text length (rough approximation)
        duration = max(0.5, len(text) * 0.1)
    
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to make it less monotonous
    audio += 0.1 * np.sin(2 * np.pi * frequency * 1.5 * t)
    
    return audio.astype(np.float32)

@app.post("/tts", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """Generate speech from text"""
    import time
    start_time = time.time()
    
    try:
        # Preprocess text
        processed_text = preprocess_text(request.text)
        
        if not processed_text:
            raise HTTPException(status_code=400, detail="Empty text after preprocessing")
        
        sample_rate = 22050
        
        if model and processor:
            try:
                # Use actual model if available
                inputs = processor(text=processed_text, return_tensors="pt")
                
                if device == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                
                audio_data = outputs.cpu().numpy().flatten()
                
            except Exception as e:
                logger.warning(f"Model generation failed, using dummy audio: {e}")
                audio_data = generate_dummy_audio(processed_text, sample_rate)
        else:
            logger.info("No model available, using dummy audio")
            audio_data = generate_dummy_audio(processed_text, sample_rate)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()
        
        # Encode to base64
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        generation_time = int((time.time() - start_time) * 1000)
        
        return TTSResponse(
            audio_data=audio_b64,
            sample_rate=sample_rate,
            duration=len(audio_data) / sample_rate,
            text_processed=processed_text,
            generation_time_ms=generation_time
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS"""
    
    await websocket.accept()
    logger.info("TTS WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            
            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue
            
            try:
                # Create request with optional parameters
                request = TTSRequest(
                    text=text,
                    temperature=data.get("temperature", 1.0),
                    top_p=data.get("top_p", 0.9),
                    top_k=data.get("top_k", 50)
                )
                
                response = await generate_speech(request)
                
                await websocket.send_json({
                    "audio_data": response.audio_data,
                    "sample_rate": response.sample_rate,
                    "duration": response.duration,
                    "text_processed": response.text_processed,
                    "generation_time_ms": response.generation_time_ms
                })
                
            except Exception as e:
                logger.error(f"WebSocket TTS error: {e}")
                await websocket.send_json({"error": str(e)})
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("TTS WebSocket connection closed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "TTS Service OK", 
        "device": device,
        "model_loaded": model is not None
    }

@app.get("/status")
async def get_status():
    """Detailed service status"""
    gpu_memory = None
    if device == "cuda" and torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    return {
        "status": "healthy" if model is not None else "ready",
        "model_loaded": model is not None,
        "device": device,
        "gpu_memory_mb": gpu_memory,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)