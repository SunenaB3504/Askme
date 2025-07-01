"""
Whisper ASR implementation for AskMe Voice Assistant
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional
import tempfile
import os

try:
    import whisper
    import torch
except ImportError as e:
    whisper = None
    torch = None
    import_error = e


class WhisperASR:
    """Whisper-based Automatic Speech Recognition"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.device = "cpu"
        self.is_initialized = False
        
        if whisper is None:
            raise ImportError(f"Whisper not available: {import_error}")
    
    async def initialize(self):
        """Initialize Whisper model"""
        try:
            self.logger.info(f"Loading Whisper model: {self.config.model}")
            
            # Determine device
            if self.config.device == "auto":
                if torch and torch.cuda.is_available():
                    self.device = "cuda"
                elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self.config.device
            
            # Load model
            self.model = whisper.load_model(
                name=self.config.model,
                device=self.device
            )
            
            self.is_initialized = True
            self.logger.info(f"✓ Whisper model loaded on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper: {e}")
            raise
    
    async def transcribe(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio data"""
        if not self.is_initialized:
            raise RuntimeError("Whisper ASR not initialized")
        
        try:
            # Convert bytes to numpy array
            audio_np = self._bytes_to_numpy(audio_data)
            
            # Transcribe
            result = self.model.transcribe(
                audio_np,
                language=self.config.language,
                task=self.config.task,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                patience=self.config.patience,
                length_penalty=self.config.length_penalty,
                repetition_penalty=self.config.repetition_penalty,
                no_speech_threshold=self.config.no_speech_threshold,
                logprob_threshold=self.config.logprob_threshold,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                condition_on_previous_text=self.config.condition_on_previous_text,
                fp16=self.device != "cpu"
            )
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "confidence": self._calculate_confidence(result)
            }
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return {"text": "", "error": str(e)}
    
    def _bytes_to_numpy(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        # Assume 16-bit PCM audio at 16kHz
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32 and normalize
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        # Ensure sample rate is 16kHz (Whisper's expected input)
        # If different sample rate, we'd need to resample here
        
        return audio_np
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate confidence score from Whisper result"""
        if "segments" in result and result["segments"]:
            # Average the confidence scores from segments
            confidences = []
            for segment in result["segments"]:
                if "avg_logprob" in segment:
                    # Convert log probability to confidence (0-1)
                    confidence = np.exp(segment["avg_logprob"])
                    confidences.append(confidence)
            
            if confidences:
                return float(np.mean(confidences))
        
        return 0.8  # Default confidence if no segments
    
    def is_ready(self) -> bool:
        """Check if ASR is ready"""
        return self.is_initialized and self.model is not None
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        self.logger.info("Whisper ASR cleaned up")
