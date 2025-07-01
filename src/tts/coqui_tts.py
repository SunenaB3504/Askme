"""
Coqui TTS implementation for AskMe Voice Assistant
"""

import asyncio
import logging
import tempfile
import os
from typing import Optional, List
import numpy as np

try:
    from TTS.api import TTS
    import torch
    TTS_AVAILABLE = True
except ImportError:
    TTS = None
    torch = None
    TTS_AVAILABLE = False


class CoquiTTS:
    """Coqui TTS engine for speech synthesis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tts = None
        self.is_initialized = False
        
        if not TTS_AVAILABLE:
            raise ImportError("TTS (Coqui TTS) not available")
    
    async def initialize(self):
        """Initialize TTS model"""
        try:
            self.logger.info(f"Loading TTS model: {self.config.model}")
            
            # Determine device
            device = "cpu"
            if self.config.device == "auto":
                if torch and torch.cuda.is_available():
                    device = "cuda"
                elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
            else:
                device = self.config.device
            
            # Initialize TTS
            gpu = device == "cuda"
            self.tts = TTS(
                model_name=self.config.model,
                gpu=gpu,
                progress_bar=False
            )
            
            self.is_initialized = True
            self.logger.info(f"✓ TTS model loaded on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    async def synthesize(self, text: str, speaker: Optional[str] = None) -> Optional[bytes]:
        """Synthesize speech from text"""
        if not self.is_initialized:
            raise RuntimeError("TTS not initialized")
        
        if not text or not text.strip():
            return None
        
        try:
            # Clean and prepare text
            text = self._prepare_text(text)
            
            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                speaker
            )
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            return None
    
    def _synthesize_sync(self, text: str, speaker: Optional[str] = None) -> Optional[bytes]:
        """Synchronous speech synthesis"""
        try:
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Synthesize speech
            if self._is_multispeaker_model():
                # Multi-speaker model
                self.tts.tts_to_file(
                    text=text,
                    speaker=speaker or self.config.speaker,
                    file_path=temp_path,
                    speed=self.config.speed
                )
            else:
                # Single-speaker model
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_path,
                    speed=self.config.speed
                )
            
            # Read audio data
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Sync synthesis failed: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass
            return None
    
    def _prepare_text(self, text: str) -> str:
        """Prepare text for synthesis"""
        # Basic text cleaning
        text = text.strip()
        
        # Remove or replace problematic characters
        replacements = {
            '"': "'",
            '"': "'", 
            '"': "'",
            ''': "'",
            ''': "'",
            '…': "...",
            '–': "-",
            '—': "-",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Ensure text ends with punctuation for natural speech
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _is_multispeaker_model(self) -> bool:
        """Check if the model supports multiple speakers"""
        try:
            return hasattr(self.tts, 'speakers') and self.tts.speakers is not None
        except:
            return False
    
    def get_available_speakers(self) -> List[str]:
        """Get list of available speakers for multi-speaker models"""
        if not self.is_initialized:
            return []
        
        try:
            if self._is_multispeaker_model():
                return list(self.tts.speakers)
            return []
        except Exception as e:
            self.logger.error(f"Failed to get speakers: {e}")
            return []
    
    def synthesize_to_file(self, text: str, output_path: str, speaker: Optional[str] = None) -> bool:
        """Synthesize speech directly to file"""
        if not self.is_initialized:
            raise RuntimeError("TTS not initialized")
        
        try:
            text = self._prepare_text(text)
            
            if self._is_multispeaker_model():
                self.tts.tts_to_file(
                    text=text,
                    speaker=speaker or self.config.speaker,
                    file_path=output_path,
                    speed=self.config.speed
                )
            else:
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speed=self.config.speed
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize to file: {e}")
            return False
    
    def clone_voice_from_audio(self, reference_audio_path: str, target_text: str, output_path: str) -> bool:
        """Clone voice from reference audio (if supported by model)"""
        if not self.is_initialized:
            return False
        
        try:
            # This requires a voice cloning model like XTTS
            if hasattr(self.tts, 'tts_with_vc'):
                self.tts.tts_with_vc(
                    text=target_text,
                    speaker_wav=reference_audio_path,
                    file_path=output_path,
                    speed=self.config.speed
                )
                return True
            else:
                self.logger.warning("Voice cloning not supported by current model")
                return False
                
        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}")
            return False
    
    def set_speaking_rate(self, rate: float):
        """Set speaking rate (speed)"""
        if 0.5 <= rate <= 2.0:
            self.config.speed = rate
        else:
            self.logger.warning(f"Invalid speaking rate: {rate}. Must be between 0.5 and 2.0")
    
    def is_ready(self) -> bool:
        """Check if TTS is ready"""
        return self.is_initialized and self.tts is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.is_initialized:
            return {"status": "not_ready"}
        
        return {
            "status": "ready",
            "model_name": self.config.model,
            "language": self.config.language,
            "speakers": self.get_available_speakers(),
            "supports_voice_cloning": hasattr(self.tts, 'tts_with_vc'),
            "current_speed": self.config.speed
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.tts is not None:
            del self.tts
            self.tts = None
        
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        self.logger.info("TTS engine cleaned up")
