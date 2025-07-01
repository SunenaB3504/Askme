"""
Voice Activity Detection for AskMe Voice Assistant
"""

import logging
import numpy as np
from typing import Optional, List
import asyncio
import threading
import time

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    WEBRTC_VAD_AVAILABLE = False


class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC VAD"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not WEBRTC_VAD_AVAILABLE:
            self.logger.warning("WebRTC VAD not available, using energy-based VAD")
            self.vad = None
        else:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(config.aggressiveness)
        
        # Audio buffer for VAD
        self.audio_buffer = []
        self.sample_rate = 16000  # WebRTC VAD requires 16kHz
        self.frame_duration_ms = config.frame_duration_ms
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # State tracking
        self.is_speech_detected = False
        self.speech_start_time = None
        self.silence_start_time = None
        
        # Thresholds
        self.min_speech_duration = 0.3  # Minimum speech duration in seconds
        self.max_silence_duration = 2.0  # Maximum silence duration in seconds
        
    async def detect_speech(self, audio_chunk: Optional[bytes] = None) -> bool:
        """
        Detect speech in audio chunk
        Returns True if speech is detected and should start recording
        """
        if audio_chunk:
            self.audio_buffer.extend(audio_chunk)
        
        # Process complete frames
        while len(self.audio_buffer) >= self.frame_size * 2:  # 16-bit samples
            frame_bytes = bytes(self.audio_buffer[:self.frame_size * 2])
            self.audio_buffer = self.audio_buffer[self.frame_size * 2:]
            
            is_speech = self._detect_speech_in_frame(frame_bytes)
            
            current_time = time.time()
            
            if is_speech:
                if not self.is_speech_detected:
                    # Speech started
                    self.is_speech_detected = True
                    self.speech_start_time = current_time
                    self.silence_start_time = None
                    self.logger.debug("Speech detected")
                
            else:  # Silence
                if self.is_speech_detected:
                    if self.silence_start_time is None:
                        self.silence_start_time = current_time
                    
                    # Check if silence has lasted too long
                    silence_duration = current_time - self.silence_start_time
                    if silence_duration > self.max_silence_duration:
                        # End of speech
                        speech_duration = self.silence_start_time - self.speech_start_time
                        if speech_duration >= self.min_speech_duration:
                            self.logger.debug(f"Speech ended (duration: {speech_duration:.2f}s)")
                            self._reset_state()
                            return True  # Signal to start processing
                        else:
                            self.logger.debug("Speech too short, ignoring")
                            self._reset_state()
        
        return False
    
    def _detect_speech_in_frame(self, frame_bytes: bytes) -> bool:
        """Detect speech in a single audio frame"""
        if self.vad and WEBRTC_VAD_AVAILABLE:
            try:
                return self.vad.is_speech(frame_bytes, self.sample_rate)
            except Exception as e:
                self.logger.warning(f"WebRTC VAD error: {e}")
                return self._energy_based_vad(frame_bytes)
        else:
            return self._energy_based_vad(frame_bytes)
    
    def _energy_based_vad(self, frame_bytes: bytes) -> bool:
        """Fallback energy-based voice activity detection"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(frame_bytes, dtype=np.int16)
        
        # Calculate energy
        energy = np.sum(audio_data.astype(np.float32) ** 2) / len(audio_data)
        
        # Simple threshold-based detection
        # This threshold may need tuning based on environment
        threshold = 1000000  # Adjust based on testing
        
        return energy > threshold
    
    def _reset_state(self):
        """Reset VAD state"""
        self.is_speech_detected = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.audio_buffer.clear()
    
    def get_speech_probability(self, audio_chunk: bytes) -> float:
        """Get speech probability for audio chunk (0.0 to 1.0)"""
        if not audio_chunk:
            return 0.0
        
        # Split into frames and get average probability
        frame_size_bytes = self.frame_size * 2
        probabilities = []
        
        for i in range(0, len(audio_chunk) - frame_size_bytes, frame_size_bytes):
            frame = audio_chunk[i:i + frame_size_bytes]
            if len(frame) == frame_size_bytes:
                is_speech = self._detect_speech_in_frame(frame)
                probabilities.append(1.0 if is_speech else 0.0)
        
        if probabilities:
            return sum(probabilities) / len(probabilities)
        return 0.0
    
    def set_aggressiveness(self, level: int):
        """Set VAD aggressiveness level (0-3)"""
        if 0 <= level <= 3:
            self.config.aggressiveness = level
            if self.vad:
                self.vad.set_mode(level)
        else:
            self.logger.warning(f"Invalid aggressiveness level: {level}")
    
    def is_available(self) -> bool:
        """Check if VAD is available and working"""
        return WEBRTC_VAD_AVAILABLE and self.vad is not None


class SilenceDetector:
    """Detect silence periods in audio streams"""
    
    def __init__(self, 
                 silence_threshold: float = 0.01,
                 min_silence_duration: float = 1.0,
                 sample_rate: int = 16000):
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        
        self.audio_buffer = []
        self.silence_start = None
        self.is_silent = False
        
    def process_audio(self, audio_chunk: bytes) -> bool:
        """
        Process audio chunk and return True if silence period detected
        """
        # Convert to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        current_time = time.time()
        
        if rms < self.silence_threshold:
            # Silent frame
            if not self.is_silent:
                self.is_silent = True
                self.silence_start = current_time
            
            # Check if silence duration exceeded threshold
            if current_time - self.silence_start >= self.min_silence_duration:
                return True
        else:
            # Non-silent frame
            self.is_silent = False
            self.silence_start = None
        
        return False
    
    def reset(self):
        """Reset silence detector state"""
        self.is_silent = False
        self.silence_start = None
        self.audio_buffer.clear()


class ContinuousVAD:
    """Continuous Voice Activity Detection for real-time processing"""
    
    def __init__(self, 
                 config,
                 on_speech_start: callable = None,
                 on_speech_end: callable = None):
        self.config = config
        self.vad = VoiceActivityDetector(config)
        self.silence_detector = SilenceDetector()
        
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        
        self.is_running = False
        self.processing_thread = None
        
        # Audio buffer for continuous processing
        self.audio_queue = asyncio.Queue()
        
    async def start(self):
        """Start continuous VAD processing"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start processing task
        asyncio.create_task(self._processing_loop())
    
    def stop(self):
        """Stop continuous VAD processing"""
        self.is_running = False
    
    async def add_audio(self, audio_chunk: bytes):
        """Add audio chunk for processing"""
        if self.is_running:
            await self.audio_queue.put(audio_chunk)
    
    async def _processing_loop(self):
        """Main processing loop for continuous VAD"""
        while self.is_running:
            try:
                # Get audio chunk with timeout
                audio_chunk = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=0.1
                )
                
                # Process with VAD
                speech_detected = await self.vad.detect_speech(audio_chunk)
                
                if speech_detected and self.on_speech_start:
                    asyncio.create_task(self.on_speech_start())
                
                # Also check for silence
                silence_detected = self.silence_detector.process_audio(audio_chunk)
                
                if silence_detected and self.on_speech_end:
                    asyncio.create_task(self.on_speech_end())
                
            except asyncio.TimeoutError:
                # No audio data, continue
                continue
            except Exception as e:
                logging.error(f"VAD processing error: {e}")
                await asyncio.sleep(0.1)
