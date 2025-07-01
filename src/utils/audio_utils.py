"""
Audio utilities for AskMe Voice Assistant
"""

import asyncio
import logging
import threading
import queue
from typing import Optional, Callable
import numpy as np

try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    wave = None
    PYAUDIO_AVAILABLE = False


class AudioRecorder:
    """Audio recording using PyAudio"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio not available")
        
        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.record_thread = None
        
    async def initialize(self):
        """Initialize audio system"""
        try:
            self.audio = pyaudio.PyAudio()
            self.logger.info("✓ Audio recorder initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio recorder: {e}")
            raise
    
    async def start(self):
        """Start audio recording"""
        if self.is_recording:
            return
        
        try:
            self.is_recording = True
            self.audio_queue = queue.Queue()  # Clear queue
            
            # Start recording in a separate thread
            self.record_thread = threading.Thread(target=self._record_loop)
            self.record_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise
    
    async def stop(self) -> Optional[bytes]:
        """Stop recording and return audio data"""
        if not self.is_recording:
            return None
        
        try:
            self.is_recording = False
            
            # Wait for recording thread to finish
            if self.record_thread:
                self.record_thread.join(timeout=5)
            
            # Collect all audio data
            audio_chunks = []
            while not self.audio_queue.empty():
                try:
                    chunk = self.audio_queue.get_nowait()
                    audio_chunks.append(chunk)
                except queue.Empty:
                    break
            
            if not audio_chunks:
                return None
            
            # Combine chunks
            audio_data = b''.join(audio_chunks)
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}")
            return None
    
    def _record_loop(self):
        """Main recording loop (runs in separate thread)"""
        try:
            # Open audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input.device_index,
                frames_per_buffer=self.config.chunk_size
            )
            
            while self.is_recording:
                try:
                    # Read audio chunk
                    data = stream.read(
                        self.config.chunk_size,
                        exception_on_overflow=False
                    )
                    self.audio_queue.put(data)
                    
                except Exception as e:
                    self.logger.error(f"Error reading audio: {e}")
                    break
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            self.logger.error(f"Recording loop error: {e}")
            self.is_recording = False
    
    def get_input_devices(self) -> list:
        """Get list of available input devices"""
        if not self.audio:
            return []
        
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': info['defaultSampleRate']
                    })
            except:
                continue
        
        return devices
    
    async def cleanup(self):
        """Cleanup audio resources"""
        await self.stop()
        
        if self.audio:
            self.audio.terminate()
            self.audio = None


class AudioPlayer:
    """Audio playback using PyAudio"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio not available")
        
        self.audio = None
        
    async def initialize(self):
        """Initialize audio system"""
        try:
            self.audio = pyaudio.PyAudio()
            self.logger.info("✓ Audio player initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio player: {e}")
            raise
    
    async def play(self, audio_data: bytes):
        """Play audio data"""
        if not audio_data:
            return
        
        try:
            # Run playback in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._play_sync, audio_data)
            
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
    
    def _play_sync(self, audio_data: bytes):
        """Synchronous audio playback"""
        try:
            # For now, assume WAV format
            # In a real implementation, you'd want to detect format
            self._play_wav(audio_data)
            
        except Exception as e:
            self.logger.error(f"Sync playback failed: {e}")
    
    def _play_wav(self, wav_data: bytes):
        """Play WAV audio data"""
        import io
        
        try:
            # Read WAV file from bytes
            wav_io = io.BytesIO(wav_data)
            wav_file = wave.open(wav_io, 'rb')
            
            # Get audio parameters
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            
            # Determine PyAudio format
            if sample_width == 1:
                format = pyaudio.paUInt8
            elif sample_width == 2:
                format = pyaudio.paInt16
            elif sample_width == 4:
                format = pyaudio.paInt32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Open audio stream
            stream = self.audio.open(
                format=format,
                channels=channels,
                rate=sample_rate,
                output=True,
                output_device_index=self.config.output.device_index
            )
            
            # Play audio
            chunk_size = 1024
            data = wav_file.readframes(chunk_size)
            
            while data:
                stream.write(data)
                data = wav_file.readframes(chunk_size)
            
            # Close stream
            stream.stop_stream()
            stream.close()
            wav_file.close()
            
        except Exception as e:
            self.logger.error(f"WAV playback error: {e}")
    
    def get_output_devices(self) -> list:
        """Get list of available output devices"""
        if not self.audio:
            return []
        
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxOutputChannels'],
                        'sample_rate': info['defaultSampleRate']
                    })
            except:
                continue
        
        return devices
    
    async def cleanup(self):
        """Cleanup audio resources"""
        if self.audio:
            self.audio.terminate()
            self.audio = None


class AudioUtils:
    """Audio utility functions"""
    
    @staticmethod
    def convert_sample_rate(audio_data: np.ndarray, 
                          source_rate: int, 
                          target_rate: int) -> np.ndarray:
        """Convert audio sample rate"""
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=source_rate, target_sr=target_rate)
        except ImportError:
            # Fallback: basic resampling (not ideal but functional)
            ratio = target_rate / source_rate
            new_length = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            return np.interp(indices, np.arange(len(audio_data)), audio_data)
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    @staticmethod
    def apply_noise_reduction(audio_data: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction"""
        # Simple high-pass filter to remove low-frequency noise
        from scipy import signal
        try:
            b, a = signal.butter(4, 80, 'high', fs=16000)
            return signal.filtfilt(b, a, audio_data)
        except ImportError:
            # Return original if scipy not available
            return audio_data
    
    @staticmethod
    def detect_silence(audio_data: np.ndarray, 
                      threshold: float = 0.01, 
                      min_silence_duration: float = 0.5,
                      sample_rate: int = 16000) -> list:
        """Detect silent regions in audio"""
        # Convert to energy
        energy = np.square(audio_data)
        
        # Apply moving average
        window_size = int(0.1 * sample_rate)  # 100ms window
        energy_smoothed = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
        
        # Find silent regions
        is_silent = energy_smoothed < threshold
        
        # Find continuous silent regions
        silent_regions = []
        start = None
        
        for i, silent in enumerate(is_silent):
            if silent and start is None:
                start = i
            elif not silent and start is not None:
                duration = (i - start) / sample_rate
                if duration >= min_silence_duration:
                    silent_regions.append((start / sample_rate, i / sample_rate))
                start = None
        
        return silent_regions
