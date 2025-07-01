"""
Core Voice Assistant implementation for AskMe

This module implements the main VoiceAssistant class that coordinates
ASR, LLM processing, and TTS components.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from ..asr.whisper_asr import WhisperASR
from ..llm.llama_inference import LlamaInference
from ..tts.coqui_tts import CoquiTTS
from ..core.config import Config
from ..utils.audio_utils import AudioRecorder, AudioPlayer
from ..utils.vad import VoiceActivityDetector


class VoiceAssistant:
    """
    Main Voice Assistant class that coordinates all components
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.asr: Optional[WhisperASR] = None
        self.llm: Optional[LlamaInference] = None
        self.tts: Optional[CoquiTTS] = None
        self.audio_recorder: Optional[AudioRecorder] = None
        self.audio_player: Optional[AudioPlayer] = None
        self.vad: Optional[VoiceActivityDetector] = None
        
        # State management
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False
        self.conversation_history = []
        
        # Callbacks for UI integration
        self.on_listening_start: Optional[Callable] = None
        self.on_listening_stop: Optional[Callable] = None
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
    async def initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing Voice Assistant components...")
        
        try:
            # Initialize ASR
            self.logger.info("Initializing ASR (Whisper)...")
            self.asr = WhisperASR(self.config.asr)
            await self.asr.initialize()
            
            # Initialize LLM
            self.logger.info("Initializing LLM...")
            self.llm = LlamaInference(self.config.llm)
            await self.llm.initialize()
            
            # Initialize TTS
            self.logger.info("Initializing TTS...")
            self.tts = CoquiTTS(self.config.tts)
            await self.tts.initialize()
            
            # Initialize audio components
            self.logger.info("Initializing audio components...")
            self.audio_recorder = AudioRecorder(self.config.audio)
            self.audio_player = AudioPlayer(self.config.audio)
            
            # Initialize VAD if enabled
            if self.config.audio.vad.enabled:
                self.vad = VoiceActivityDetector(self.config.audio.vad)
            
            self.logger.info("✓ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start(self):
        """Start the voice assistant"""
        self.logger.info("Starting Voice Assistant...")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._conversation_loop()),
            asyncio.create_task(self._health_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Voice Assistant stopped")
    
    async def shutdown(self):
        """Shutdown the voice assistant"""
        self.logger.info("Shutting down Voice Assistant...")
        
        # Stop audio recording
        if self.audio_recorder:
            await self.audio_recorder.stop()
        
        # Cleanup components
        if self.asr:
            await self.asr.cleanup()
        
        if self.llm:
            await self.llm.cleanup()
        
        if self.tts:
            await self.tts.cleanup()
        
        self.logger.info("✓ Voice Assistant shutdown complete")
    
    async def start_listening(self) -> bool:
        """Start listening for voice input"""
        if self.is_listening or self.is_speaking:
            return False
        
        try:
            self.is_listening = True
            self.logger.debug("Started listening for voice input")
            
            if self.on_listening_start:
                self.on_listening_start()
            
            # Start audio recording
            await self.audio_recorder.start()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start listening: {e}")
            self.is_listening = False
            return False
    
    async def stop_listening(self) -> Optional[str]:
        """Stop listening and process audio"""
        if not self.is_listening:
            return None
        
        try:
            self.is_listening = False
            
            if self.on_listening_stop:
                self.on_listening_stop()
            
            # Stop recording and get audio data
            audio_data = await self.audio_recorder.stop()
            
            if audio_data is None or len(audio_data) == 0:
                self.logger.warning("No audio data captured")
                return None
            
            # Process the audio
            transcript = await self._process_audio(audio_data)
            return transcript
            
        except Exception as e:
            self.logger.error(f"Failed to stop listening: {e}")
            if self.on_error:
                self.on_error(f"Audio processing error: {str(e)}")
            return None
    
    async def process_text_input(self, text: str) -> str:
        """Process text input directly (without ASR)"""
        if self.is_processing:
            return "I'm currently processing another request. Please wait."
        
        try:
            self.is_processing = True
            
            # Process with LLM
            response = await self._process_with_llm(text)
            
            # Add to conversation history
            self._add_to_history(text, response)
            
            # Synthesize speech response
            await self._speak_response(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process text input: {e}")
            error_msg = "I'm sorry, I encountered an error processing your request."
            if self.on_error:
                self.on_error(str(e))
            return error_msg
        finally:
            self.is_processing = False
    
    async def _conversation_loop(self):
        """Main conversation loop"""
        while True:
            try:
                # Check if we should start listening automatically
                if not self.is_listening and not self.is_processing and not self.is_speaking:
                    if self.config.audio.vad.enabled and self.vad:
                        # Use VAD to detect speech
                        if await self.vad.detect_speech():
                            await self.start_listening()
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in conversation loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _health_monitor(self):
        """Monitor system health and performance"""
        while True:
            try:
                # Monitor memory usage
                import psutil
                memory_percent = psutil.virtual_memory().percent
                
                if memory_percent > 90:
                    self.logger.warning(f"High memory usage: {memory_percent}%")
                
                # Monitor component health
                if self.llm and not self.llm.is_healthy():
                    self.logger.warning("LLM component health check failed")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(30)
    
    async def _process_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio data through ASR"""
        try:
            start_time = time.time()
            
            # Transcribe audio
            result = await self.asr.transcribe(audio_data)
            
            if not result or not result.get("text"):
                self.logger.warning("No text transcribed from audio")
                return None
            
            transcript = result["text"].strip()
            processing_time = time.time() - start_time
            
            self.logger.info(f"Transcribed in {processing_time:.2f}s: {transcript}")
            
            if self.on_transcript:
                self.on_transcript(transcript)
            
            # Process with LLM
            response = await self._process_with_llm(transcript)
            
            # Add to conversation history
            self._add_to_history(transcript, response)
            
            # Synthesize speech response
            await self._speak_response(response)
            
            return transcript
            
        except Exception as e:
            self.logger.error(f"Failed to process audio: {e}")
            if self.on_error:
                self.on_error(f"Audio processing error: {str(e)}")
            return None
    
    async def _process_with_llm(self, text: str) -> str:
        """Process text with LLM"""
        try:
            start_time = time.time()
            
            # Prepare context with conversation history
            context = self._build_context(text)
            
            # Generate response
            response = await self.llm.generate_response(context)
            
            processing_time = time.time() - start_time
            self.logger.info(f"LLM response generated in {processing_time:.2f}s")
            
            if self.on_response:
                self.on_response(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM processing error: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    async def _speak_response(self, text: str):
        """Synthesize and play speech response"""
        if not text:
            return
        
        try:
            self.is_speaking = True
            start_time = time.time()
            
            # Generate speech
            audio_data = await self.tts.synthesize(text)
            
            if audio_data:
                # Play audio
                await self.audio_player.play(audio_data)
                
                synthesis_time = time.time() - start_time
                self.logger.info(f"Speech synthesized and played in {synthesis_time:.2f}s")
            else:
                self.logger.warning("No audio data generated from TTS")
                
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
        finally:
            self.is_speaking = False
    
    def _build_context(self, current_input: str) -> str:
        """Build conversation context for LLM"""
        # Start with system prompt
        context = self.config.advanced.system_prompt + "\n\n"
        
        # Add recent conversation history
        max_history = 5  # Keep last 5 exchanges
        recent_history = self.conversation_history[-max_history:] if self.conversation_history else []
        
        for entry in recent_history:
            context += f"Human: {entry['input']}\n"
            context += f"Assistant: {entry['output']}\n\n"
        
        # Add current input
        context += f"Human: {current_input}\n"
        context += "Assistant: "
        
        return context
    
    def _add_to_history(self, user_input: str, assistant_output: str):
        """Add conversation to history"""
        entry = {
            "input": user_input,
            "output": assistant_output,
            "timestamp": time.time()
        }
        
        self.conversation_history.append(entry)
        
        # Limit history size
        max_history = 50
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
        
        # Save to file if enabled
        if self.config.privacy.store_conversations:
            self._save_conversation_history()
    
    def _save_conversation_history(self):
        """Save conversation history to file"""
        try:
            import json
            from datetime import datetime
            
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d")
            history_file = logs_dir / f"conversations_{timestamp}.json"
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save conversation history: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the voice assistant"""
        return {
            "is_listening": self.is_listening,
            "is_processing": self.is_processing,
            "is_speaking": self.is_speaking,
            "components": {
                "asr": self.asr is not None and self.asr.is_ready(),
                "llm": self.llm is not None and self.llm.is_ready(),
                "tts": self.tts is not None and self.tts.is_ready(),
                "audio": self.audio_recorder is not None and self.audio_player is not None
            },
            "conversation_length": len(self.conversation_history)
        }
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")
    
    def set_callbacks(self, 
                     on_listening_start: Callable = None,
                     on_listening_stop: Callable = None,
                     on_transcript: Callable[[str], None] = None,
                     on_response: Callable[[str], None] = None,
                     on_error: Callable[[str], None] = None):
        """Set callback functions for UI integration"""
        self.on_listening_start = on_listening_start
        self.on_listening_stop = on_listening_stop
        self.on_transcript = on_transcript
        self.on_response = on_response
        self.on_error = on_error
