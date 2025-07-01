"""
LLaMA inference engine for AskMe Voice Assistant
"""

import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator
import threading
import queue
import time

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False


class LlamaInference:
    """LLaMA model inference engine using llama.cpp"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.is_initialized = False
        self._inference_lock = threading.Lock()
        
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available")
    
    async def initialize(self):
        """Initialize LLaMA model"""
        try:
            model_path = self._get_model_path()
            self.logger.info(f"Loading LLaMA model from: {model_path}")
            
            # Determine number of GPU layers
            n_gpu_layers = 0
            if self.config.device in ["cuda", "auto"]:
                try:
                    import torch
                    if torch.cuda.is_available():
                        n_gpu_layers = self.config.gpu_layers
                except ImportError:
                    pass
            
            # Initialize model
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.config.context_length,
                n_batch=self.config.batch_size,
                n_threads=self.config.threads if self.config.threads > 0 else None,
                n_gpu_layers=n_gpu_layers,
                use_mmap=True,
                use_mlock=False,
                verbose=False
            )
            
            self.is_initialized = True
            self.logger.info(f"✓ LLaMA model loaded (GPU layers: {n_gpu_layers})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLaMA model: {e}")
            raise
    
    def _get_model_path(self) -> str:
        """Get the full path to the model file"""
        import os
        from pathlib import Path
        
        # Check if path is absolute
        if os.path.isabs(self.config.path):
            model_dir = Path(self.config.path)
        else:
            model_dir = Path(self.config.path)
        
        # If model_file is specified, use it
        if hasattr(self.config, 'model_file') and self.config.model_file:
            model_path = model_dir / self.config.model_file
        else:
            # Look for .gguf files
            gguf_files = list(model_dir.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {model_dir}")
            
            # Use the first GGUF file found
            model_path = gguf_files[0]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return str(model_path)
    
    async def generate_response(self, prompt: str) -> str:
        """Generate response for given prompt"""
        if not self.is_initialized:
            raise RuntimeError("LLaMA model not initialized")
        
        try:
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self._generate_sync, 
                prompt
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return "I apologize, but I encountered an error generating a response."
    
    def _generate_sync(self, prompt: str) -> str:
        """Synchronous response generation"""
        with self._inference_lock:
            try:
                start_time = time.time()
                
                # Generate response
                output = self.model(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repeat_penalty=1.1,
                    stop=["Human:", "User:", "\n\n"],
                    echo=False
                )
                
                response_text = output["choices"][0]["text"].strip()
                
                generation_time = time.time() - start_time
                token_count = len(output["choices"][0]["text"].split())
                tokens_per_second = token_count / generation_time if generation_time > 0 else 0
                
                self.logger.debug(f"Generated {token_count} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
                
                return self._post_process_response(response_text)
                
            except Exception as e:
                self.logger.error(f"Sync generation failed: {e}")
                return "I'm sorry, I encountered an error processing your request."
    
    async def generate_response_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        if not self.is_initialized:
            raise RuntimeError("LLaMA model not initialized")
        
        # Create a queue for streaming tokens
        token_queue = queue.Queue()
        
        def generate_stream():
            """Generate tokens in a separate thread"""
            try:
                with self._inference_lock:
                    stream = self.model(
                        prompt,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        repeat_penalty=1.1,
                        stop=["Human:", "User:", "\n\n"],
                        echo=False,
                        stream=True
                    )
                    
                    for token in stream:
                        text = token["choices"][0]["text"]
                        token_queue.put(text)
                    
                    token_queue.put(None)  # End marker
                    
            except Exception as e:
                token_queue.put(f"Error: {e}")
                token_queue.put(None)
        
        # Start generation in thread
        thread = threading.Thread(target=generate_stream)
        thread.start()
        
        # Yield tokens as they arrive
        try:
            while True:
                try:
                    token = token_queue.get(timeout=30)  # 30 second timeout
                    if token is None:  # End marker
                        break
                    if token.startswith("Error:"):
                        self.logger.error(token)
                        break
                    yield token
                except queue.Empty:
                    self.logger.warning("Token generation timeout")
                    break
        finally:
            thread.join(timeout=5)
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response"""
        # Remove any unwanted prefixes/suffixes
        response = response.strip()
        
        # Remove common AI assistant prefixes
        prefixes_to_remove = [
            "Assistant:", "AI:", "Response:", "Answer:",
            "I am AskMe.", "As AskMe,", "As an AI,"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Apply response filters if configured
        if hasattr(self.config, 'response_filters'):
            response = self._apply_response_filters(response)
        
        return response
    
    def _apply_response_filters(self, response: str) -> str:
        """Apply configured response filters"""
        filters = self.config.response_filters
        
        # Length filter
        if len(response) > filters.max_length:
            # Try to cut at sentence boundary
            sentences = response.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) < filters.max_length:
                    truncated += sentence + ". "
                else:
                    break
            response = truncated.strip()
        
        # Additional filters can be added here
        # - Profanity filter
        # - Personal information filter
        # - Content safety filters
        
        return response
    
    def is_ready(self) -> bool:
        """Check if LLM is ready"""
        return self.is_initialized and self.model is not None
    
    def is_healthy(self) -> bool:
        """Health check for the LLM"""
        if not self.is_ready():
            return False
        
        try:
            # Quick health check with minimal prompt
            test_output = self.model("Hi", max_tokens=1, temperature=0.1)
            return "choices" in test_output and len(test_output["choices"]) > 0
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        if not self.is_ready():
            return {"status": "not_ready"}
        
        return {
            "status": "ready",
            "model_path": self._get_model_path(),
            "context_length": self.config.context_length,
            "device": self.config.device,
            "gpu_layers": getattr(self.config, 'gpu_layers', 0),
            "is_healthy": self.is_healthy()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_initialized = False
        self.logger.info("LLaMA model cleaned up")
