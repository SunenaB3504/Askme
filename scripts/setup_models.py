#!/usr/bin/env python3
"""
Model setup script for AskMe Voice Assistant

This script downloads and sets up the required models for the voice assistant:
- Whisper ASR models
- Coqui TTS models  
- Base LLM models (Mistral-7B or Phi-2)
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm


class ModelDownloader:
    """Handles downloading and setting up models for AskMe"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model definitions
        self.models = {
            "whisper": {
                "tiny": {
                    "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e/tiny.pt",
                    "size": "39MB",
                    "sha256": "65147644a518d12f04e32d6f3b26facc3f8dd46e"
                },
                "base": {
                    "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205/base.pt",
                    "size": "74MB", 
                    "sha256": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205"
                },
                "small": {
                    "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19a9b982f6c06d8c58e33e/small.pt",
                    "size": "244MB",
                    "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19a9b982f6c06d8c58e33e"
                }
            },
            "llm": {
                "mistral-7b": {
                    "base_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
                    "gguf_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf",
                    "size": "4.1GB",
                    "description": "Mistral 7B Instruct model (4-bit quantized)"
                },
                "phi-2": {
                    "base_url": "https://huggingface.co/microsoft/phi-2",
                    "gguf_url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf",
                    "size": "1.6GB",
                    "description": "Microsoft Phi-2 model (4-bit quantized)"
                }
            }
        }
    
    def download_file(self, url: str, destination: Path, description: str = "") -> bool:
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return False
    
    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify file checksum"""
        if not expected_sha256:
            return True  # Skip verification if no checksum provided
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()[:len(expected_sha256)] == expected_sha256
    
    def setup_whisper_models(self, models: List[str] = None) -> bool:
        """Setup Whisper ASR models"""
        if models is None:
            models = ["base"]  # Default to base model
        
        print("🎤 Setting up Whisper ASR models...")
        whisper_dir = self.models_dir / "whisper"
        whisper_dir.mkdir(exist_ok=True)
        
        success = True
        for model_name in models:
            if model_name not in self.models["whisper"]:
                print(f"❌ Unknown Whisper model: {model_name}")
                success = False
                continue
            
            model_info = self.models["whisper"][model_name]
            model_file = whisper_dir / f"{model_name}.pt"
            
            if model_file.exists():
                print(f"✓ Whisper {model_name} already exists")
                continue
            
            print(f"📥 Downloading Whisper {model_name} ({model_info['size']})...")
            if self.download_file(
                model_info["url"], 
                model_file, 
                f"Whisper {model_name}"
            ):
                # Verify checksum if provided
                if self.verify_checksum(model_file, model_info.get("sha256", "")):
                    print(f"✓ Whisper {model_name} downloaded and verified")
                else:
                    print(f"⚠️ Whisper {model_name} downloaded but checksum verification failed")
            else:
                success = False
        
        return success
    
    def setup_tts_models(self) -> bool:
        """Setup Coqui TTS models"""
        print("🗣️ Setting up Coqui TTS models...")
        
        try:
            # Import TTS to trigger model download
            from TTS.api import TTS
            
            # Initialize default TTS model (this will download it)
            print("📥 Downloading default TTS model...")
            tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            
            print("✓ TTS models ready")
            return True
        except Exception as e:
            print(f"❌ Failed to setup TTS models: {e}")
            return False
    
    def setup_llm_models(self, model_name: str = "mistral-7b") -> bool:
        """Setup LLM models"""
        if model_name not in self.models["llm"]:
            print(f"❌ Unknown LLM model: {model_name}")
            return False
        
        print(f"🧠 Setting up LLM model: {model_name}...")
        
        model_info = self.models["llm"][model_name]
        llm_dir = self.models_dir / "llm" / model_name
        llm_dir.mkdir(parents=True, exist_ok=True)
        
        # Download GGUF model for llama.cpp
        gguf_file = llm_dir / f"{model_name}.gguf"
        
        if gguf_file.exists():
            print(f"✓ {model_name} GGUF model already exists")
            return True
        
        print(f"📥 Downloading {model_name} GGUF model ({model_info['size']})...")
        if self.download_file(
            model_info["gguf_url"],
            gguf_file,
            f"{model_name} GGUF"
        ):
            print(f"✓ {model_name} model downloaded successfully")
            return True
        else:
            return False
    
    def setup_all_models(self, 
                        whisper_models: List[str] = None,
                        llm_model: str = "mistral-7b") -> bool:
        """Setup all required models"""
        print("🚀 Setting up AskMe models...")
        
        success = True
        
        # Setup Whisper models
        if not self.setup_whisper_models(whisper_models):
            success = False
        
        # Setup TTS models
        if not self.setup_tts_models():
            success = False
        
        # Setup LLM model
        if not self.setup_llm_models(llm_model):
            success = False
        
        return success
    
    def list_available_models(self):
        """List all available models"""
        print("📋 Available models:")
        print("\n🎤 Whisper ASR models:")
        for name, info in self.models["whisper"].items():
            print(f"  - {name}: {info['size']}")
        
        print("\n🧠 LLM models:")
        for name, info in self.models["llm"].items():
            print(f"  - {name}: {info['size']} - {info['description']}")
        
        print("\n🗣️ TTS models:")
        print("  - tacotron2-DDC: Default English TTS model")
        print("  - Additional models available through Coqui TTS")
    
    def check_existing_models(self):
        """Check which models are already downloaded"""
        print("🔍 Checking existing models...")
        
        # Check Whisper models
        whisper_dir = self.models_dir / "whisper"
        if whisper_dir.exists():
            whisper_files = list(whisper_dir.glob("*.pt"))
            if whisper_files:
                print(f"✓ Found Whisper models: {[f.stem for f in whisper_files]}")
            else:
                print("❌ No Whisper models found")
        else:
            print("❌ Whisper models directory not found")
        
        # Check LLM models
        llm_dir = self.models_dir / "llm"
        if llm_dir.exists():
            gguf_files = list(llm_dir.glob("**/*.gguf"))
            if gguf_files:
                print(f"✓ Found LLM models: {[f.parent.name for f in gguf_files]}")
            else:
                print("❌ No LLM models found")
        else:
            print("❌ LLM models directory not found")
        
        # Check TTS models (these are managed by Coqui TTS)
        try:
            from TTS.api import TTS
            print("✓ TTS system available")
        except ImportError:
            print("❌ TTS system not available (install TTS package)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Setup models for AskMe Voice Assistant"
    )
    
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Directory to store models"
    )
    
    parser.add_argument(
        "--whisper-models",
        nargs="+",
        default=["base"],
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper models to download"
    )
    
    parser.add_argument(
        "--llm-model",
        default="mistral-7b",
        choices=["mistral-7b", "phi-2"],
        help="LLM model to download"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check existing models"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download existing models"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.list:
        downloader.list_available_models()
        return 0
    
    if args.check:
        downloader.check_existing_models()
        return 0
    
    # Setup models
    success = downloader.setup_all_models(
        whisper_models=args.whisper_models,
        llm_model=args.llm_model
    )
    
    if success:
        print("\n🎉 All models setup successfully!")
        print("\nNext steps:")
        print("1. Copy configs/config.example.yaml to configs/config.yaml")
        print("2. Update model paths in the configuration file")
        print("3. Run: python main.py")
        return 0
    else:
        print("\n❌ Some models failed to setup. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
