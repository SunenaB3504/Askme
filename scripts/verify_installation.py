#!/usr/bin/env python3
"""
Installation verification script for AskMe Voice Assistant

This script verifies that all components are properly installed and configured.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
import platform
import psutil


class InstallationVerifier:
    """Verifies AskMe installation"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def check_python_version(self) -> bool:
        """Check Python version"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            self.results["python"] = f"✓ Python {version.major}.{version.minor}.{version.micro}"
            return True
        else:
            self.results["python"] = f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)"
            self.errors.append("Python 3.9+ required")
            return False
    
    def check_package_installation(self, package_name: str, import_name: str = None) -> bool:
        """Check if a Python package is installed"""
        import_name = import_name or package_name
        
        try:
            importlib.import_module(import_name)
            self.results[package_name] = f"✓ {package_name} installed"
            return True
        except ImportError:
            self.results[package_name] = f"❌ {package_name} not installed"
            self.errors.append(f"Missing package: {package_name}")
            return False
    
    def check_torch_installation(self) -> bool:
        """Check PyTorch installation and device support"""
        try:
            import torch
            version = torch.__version__
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_info = f"CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)"
            else:
                device_info = "CPU only"
            
            # Check MPS availability (Apple Silicon)
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if mps_available:
                device_info += ", MPS available"
            
            self.results["pytorch"] = f"✓ PyTorch {version} ({device_info})"
            return True
            
        except ImportError:
            self.results["pytorch"] = "❌ PyTorch not installed"
            self.errors.append("PyTorch not installed")
            return False
    
    def check_audio_system(self) -> bool:
        """Check audio system availability"""
        try:
            import pyaudio
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Check for input devices
            input_devices = []
            output_devices = []
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append(info['name'])
                if info['maxOutputChannels'] > 0:
                    output_devices.append(info['name'])
            
            p.terminate()
            
            if input_devices and output_devices:
                self.results["audio"] = f"✓ Audio system ({len(input_devices)} input, {len(output_devices)} output devices)"
                return True
            else:
                self.results["audio"] = "❌ No audio devices found"
                self.errors.append("No audio devices available")
                return False
                
        except Exception as e:
            self.results["audio"] = f"❌ Audio system error: {str(e)}"
            self.errors.append(f"Audio system error: {str(e)}")
            return False
    
    def check_whisper_models(self) -> bool:
        """Check Whisper model availability"""
        try:
            import whisper
            
            # Try to load a model
            model = whisper.load_model("base")
            self.results["whisper"] = "✓ Whisper models available"
            return True
            
        except Exception as e:
            self.results["whisper"] = f"❌ Whisper error: {str(e)}"
            self.errors.append(f"Whisper error: {str(e)}")
            return False
    
    def check_tts_models(self) -> bool:
        """Check TTS model availability"""
        try:
            from TTS.api import TTS
            
            # Try to initialize TTS
            tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            self.results["tts"] = "✓ TTS models available"
            return True
            
        except Exception as e:
            self.results["tts"] = f"❌ TTS error: {str(e)}"
            self.errors.append(f"TTS error: {str(e)}")
            return False
    
    def check_llm_models(self) -> bool:
        """Check LLM model availability"""
        try:
            from llama_cpp import Llama
            
            # Check for model files
            model_dirs = [
                Path("./models/llm"),
                Path("./models"),
                Path("../models")
            ]
            
            gguf_files = []
            for model_dir in model_dirs:
                if model_dir.exists():
                    gguf_files.extend(model_dir.glob("**/*.gguf"))
            
            if gguf_files:
                self.results["llm"] = f"✓ LLM models found ({len(gguf_files)} GGUF files)"
                return True
            else:
                self.results["llm"] = "❌ No LLM models found"
                self.errors.append("No LLM models found")
                return False
                
        except Exception as e:
            self.results["llm"] = f"❌ LLM error: {str(e)}"
            self.errors.append(f"LLM error: {str(e)}")
            return False
    
    def check_configuration(self) -> bool:
        """Check configuration file"""
        config_files = [
            Path("configs/config.yaml"),
            Path("config.yaml"),
            Path("configs/config.example.yaml")
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    self.results["config"] = f"✓ Configuration file: {config_file}"
                    return True
                except Exception as e:
                    self.results["config"] = f"❌ Invalid config file: {str(e)}"
                    self.errors.append(f"Invalid config file: {str(e)}")
                    return False
        
        self.results["config"] = "❌ No configuration file found"
        self.errors.append("No configuration file found")
        return False
    
    def check_system_resources(self) -> bool:
        """Check system resources"""
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # CPU check
        cpu_count = psutil.cpu_count()
        
        # Disk space check
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        
        issues = []
        if memory_gb < 8:
            issues.append("Less than 8GB RAM")
        if cpu_count < 4:
            issues.append("Less than 4 CPU cores")
        if disk_free_gb < 10:
            issues.append("Less than 10GB free disk space")
        
        if issues:
            self.results["resources"] = f"⚠️ Resource constraints: {', '.join(issues)}"
        else:
            self.results["resources"] = f"✓ Resources OK ({memory_gb:.1f}GB RAM, {cpu_count} cores, {disk_free_gb:.1f}GB free)"
        
        return len(issues) == 0
    
    def run_all_checks(self) -> bool:
        """Run all verification checks"""
        print("🔍 Verifying AskMe Voice Assistant installation...\n")
        
        checks = [
            ("Python Environment", self.check_python_version),
            ("PyTorch", self.check_torch_installation),
            ("Core Packages", lambda: all([
                self.check_package_installation("yaml", "yaml"),
                self.check_package_installation("numpy"),
                self.check_package_installation("requests"),
                self.check_package_installation("tqdm"),
                self.check_package_installation("fastapi"),
                self.check_package_installation("uvicorn"),
            ])),
            ("Audio System", self.check_audio_system),
            ("Whisper ASR", self.check_whisper_models),
            ("TTS Models", self.check_tts_models),
            ("LLM Models", self.check_llm_models),
            ("Configuration", self.check_configuration),
            ("System Resources", self.check_system_resources),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"Checking {check_name}...")
            try:
                result = check_func()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"❌ {check_name}: Error - {str(e)}")
                all_passed = False
        
        return all_passed
    
    def print_results(self):
        """Print verification results"""
        print("\n" + "="*60)
        print("VERIFICATION RESULTS")
        print("="*60)
        
        for component, result in self.results.items():
            print(f"{component:15}: {result}")
        
        if self.errors:
            print("\n" + "!"*60)
            print("ISSUES FOUND")
            print("!"*60)
            for error in self.errors:
                print(f"• {error}")
            
            print("\nSUGGESTED FIXES:")
            print("1. Run: pip install -r requirements.txt")
            print("2. Run: python scripts/setup_models.py")
            print("3. Copy configs/config.example.yaml to configs/config.yaml")
            print("4. Check audio device connections")
        else:
            print("\n🎉 All checks passed! AskMe is ready to use.")
            print("\nTo start the voice assistant:")
            print("  python main.py")
    
    def generate_system_info(self):
        """Generate system information report"""
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        
        print(f"Operating System: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        print(f"Python Version: {platform.python_version()}")
        print(f"Python Executable: {sys.executable}")
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        
        # CPU info
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        
        # GPU info (if available)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        except:
            pass


def main():
    """Main function"""
    verifier = InstallationVerifier()
    
    # Run system info
    verifier.generate_system_info()
    
    # Run verification
    success = verifier.run_all_checks()
    
    # Print results
    verifier.print_results()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
