# Installation Guide - AskMe Voice Assistant

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 20.04+
- **CPU**: 4-core processor (Intel i5-8400 / AMD Ryzen 5 2600 or equivalent)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space (SSD recommended)
- **Audio**: Microphone and speakers/headphones

### Recommended Requirements
- **CPU**: 8-core processor with AVX2 support
- **RAM**: 32GB for optimal performance
- **Storage**: NVMe SSD with 50GB free space
- **GPU**: RTX 3060+ (optional, for faster training)
- **Audio**: Noise-canceling microphone

## Pre-installation Setup

### 1. Install Python 3.9+

#### Windows
```powershell
# Using winget
winget install Python.Python.3.11

# Or download from python.org
# https://www.python.org/downloads/windows/
```

#### macOS
```bash
# Using Homebrew
brew install python@3.11

# Or using pyenv
pyenv install 3.11.7
pyenv global 3.11.7
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv
```

### 2. Install Git
```bash
# Windows (using winget)
winget install Git.Git

# macOS
brew install git

# Linux
sudo apt install git
```

### 3. Install System Dependencies

#### Windows
```powershell
# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools

# Install PortAudio for pyaudio
# Download from: http://www.portaudio.com/download.html
```

#### macOS
```bash
# Install PortAudio
brew install portaudio

# Install cmake
brew install cmake
```

#### Linux
```bash
sudo apt install -y \
    build-essential \
    cmake \
    portaudio19-dev \
    python3-dev \
    ffmpeg \
    libsndfile1
```

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/askme-voice-assistant.git
cd askme-voice-assistant
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv askme_env

# Activate environment
# Windows
askme_env\Scripts\activate

# macOS/Linux
source askme_env/bin/activate
```

### 3. Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Install PyTorch (with appropriate CUDA support)

#### CPU Only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### CUDA 11.8 (for NVIDIA GPUs)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1 (for newer NVIDIA GPUs)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install Additional Components

#### Install Whisper
```bash
pip install openai-whisper
```

#### Install Coqui TTS
```bash
pip install TTS
```

#### Install llama.cpp Python bindings
```bash
pip install llama-cpp-python
```

## Model Setup

### 1. Download Base Models

Run the setup script to download required models:
```bash
python scripts/setup_models.py
```

This will download:
- Whisper base model (~74MB)
- Coqui TTS model (~100MB)
- Base LLM model (Mistral-7B or Phi-2)

### 2. Verify Installation

```bash
python scripts/verify_installation.py
```

Expected output:
```
✓ Python environment: OK
✓ PyTorch installation: OK
✓ Whisper model: OK
✓ TTS model: OK
✓ LLM model: OK
✓ Audio system: OK
✓ All systems ready!
```

## Configuration

### 1. Basic Configuration

Copy the example configuration:
```bash
cp configs/config.example.yaml configs/config.yaml
```

Edit `configs/config.yaml`:
```yaml
# Model configurations
models:
  llm:
    name: "mistral-7b-instruct"
    path: "./models/mistral-7b-instruct-gguf"
    context_length: 4096
    temperature: 0.7
  
  asr:
    model: "base"
    language: "en"
    
  tts:
    model: "tts_models/en/ljspeech/tacotron2-DDC"
    speaker: null

# Audio settings
audio:
  sample_rate: 16000
  chunk_size: 1024
  channels: 1
  
# UI settings
ui:
  host: "localhost"
  port: 8080
  theme: "dark"
```

### 2. Advanced Configuration

For custom model paths, specialized settings, or domain-specific configurations, see [Configuration Guide](configuration.md).

## Running the Application

### 1. Start the Voice Assistant

```bash
# Start with default configuration
python main.py

# Start with custom configuration
python main.py --config configs/custom_config.yaml

# Start in development mode
python main.py --dev
```

### 2. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8080
```

### 3. Test Voice Functionality

1. Click the "Start Listening" button
2. Say "Hello, can you hear me?"
3. Wait for transcription and response
4. Listen to the synthesized response

## Troubleshooting

### Common Issues

#### 1. Audio Device Not Found
```bash
# Windows: Install Windows audio drivers
# macOS: Check System Preferences > Security & Privacy > Microphone
# Linux: Install PulseAudio
sudo apt install pulseaudio pulseaudio-utils
```

#### 2. CUDA Out of Memory
```yaml
# In config.yaml, reduce batch size or use CPU
models:
  llm:
    device: "cpu"  # Change from "cuda" to "cpu"
```

#### 3. Slow Response Times
```yaml
# Use smaller models
models:
  llm:
    name: "phi-2"  # Smaller than Mistral-7B
  asr:
    model: "tiny"  # Faster than "base"
```

#### 4. Permission Errors (Windows)
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Performance Optimization

#### 1. Enable Hardware Acceleration

For NVIDIA GPUs:
```yaml
models:
  llm:
    device: "cuda"
    gpu_layers: 35  # Adjust based on VRAM
```

For Apple Silicon Macs:
```yaml
models:
  llm:
    device: "mps"
```

#### 2. Optimize for Low-Memory Systems

```yaml
models:
  llm:
    context_length: 2048  # Reduce from 4096
    n_batch: 256  # Reduce batch size
```

## Development Setup

### 1. Install Development Tools

```bash
pip install -r requirements-dev.txt
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test category
pytest tests/test_asr.py
```

### 4. Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/
```

## Next Steps

1. **Basic Usage**: See [User Guide](user_guide.md)
2. **Customization**: See [Model Training Guide](model_training.md)
3. **API Integration**: See [API Reference](api_reference.md)
4. **Performance Tuning**: See [Performance Guide](performance.md)

## Support

For installation issues:
1. Check the [FAQ](faq.md)
2. Search [GitHub Issues](https://github.com/yourusername/askme-voice-assistant/issues)
3. Create a new issue with:
   - Operating system details
   - Hardware specifications
   - Full error messages
   - Configuration file (remove sensitive data)
