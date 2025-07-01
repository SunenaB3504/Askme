# AskMe - Offline Privacy-Centric Voice Assistant

[![GitHub Repository](https://img.shields.io/badge/GitHub-AskMe-blue?logo=github)](https://github.com/SunenaB3504/Askme)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)

## 🚀 Quick Links
- **📁 Repository**: [https://github.com/SunenaB3504/Askme](https://github.com/SunenaB3504/Askme)
- **📖 Documentation**: [docs/](docs/)
- **🎯 Quick Start for Nia**: [QUICK_START_NIA.md](QUICK_START_NIA.md)
- **📋 GitHub Guide**: [docs/github_guide.md](docs/github_guide.md)

## Project Overview

AskMe is a fully offline, privacy-centric voice assistant that leverages a custom-made Large Language Model (LLM) to provide intelligent conversational interaction without compromising user privacy. This system operates entirely on local hardware, ensuring that user data never leaves the device.

## Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Speech   │───▶│   ASR (Whisper) │───▶│  Text Input     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Speech Output  │◀───│ TTS (Coqui TTS)│◀───│ Custom LLM      │
└─────────────────┘    └─────────────────┘    │ (Fine-tuned)    │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   OpenWebUI     │
                                              │   Interface     │
                                              └─────────────────┘
```

## Key Features

- **Complete Privacy**: All processing happens locally, no data sent to cloud
- **Multi-language Support**: English, Hindi, Tamil, and more
- **Real-time Processing**: Low-latency speech-to-text and response generation
- **Customizable**: Adaptable to specific domains and use cases
- **Open Source**: Built on open-source frameworks and models

## Quick Start

1. Set up the environment: `python setup.py install`
2. Download and configure models: `python scripts/setup_models.py`
3. Start the voice assistant: `python main.py`

## Documentation Structure

- [Installation Guide](docs/installation.md)
- [Model Training Guide](docs/model_training.md)
- [API Reference](docs/api_reference.md)
- [Performance Benchmarks](docs/performance.md)
- [Privacy & Security](docs/privacy.md)

## Project Structure

```
askme/
├── src/
│   ├── llm/           # LLM fine-tuning and inference
│   ├── asr/           # Speech recognition with Whisper
│   ├── tts/           # Text-to-speech with Coqui TTS
│   ├── ui/            # OpenWebUI integration
│   └── core/          # Core application logic
├── models/            # Trained models and configurations
├── data/              # Training datasets
├── docs/              # Documentation
├── tests/             # Unit and integration tests
├── scripts/           # Setup and utility scripts
└── configs/           # Configuration files
```

## License

MIT License - See [LICENSE](LICENSE) for details.
