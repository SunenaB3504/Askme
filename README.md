# AskMe - Educational Voice Assistant for Young Learners

[![GitHub Repository](https://img.shields.io/badge/GitHub-AskMe-blue?logo=github)](https://github.com/SunenaB3504/Askme)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Deploy to Replit](https://img.shields.io/badge/Deploy-Replit-orange?logo=replit)](https://replit.com/github/SunenaB3504/Askme)

## 🎯 Project Description

**AskMe** is an educational web application designed as a learning companion for **Nia, a 9-year-old student**. This child-friendly assistant helps with CBSE Class 4 English curriculum through interactive conversations, practice questions, and engaging learning activities.

### 🌟 Key Features
- **📚 Educational Content**: Based on CBSE Class 4 English textbook
- **🎨 Child-Friendly Interface**: Colorful, engaging design for young learners
- **💬 Interactive Q&A**: Ask questions about chapters and get helpful answers
- **🎲 Practice Questions**: 164+ randomly generated questions from curriculum
- **🔒 Privacy-First**: No data collection, safe for children
- **🌐 Web-Based**: Accessible from any device with a browser

### 🎓 Educational Value
- **Reading Comprehension**: Helps understand English literature
- **Interactive Learning**: Makes studying fun and engaging
- **Self-Paced**: Students can learn at their own speed
- **Curriculum Aligned**: Follows CBSE Class 4 English syllabus
- **Encouraging Responses**: Builds confidence in young learners

## 🚀 Quick Deploy Options

### Deploy to Replit (1-Click)
1. Click: [![Deploy to Replit](https://img.shields.io/badge/Deploy-Replit-orange?logo=replit)](https://replit.com/github/SunenaB3504/Askme)
2. Wait for automatic setup
3. Click "Run" to start the application
4. Get your public URL instantly!

### Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Connect GitHub and select this repository
3. Automatic deployment with public URL

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
