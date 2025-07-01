# Usage Guide - AskMe Voice Assistant

## Overview

This guide covers how to use the AskMe voice assistant once it's installed and configured. AskMe is a privacy-focused, offline voice assistant that runs entirely on your local machine.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Voice Interaction](#voice-interaction)
3. [Web Interface](#web-interface)
4. [Text-Only Mode](#text-only-mode)
5. [Configuration Options](#configuration-options)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

## Quick Start

### 1. Start the Assistant

```bash
# Navigate to the AskMe directory
cd c:\Users\Admin\AI-training\Askme

# Start the voice assistant
python main.py
```

### 2. Access the Web Interface

Once started, open your web browser and go to:
```
http://localhost:8000
```

### 3. Begin Interaction

- **Voice Mode**: Click the microphone button and speak
- **Text Mode**: Type your message in the chat interface
- **Push-to-Talk**: Hold spacebar while speaking (configurable)

## Voice Interaction

### Basic Voice Commands

#### Starting a Conversation
- "Hello AskMe"
- "Hey assistant"
- "Can you help me?"

#### Getting Information
- "What's the weather like?" (Note: Offline assistant will explain limitations)
- "Explain quantum physics"
- "How do I cook pasta?"
- "Tell me about machine learning"

#### Task Assistance
- "Help me plan my day"
- "Give me ideas for dinner"
- "How can I improve my productivity?"
- "Suggest some creative writing prompts"

#### General Conversation
- "How are you today?"
- "Tell me a joke"
- "What can you do?"
- "Help me brainstorm ideas for..."

### Voice Control Features

#### Microphone Control
```javascript
// The web interface provides these controls:
- Click microphone: Start/stop recording
- Push-to-talk: Hold spacebar (default)
- Voice activity detection: Automatic start/stop
- Mute: Disable microphone input
```

#### Audio Settings
- **Input Gain**: Adjust microphone sensitivity
- **Noise Suppression**: Enable/disable background noise filtering
- **Voice Activity Threshold**: Set sensitivity for automatic recording

## Web Interface

### Main Interface Components

#### 1. Chat Window
- **Message History**: Scrollable conversation log
- **User Messages**: Your input (text/transcribed speech)
- **Assistant Responses**: AskMe's replies
- **Timestamps**: Message timing information

#### 2. Control Panel
```
┌─────────────────────────────────────┐
│  🎤 Microphone    🔊 Speaker        │
│  ⚙️ Settings      📊 Status         │
│  💾 Export        🗑️ Clear          │
└─────────────────────────────────────┘
```

#### 3. Input Methods
- **Text Input**: Type messages directly
- **Voice Input**: Click microphone button
- **File Upload**: Send audio files for transcription

### Interface Features

#### Chat Controls
```html
<!-- Message export options -->
Export Conversation:
- Plain Text (.txt)
- JSON format (.json)  
- Audio recordings (.wav)

<!-- Clear options -->
Clear:
- Current session only
- All conversation history
- Audio cache
```

#### Settings Panel
```yaml
Audio Settings:
  microphone_gain: 0.8
  noise_suppression: true
  voice_activity_detection: true
  push_to_talk_key: "space"

Display Settings:
  theme: "dark"  # dark, light, auto
  font_size: "medium"
  show_timestamps: true
  auto_scroll: true

Privacy Settings:
  save_conversations: false
  save_audio: false
  encryption_enabled: true
```

## Text-Only Mode

### Starting Text Mode
```bash
# Start without voice components (faster startup)
python main.py --text-only

# Or disable voice in web interface
# Settings → Audio → Disable Voice Input
```

### Text Commands
All voice commands work as text input:

```
User: How do I bake a chocolate cake?

AskMe: I'd be happy to help you bake a chocolate cake! Here's a simple recipe:

**Ingredients:**
- 1¾ cups all-purpose flour
- 2 cups sugar
- ¾ cup cocoa powder
- 2 teaspoons baking soda
- 1 teaspoon baking powder
- 1 teaspoon salt
- 2 eggs
- 1 cup buttermilk
- 1 cup strong black coffee (cooled)
- ½ cup vegetable oil
- 1 teaspoon vanilla extract

**Instructions:**
1. Preheat oven to 350°F (175°C)
2. Mix dry ingredients in a large bowl
3. Combine wet ingredients separately
4. Gradually mix wet into dry ingredients
5. Pour into greased 9x13 pan
6. Bake 30-35 minutes until toothpick comes out clean
7. Cool completely before frosting

Would you like a frosting recipe to go with it?
```

## Configuration Options

### Basic Configuration
Edit `configs/config.yaml`:

```yaml
# Assistant behavior
assistant:
  name: "AskMe"
  personality: "helpful"  # helpful, casual, professional
  response_length: "medium"  # short, medium, long
  creativity: 0.7  # 0.0-1.0

# Audio settings
audio:
  sample_rate: 16000
  chunk_size: 1024
  channels: 1
  
# Model settings  
llm:
  model_path: "./models/askme-mistral-7b"
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
```

### Advanced Configuration

#### Custom Wake Words
```yaml
wake_words:
  enabled: true
  phrases: ["Hey AskMe", "Hello Assistant"]
  sensitivity: 0.7
  timeout: 5  # seconds
```

#### Response Customization
```yaml
responses:
  greeting: "Hello! I'm AskMe, your offline assistant. How can I help?"
  goodbye: "Goodbye! Remember, your privacy is always protected with me."
  error: "I apologize, but I encountered an issue. Could you try again?"
  unknown: "I'm not sure about that. Could you rephrase your question?"
```

## Common Use Cases

### 1. Learning and Education
```
Examples:
"Explain photosynthesis in simple terms"
"Help me understand calculus derivatives"
"What are the main causes of World War I?"
"Teach me about programming concepts"
```

### 2. Creative Writing
```
Examples:
"Give me a story prompt about space exploration"
"Help me brainstorm character ideas for my novel"
"Suggest rhymes for the word 'mountain'"
"What's a good opening line for a mystery story?"
```

### 3. Problem Solving
```
Examples:
"I'm having trouble organizing my schedule"
"Help me think through this decision about job offers"
"What are some ways to reduce stress?"
"How can I improve my study habits?"
```

### 4. General Knowledge
```
Examples:
"How do solar panels work?"
"What are the benefits of meditation?"
"Explain blockchain technology"
"What makes a good leader?"
```

### 5. Practical Assistance
```
Examples:
"Help me plan a healthy meal"
"What are some exercise routines for beginners?"
"How do I troubleshoot my computer running slowly?"
"Give me tips for public speaking"
```

## Troubleshooting

### Common Issues

#### 1. Microphone Not Working
```
Symptoms: No audio input detected
Solutions:
1. Check microphone permissions in browser
2. Verify microphone is not muted
3. Test with: Settings → Audio → Test Microphone
4. Restart the browser/application
5. Check Windows audio settings
```

#### 2. Assistant Not Responding
```
Symptoms: No text response generated
Solutions:
1. Check model status: Status panel should show "Model Loaded"
2. Verify sufficient RAM available (6GB+ recommended)
3. Check logs: python main.py --debug
4. Restart the assistant
```

#### 3. Slow Response Times
```
Symptoms: Long delays between input and response
Solutions:
1. Reduce max_tokens in config (try 256)
2. Close other applications to free memory
3. Use quantized model (Q4 or Q8)
4. Check CPU usage in task manager
```

#### 4. Audio Quality Issues
```
Symptoms: Poor transcription accuracy
Solutions:
1. Improve microphone positioning
2. Reduce background noise
3. Adjust input gain in settings
4. Enable noise suppression
5. Speak clearly and at moderate pace
```

### Performance Optimization

#### For Slower Computers
```yaml
# config.yaml optimizations
llm:
  model_path: "./models/askme-mistral-7b-q4.gguf"  # Use quantized model
  max_tokens: 256  # Reduce response length
  context_length: 1024  # Reduce context window

audio:
  chunk_size: 2048  # Larger chunks, less processing
  
ui:
  enable_animations: false  # Disable UI animations
```

#### For Better Performance
```yaml
# High-performance settings
llm:
  model_path: "./models/askme-mistral-7b-fp16"
  max_tokens: 512
  batch_size: 8  # If using GPU
  
audio:
  sample_rate: 48000  # Higher quality audio
  noise_suppression: true
```

## Advanced Features

### 1. Conversation Memory
```
The assistant maintains context within a session:

User: "My name is Sarah and I love hiking"
AskMe: "Nice to meet you, Sarah! Hiking is wonderful..."

User: "What should I pack for a day hike?"
AskMe: "For your day hike, Sarah, I'd recommend..."
```

### 2. Multi-turn Conversations
```
User: "I want to learn to cook"
AskMe: "That's great! What type of cuisine interests you most?"

User: "Italian food"
AskMe: "Excellent choice! Let's start with some basic Italian dishes..."

User: "What about pasta?"
AskMe: "Perfect! Pasta is fundamental to Italian cooking..."
```

### 3. Export and Import
```bash
# Export conversation history
python scripts/export_conversations.py --format json --output conversations.json

# Import previous conversations
python scripts/import_conversations.py --input conversations.json
```

### 4. Voice Profiles
```bash
# Create custom voice profile
python scripts/create_voice_profile.py --name "my_voice" --samples ./voice_samples/

# Use custom voice for TTS
# Edit config.yaml:
tts:
  voice_profile: "my_voice"
```

### 5. Offline Operation Verification
```
To verify complete offline operation:
1. Disconnect from internet
2. Start AskMe: python main.py
3. Test all functions (voice, text, TTS)
4. Everything should work normally

The assistant will remind you it's offline when asked about:
- Current weather
- Latest news  
- Real-time information
- Internet searches
```

### 6. Privacy Dashboard
Access via web interface: `Settings → Privacy`

```
Privacy Status:
✅ All processing done locally
✅ No data sent to external servers
✅ Conversation encryption enabled
✅ No tracking or analytics
✅ Audio automatically deleted after processing

Storage Information:
- Model files: 4.2 GB
- Conversation cache: 12 MB
- Audio cache: 0 MB (auto-deleted)
- Configuration: 2 KB
```

## Keyboard Shortcuts

### Web Interface Shortcuts
```
Ctrl + Enter: Send message
Space (hold): Push-to-talk
Ctrl + M: Toggle microphone
Ctrl + K: Clear conversation
Ctrl + E: Export conversation
Ctrl + S: Open settings
Esc: Stop current operation
```

### Terminal Shortcuts (when running in console mode)
```
Ctrl + C: Stop assistant
Ctrl + D: Exit gracefully
Tab: Auto-complete commands
Up/Down: Navigate command history
```

## Getting Help

### Built-in Help
```
Ask the assistant directly:
"What can you help me with?"
"How do I use your features?"
"What are your capabilities?"
"Show me available commands"
```

### Logs and Debugging
```bash
# Enable detailed logging
python main.py --debug --log-level DEBUG

# View logs
tail -f logs/askme.log

# Check system status
python scripts/system_check.py
```

### Performance Monitoring
```
Access via web interface: Status → Performance

Real-time metrics:
- Memory usage
- CPU utilization  
- Response times
- Model temperature
- Audio levels
```

This usage guide should help you get the most out of your AskMe voice assistant while maintaining complete privacy and offline operation.
