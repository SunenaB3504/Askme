"""
🎤 Nia's Voice-Enabled Learning Assistant
Enhanced web version with browser voice support
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import os
from pathlib import Path

app = FastAPI(title="Nia's Voice Learning Assistant")

# Load educational data
def load_training_data():
    data_path = Path("data/nia_english/processed_chapters.json")
    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

chapters = load_training_data()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎤 Nia's Voice Learning Assistant</title>
        <style>
            body { 
                font-family: 'Comic Sans MS', cursive; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0; 
                padding: 20px; 
                color: white;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.1); 
                border-radius: 20px; 
                padding: 30px; 
                backdrop-filter: blur(10px);
            }
            h1 { 
                text-align: center; 
                font-size: 2.5em; 
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .chat-container { 
                height: 400px; 
                overflow-y: auto; 
                border: 2px solid rgba(255,255,255,0.3); 
                border-radius: 15px; 
                padding: 20px; 
                margin: 20px 0; 
                background: rgba(255,255,255,0.05);
            }
            .input-container { 
                display: flex; 
                gap: 10px; 
                margin: 20px 0; 
            }
            input { 
                flex: 1; 
                padding: 15px; 
                border: none; 
                border-radius: 25px; 
                font-size: 1.1em; 
                background: rgba(255,255,255,0.9);
            }
            button { 
                padding: 15px 25px; 
                border: none; 
                border-radius: 25px; 
                font-size: 1.1em; 
                cursor: pointer; 
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .send-btn { 
                background: #4CAF50; 
                color: white; 
            }
            .voice-btn { 
                background: #ff9800; 
                color: white; 
                min-width: 120px;
            }
            .voice-btn:hover { 
                background: #f57c00; 
                transform: scale(1.05);
            }
            .voice-btn.listening { 
                background: #f44336; 
                animation: pulse 1s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            .message { 
                margin: 10px 0; 
                padding: 15px; 
                border-radius: 15px; 
                line-height: 1.6;
            }
            .user { 
                background: #e3f2fd; 
                text-align: right; 
                border-bottom-right-radius: 5px;
                color: #333;
            }
            .assistant { 
                background: #f1f8e9; 
                border-bottom-left-radius: 5px;
                color: #333;
            }
            .voice-status {
                text-align: center;
                margin: 10px 0;
                font-size: 1.1em;
                min-height: 25px;
            }
            .chapters { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px; 
                margin: 20px 0; 
            }
            .chapter-card { 
                background: rgba(255,255,255,0.2); 
                padding: 15px; 
                border-radius: 10px; 
                cursor: pointer; 
                transition: all 0.3s ease;
                text-align: center;
            }
            .chapter-card:hover { 
                background: rgba(255,255,255,0.3); 
                transform: translateY(-3px);
            }
            .controls {
                display: flex;
                gap: 10px;
                justify-content: center;
                margin: 20px 0;
            }
            .control-btn {
                background: #9c27b0;
                color: white;
                padding: 10px 20px;
                border-radius: 20px;
                border: none;
                cursor: pointer;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Nia's Voice Learning Assistant 🌟</h1>
            
            <div style="text-align: center; margin: 20px 0;">
                <p style="font-size: 1.2em;">Hi Nia! I'm your voice-enabled learning helper! 🎤✨</p>
                <p>You can type OR speak to me! I'll read my answers out loud too! 📚🔊</p>
            </div>

            <div class="voice-status" id="voiceStatus">
                Ready to listen! Click the microphone button and speak. 🎤
            </div>

            <div class="controls">
                <button class="control-btn" onclick="getRandomQuestion()">🎲 Random Question</button>
                <button class="control-btn" onclick="getEncouragement()">⭐ Encouragement</button>
                <button class="control-btn" onclick="toggleVoiceOutput()">🔊 Voice: ON</button>
            </div>

            <div class="chat-container" id="chatContainer">
                <div class="message assistant">
                    👋 Hello Nia! I'm so excited to help you learn today! You can ask me about any of your English chapters, or I can give you practice questions. 
                    <br><br>🎤 <strong>Voice Feature:</strong> Click the microphone button and speak your question!
                    <br>🔊 <strong>Listen:</strong> I'll read all my answers out loud to you!
                </div>
            </div>

            <div class="input-container">
                <input type="text" id="userInput" placeholder="Type your question here, or use the microphone! 😊" 
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="voice-btn" id="voiceBtn" onclick="toggleVoiceInput()">🎤 Speak</button>
                <button class="send-btn" onclick="sendMessage()">Send 📤</button>
            </div>

            <div class="chapters">
                <div class="chapter-card" onclick="askAboutChapter('River Bank')">🏞️ The River Bank</div>
                <div class="chapter-card" onclick="askAboutChapter('Champa Flower')">🌸 The Champa Flower</div>
                <div class="chapter-card" onclick="askAboutChapter('Animals Talk')">🐾 How Animals Talk</div>
                <div class="chapter-card" onclick="askAboutChapter('Grandmother Read')">👵 Teaching Grandmother</div>
                <div class="chapter-card" onclick="askAboutChapter('Little Kite')">🪁 The Little Kite</div>
                <div class="chapter-card" onclick="askAboutChapter('Kalam')">🚀 Dr. Kalam</div>
                <div class="chapter-card" onclick="askAboutChapter('Edison')">💡 Thomas Edison</div>
                <div class="chapter-card" onclick="askAboutChapter('Global Warming')">🌍 Global Warming</div>
            </div>
        </div>

        <script>
            let recognition;
            let synthesis = window.speechSynthesis;
            let isListening = false;
            let voiceOutputEnabled = true;

            // Initialize voice features
            window.onload = function() {
                initVoiceRecognition();
                initVoiceSynthesis();
            }

            function initVoiceRecognition() {
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                    recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                    recognition.continuous = false;
                    recognition.interimResults = false;
                    recognition.lang = 'en-US';
                    
                    recognition.onstart = function() {
                        updateVoiceStatus('🎤 Listening... Speak now!', '#ff6b6b');
                    };
                    
                    recognition.onresult = function(event) {
                        const transcript = event.results[0][0].transcript;
                        document.getElementById('userInput').value = transcript;
                        updateVoiceStatus('✅ Got it! "' + transcript + '"', '#4CAF50');
                        setTimeout(() => sendMessage(), 500);
                    };
                    
                    recognition.onerror = function(event) {
                        updateVoiceStatus('❌ Voice error: ' + event.error, '#ff6b6b');
                        resetVoiceButton();
                    };
                    
                    recognition.onend = function() {
                        resetVoiceButton();
                        if (!document.getElementById('userInput').value) {
                            updateVoiceStatus('🎤 Ready to listen again!', '#9c27b0');
                        }
                    };
                } else {
                    updateVoiceStatus('❌ Voice input not supported in this browser', '#ff6b6b');
                }
            }

            function initVoiceSynthesis() {
                if ('speechSynthesis' in window) {
                    // Wait for voices to load
                    if (synthesis.getVoices().length === 0) {
                        synthesis.addEventListener('voiceschanged', function() {
                            console.log('Voices loaded:', synthesis.getVoices().length);
                        });
                    }
                } else {
                    console.log('Speech synthesis not supported');
                }
            }

            function toggleVoiceInput() {
                if (!recognition) {
                    updateVoiceStatus('❌ Voice input not available', '#ff6b6b');
                    return;
                }

                if (!isListening) {
                    startListening();
                } else {
                    stopListening();
                }
            }

            function startListening() {
                recognition.start();
                isListening = true;
                const btn = document.getElementById('voiceBtn');
                btn.innerHTML = '⏹️ Stop';
                btn.classList.add('listening');
            }

            function stopListening() {
                recognition.stop();
                resetVoiceButton();
            }

            function resetVoiceButton() {
                isListening = false;
                const btn = document.getElementById('voiceBtn');
                btn.innerHTML = '🎤 Speak';
                btn.classList.remove('listening');
            }

            function updateVoiceStatus(message, color) {
                const status = document.getElementById('voiceStatus');
                status.innerHTML = message;
                status.style.color = color;
            }

            function speakResponse(text) {
                if (!voiceOutputEnabled || !('speechSynthesis' in window)) return;

                // Clean text for speech
                const cleanText = text.replace(/[🎤🔊📚✨👋⭐🎲💡🚀🌍🪁👵🐾🌸🏞️📤]/g, '');
                
                const utterance = new SpeechSynthesisUtterance(cleanText);
                
                // Try to find a suitable voice
                const voices = synthesis.getVoices();
                const preferredVoice = voices.find(voice => 
                    voice.name.toLowerCase().includes('female') || 
                    voice.name.toLowerCase().includes('woman') ||
                    voice.lang === 'en-US'
                );
                
                if (preferredVoice) {
                    utterance.voice = preferredVoice;
                }
                
                utterance.rate = 0.8; // Slower for children
                utterance.pitch = 1.1; // Slightly higher pitch
                utterance.volume = 0.8;
                
                synthesis.speak(utterance);
            }

            function toggleVoiceOutput() {
                voiceOutputEnabled = !voiceOutputEnabled;
                const btn = event.target;
                btn.innerHTML = voiceOutputEnabled ? '🔊 Voice: ON' : '🔇 Voice: OFF';
                btn.style.background = voiceOutputEnabled ? '#9c27b0' : '#666';
                
                if (voiceOutputEnabled) {
                    speakResponse('Voice output is now on!');
                }
            }

            function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                if (!message) return;

                addMessage('user', message);
                input.value = '';
                updateVoiceStatus('🎤 Ready to listen!', '#9c27b0');

                setTimeout(() => {
                    let response = getResponse(message);
                    addMessage('assistant', response);
                    speakResponse(response);
                }, 500);
            }

            function addMessage(sender, message) {
                const container = document.getElementById('chatContainer');
                const div = document.createElement('div');
                div.className = `message ${sender}`;
                div.innerHTML = message;
                container.appendChild(div);
                container.scrollTop = container.scrollHeight;
            }

            function askAboutChapter(chapter) {
                const input = document.getElementById('userInput');
                input.value = `Tell me about ${chapter}`;
                sendMessage();
            }

            function getRandomQuestion() {
                const questions = [
                    "What did the Little Kite learn about trying new things?",
                    "How do animals communicate with each other?", 
                    "What made Dr. Kalam such a wonderful person?",
                    "Why is it important to teach others to read?",
                    "What can we do to help with global warming?",
                    "What did Thomas Edison invent that lights up our homes?",
                    "What lesson did Mole learn at the River Bank?"
                ];
                const randomQ = questions[Math.floor(Math.random() * questions.length)];
                addMessage('assistant', `Here's a fun question for you: ${randomQ} 🤔 Take your time to think about it!`);
                speakResponse(`Here's a fun question for you: ${randomQ}`);
            }

            function getEncouragement() {
                const encouragements = [
                    "You're doing amazing, Nia! Keep up the great work! ⭐",
                    "I'm so proud of how hard you're trying! You're a super learner! 🌟",
                    "Every question you ask makes you smarter! Keep being curious! 🧠✨",
                    "You have such a bright mind, Nia! I love helping you learn! 💫",
                    "Reading and learning is making you stronger every day! 💪📚"
                ];
                const randomEnc = encouragements[Math.floor(Math.random() * encouragements.length)];
                addMessage('assistant', randomEnc);
                speakResponse(randomEnc);
            }

            function getResponse(message) {
                const lowerMessage = message.toLowerCase();
                
                if (lowerMessage.includes('river bank') || lowerMessage.includes('mole')) {
                    return "Great choice, Nia! 🏞️ The River Bank is about Mole and Water Rat who become friends! Mole had never been in a boat before, and Water Rat showed him the wonderful river. The story teaches us about friendship and trying new things. There's nothing better than messing about in boats with friends! What would you like to know more about? 😊";
                }
                
                if (lowerMessage.includes('champa') || lowerMessage.includes('flower')) {
                    return "Beautiful choice! 🌸 The Champa Flower is a lovely story that teaches us about nature and beauty. Flowers are so special because they make our world colorful and smell wonderful! What do you find most interesting about flowers? 🌺";
                }
                
                if (lowerMessage.includes('animal') || lowerMessage.includes('talk')) {
                    return "How exciting! 🐾 Animals have so many different ways to communicate! They use sounds, body language, and even smells to talk to each other. Birds sing, dogs bark, cats meow, and elephants trumpet! Each animal has its own special way. What's your favorite animal and how do you think it talks? 🦜";
                }
                
                if (lowerMessage.includes('grandmother') || lowerMessage.includes('read')) {
                    return "What a heartwarming story! 👵❤️ Teaching someone to read is such a beautiful gift. It shows how learning can happen at any age and how much love we can share through teaching. The story teaches us to be patient, kind, and helpful to others. Have you ever taught someone something new? 📖";
                }
                
                if (lowerMessage.includes('kite') || lowerMessage.includes('fly')) {
                    return "What a wonderful story about courage! 🪁 The little kite was scared to fly at first, but with practice and encouragement, it learned to soar high in the sky! This teaches us that it's okay to be scared of new things, but we should always try our best. Just like you're doing with your studies! ✨";
                }
                
                if (lowerMessage.includes('kalam') || lowerMessage.includes('scientist')) {
                    return "Dr. Kalam was amazing! 🚀 He was a brilliant scientist and became the President of India. He loved learning and always encouraged children to dream big! He showed us that with hard work and dedication, we can achieve anything. What do you want to be when you grow up, Nia? 🌟";
                }
                
                if (lowerMessage.includes('edison') || lowerMessage.includes('inventor')) {
                    return "Thomas Edison was an incredible inventor! 💡 He invented the light bulb and many other things we use today. He tried thousands of times before succeeding - he never gave up! He said 'I have not failed, I've just found 10,000 ways that won't work.' This teaches us to keep trying even when things are difficult! 🔬";
                }
                
                if (lowerMessage.includes('global warming') || lowerMessage.includes('earth')) {
                    return "This is such an important topic! 🌍 Global warming happens when Earth gets too hot because of gases in the air. We can help by planting trees, saving energy, using less paper, and taking care of nature. Every little action helps make our planet healthier! What can you do at home to help Earth? 🌱";
                }
                
                return "That's a great question, Nia! 🤔 I love how curious you are! Learning happens when we ask questions and explore. Keep asking, keep wondering, and keep being the amazing learner you are! Is there anything specific about your English chapters you'd like to know? 📚✨";
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8002))
    host = "0.0.0.0" if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RENDER") else "127.0.0.1"
    
    print("🎤 Starting Nia's VOICE-ENABLED Learning Assistant...")
    print("📚 Loaded training data and questions")
    print(f"🔊 Voice features: Speech recognition + Text-to-speech")
    print(f"🌟 Server will be available at: http://{host}:{port}")
    print("💫 Ready to help Nia learn with VOICE!")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
