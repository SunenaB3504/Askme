#!/usr/bin/env python3
"""
Simple launcher for Nia's Educational Assistant
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
from pathlib import Path

# Create FastAPI app
app = FastAPI(title="Nia's Learning Assistant")

# Load Nia's training data
def load_nia_data():
    try:
        with open("data/nia_english/nia_training_data.json", "r", encoding="utf-8") as f:
            training_data = json.load(f)
        
        with open("data/nia_english/generated_questions.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
            
        with open("data/nia_english/processed_chapters.json", "r", encoding="utf-8") as f:
            chapters = json.load(f)
            
        return training_data, questions, chapters
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], {}, {}

# Load data
training_data, questions, chapters = load_nia_data()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nia's Learning Assistant</title>
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
                padding: 30px; 
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { 
                text-align: center; 
                color: #FFD700; 
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }
            .chat-container { 
                background: white; 
                color: #333; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
                min-height: 400px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            input[type="text"] { 
                width: 70%; 
                padding: 15px; 
                border: 2px solid #667eea; 
                border-radius: 25px; 
                font-size: 16px;
                outline: none;
            }
            button { 
                padding: 15px 25px; 
                background: #667eea; 
                color: white; 
                border: none; 
                border-radius: 25px; 
                cursor: pointer; 
                font-size: 16px;
                margin-left: 10px;
                transition: all 0.3s ease;
            }
            button:hover { 
                background: #5a67d8; 
                transform: translateY(-2px);
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
            }
            .assistant { 
                background: #f1f8e9; 
                border-bottom-left-radius: 5px;
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 Nia's Learning Assistant 🌟</h1>
            
            <div style="text-align: center; margin: 20px 0;">
                <p style="font-size: 1.2em;">Hi Nia! I'm your personal learning helper! 📚✨</p>
                <p>I know all about your English chapters and I'm here to help you learn and have fun!</p>
            </div>

            <div class="chapters">
                <div class="chapter-card" onclick="askAboutChapter('River Bank')">
                    🏞️ The River Bank
                </div>
                <div class="chapter-card" onclick="askAboutChapter('Champa Flower')">
                    🌸 The Champa Flower
                </div>
                <div class="chapter-card" onclick="askAboutChapter('Animals Talk')">
                    🐾 How Animals Talk
                </div>
                <div class="chapter-card" onclick="askAboutChapter('Grandmother Read')">
                    👵 Teaching Grandmother
                </div>
                <div class="chapter-card" onclick="askAboutChapter('Little Kite')">
                    🪁 The Little Kite
                </div>
                <div class="chapter-card" onclick="askAboutChapter('Kalam')">
                    🚀 Wonderful Kalam
                </div>
                <div class="chapter-card" onclick="askAboutChapter('Thomas Edison')">
                    💡 Thomas Edison
                </div>
                <div class="chapter-card" onclick="askAboutChapter('Global Warming')">
                    🌍 Global Warming
                </div>
            </div>

            <div class="chat-container" id="chatContainer">
                <div class="message assistant">
                    <strong>Your Learning Helper:</strong> Hi Nia! 👋 I'm so excited to help you learn! You can:
                    <br>• Click on any chapter above to learn about it
                    <br>• Ask me questions like "Tell me about The River Bank"
                    <br>• Ask for practice questions: "Give me a question about animals"
                    <br>• Get help: "I don't understand this chapter"
                    <br><br>What would you like to learn about today? 🤔
                </div>
            </div>

            <div style="margin-top: 20px; text-align: center;">
                <input type="text" id="userInput" placeholder="Type your question here, Nia!" onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Ask! 🎯</button>
            </div>

            <div style="text-align: center; margin-top: 20px;">
                <button onclick="getRandomQuestion()" style="background: #FF6B6B;">📝 Random Question</button>
                <button onclick="getEncouragement()" style="background: #4ECDC4;">💪 Encouragement</button>
                <button onclick="getStudyTip()" style="background: #45B7D1;">📚 Study Tip</button>
            </div>
        </div>

        <script>
            function addMessage(sender, message) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                messageDiv.innerHTML = `<strong>${sender === 'user' ? 'Nia' : 'Your Learning Helper'}:</strong> ${message}`;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function askAboutChapter(chapter) {
                const input = document.getElementById('userInput');
                input.value = `Tell me about ${chapter}`;
                sendMessage();
            }

            function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                if (!message) return;

                addMessage('user', message);
                input.value = '';

                // Simple response logic based on Nia's training data
                setTimeout(() => {
                    let response = getResponse(message);
                    addMessage('assistant', response);
                }, 500);
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
                    return "This is such an important topic! 🌍 Global warming means our Earth is getting warmer because of pollution. We can help by planting trees, saving water, and keeping our environment clean. Every small action matters! You can help by not wasting paper, turning off lights, and taking care of plants. What will you do to help our Earth? 🌱";
                }
                
                if (lowerMessage.includes('question') || lowerMessage.includes('quiz')) {
                    return "I love that you want to practice! 📝 Here's a question for you: In 'The River Bank' story, who had never been in a boat before? A) Water Rat B) Mole C) The Fish D) The Duck. Take your time and think about it! 🤔";
                }
                
                if (lowerMessage.includes('hard') || lowerMessage.includes('difficult')) {
                    return "Oh Nia, I understand that learning can feel challenging sometimes! 💪 But you know what? Every time you try something hard, your brain gets stronger - just like exercising makes your muscles stronger! You're so smart and brave. Let's break it down into smaller, easier pieces. What part would you like me to help you with first? 🌟";
                }
                
                if (lowerMessage.includes('wrong') || lowerMessage.includes('mistake')) {
                    return "That's perfectly okay, Nia! 🤗 Making mistakes is how we learn and grow. Even the smartest people in the world make mistakes - that's completely normal! What matters is that you tried, and I'm so proud of you for that. Let me help you understand the right answer, and then you'll know it forever! You're doing amazing! ✨";
                }
                
                if (lowerMessage.includes('study') || lowerMessage.includes('help')) {
                    return "I'm so proud of you for wanting to study! 📚 Here are some fun ways to learn: 1) Read the stories out loud with expression, 2) Draw pictures of the characters, 3) Ask yourself questions about what happened, 4) Tell the story to your family or toys! Remember, learning should be fun. You're such a wonderful student! 🎓";
                }
                
                // Default encouraging response
                return "That's a great question, Nia! 🌟 I love how curious you are about learning! While I'm still learning to understand everything perfectly, I want you to know that you're doing an amazing job. Keep asking questions - that's how smart people learn! Is there a specific chapter or topic you'd like to explore together? 📚✨";
            }

            function getRandomQuestion() {
                const questions = [
                    "In 'The River Bank', who became friends with Mole? 🏞️",
                    "What did Thomas Edison invent that lights up our homes? 💡",
                    "How can we help save our Earth from global warming? 🌍",
                    "What lesson did the Little Kite learn about trying new things? 🪁",
                    "Why is it wonderful to teach someone to read? 📖"
                ];
                const randomQ = questions[Math.floor(Math.random() * questions.length)];
                addMessage('assistant', `Here's a fun question for you: ${randomQ} Take your time to think about it! 😊`);
            }

            function getEncouragement() {
                const encouragements = [
                    "You're such a bright and curious student, Nia! 🌟 Keep up the amazing work!",
                    "I'm so proud of how hard you're working! 💪 You're going to achieve great things!",
                    "Your questions show how smart you are! 🧠 Never stop being curious!",
                    "You have such a wonderful mind for learning! 📚 I believe in you completely!",
                    "Every day you're getting smarter and stronger! 🚀 You're absolutely amazing!"
                ];
                const randomEnc = encouragements[Math.floor(Math.random() * encouragements.length)];
                addMessage('assistant', randomEnc);
            }

            function getStudyTip() {
                const tips = [
                    "Try reading the stories aloud - it helps you remember better! 🗣️",
                    "Draw pictures of the characters and scenes - it makes learning fun! 🎨",
                    "Ask 'who, what, where, when, why' questions about each story! ❓",
                    "Explain the story to a family member or even your toys! 🧸",
                    "Take breaks when studying - your brain needs rest to grow strong! 😴"
                ];
                const randomTip = tips[Math.floor(Math.random() * tips.length)];
                addMessage('assistant', `Study Tip: ${randomTip}`);
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            // Welcome message
            window.onload = function() {
                setTimeout(() => {
                    addMessage('assistant', "Welcome back, Nia! 🎉 I'm so excited to help you learn today. Which chapter would you like to explore? 📖✨");
                }, 1000);
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Nia's Learning Assistant is running!"}

@app.get("/api/chapters")
async def get_chapters():
    return {"chapters": list(chapters.keys())}

@app.get("/api/questions")
async def get_questions():
    return questions

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8001))
    host = "0.0.0.0" if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RENDER") else "127.0.0.1"
    
    print("🎓 Starting Nia's Learning Assistant...")
    print("📚 Loaded training data and questions")
    print(f"🌟 Server will be available at: http://{host}:{port}")
    print("💫 Ready to help Nia learn!")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
