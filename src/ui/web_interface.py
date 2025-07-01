"""
Web interface for AskMe Voice Assistant using FastAPI
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..core.config import Config
from ..core.assistant import VoiceAssistant


class WebInterface:
    """Web interface for AskMe Voice Assistant"""
    
    def __init__(self, config: Config, assistant: VoiceAssistant):
        self.config = config
        self.assistant = assistant
        self.logger = logging.getLogger(__name__)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="AskMe Voice Assistant",
            description="Privacy-focused offline voice assistant",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.ui.security.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # WebSocket connections
        self.active_connections: list[WebSocket] = []
        
    async def initialize(self):
        """Initialize web interface"""
        # Setup static files and templates
        static_dir = Path(__file__).parent / "static"
        templates_dir = Path(__file__).parent / "templates"
        
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        if templates_dir.exists():
            self.templates = Jinja2Templates(directory=str(templates_dir))
        
        self.logger.info("✓ Web interface initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            """Main interface page"""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AskMe Voice Assistant</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background: #1a1a1a;
                        color: #ffffff;
                    }
                    .container {
                        background: #2d2d2d;
                        border-radius: 12px;
                        padding: 30px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                    }
                    h1 {
                        text-align: center;
                        color: #4CAF50;
                        margin-bottom: 30px;
                    }
                    .controls {
                        text-align: center;
                        margin: 30px 0;
                    }
                    button {
                        background: #4CAF50;
                        color: white;
                        border: none;
                        padding: 15px 30px;
                        font-size: 16px;
                        border-radius: 8px;
                        cursor: pointer;
                        margin: 10px;
                        transition: background 0.3s;
                    }
                    button:hover {
                        background: #45a049;
                    }
                    button:disabled {
                        background: #666;
                        cursor: not-allowed;
                    }
                    .listening {
                        background: #ff4444;
                    }
                    .listening:hover {
                        background: #ff6666;
                    }
                    .chat-container {
                        height: 400px;
                        overflow-y: auto;
                        border: 1px solid #444;
                        border-radius: 8px;
                        padding: 20px;
                        margin: 20px 0;
                        background: #1a1a1a;
                    }
                    .message {
                        margin: 15px 0;
                        padding: 12px;
                        border-radius: 8px;
                    }
                    .user-message {
                        background: #0084ff;
                        margin-left: 20%;
                    }
                    .assistant-message {
                        background: #4CAF50;
                        margin-right: 20%;
                    }
                    .status {
                        text-align: center;
                        padding: 10px;
                        border-radius: 8px;
                        margin: 10px 0;
                        background: #333;
                    }
                    .text-input {
                        width: 100%;
                        padding: 12px;
                        border: 1px solid #444;
                        border-radius: 8px;
                        background: #2d2d2d;
                        color: #fff;
                        font-size: 16px;
                        margin: 10px 0;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎤 AskMe Voice Assistant</h1>
                    
                    <div class="status" id="status">Ready</div>
                    
                    <div class="controls">
                        <button id="voiceBtn" onclick="toggleVoice()">Start Listening</button>
                        <button onclick="clearChat()">Clear Chat</button>
                    </div>
                    
                    <div class="chat-container" id="chatContainer"></div>
                    
                    <input type="text" class="text-input" id="textInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendText()">Send Text</button>
                </div>
                
                <script>
                    let ws = null;
                    let isListening = false;
                    let isConnected = false;
                    
                    function connectWebSocket() {
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                        
                        ws.onopen = function() {
                            isConnected = true;
                            updateStatus('Connected');
                        };
                        
                        ws.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            handleMessage(data);
                        };
                        
                        ws.onclose = function() {
                            isConnected = false;
                            updateStatus('Disconnected - Reconnecting...');
                            setTimeout(connectWebSocket, 2000);
                        };
                        
                        ws.onerror = function(error) {
                            console.error('WebSocket error:', error);
                            updateStatus('Connection error');
                        };
                    }
                    
                    function handleMessage(data) {
                        switch(data.type) {
                            case 'status':
                                updateStatus(data.message);
                                break;
                            case 'transcript':
                                addMessage(data.text, 'user');
                                break;
                            case 'response':
                                addMessage(data.text, 'assistant');
                                break;
                            case 'listening_start':
                                isListening = true;
                                updateVoiceButton();
                                updateStatus('Listening...');
                                break;
                            case 'listening_stop':
                                isListening = false;
                                updateVoiceButton();
                                updateStatus('Processing...');
                                break;
                            case 'error':
                                updateStatus('Error: ' + data.message);
                                break;
                        }
                    }
                    
                    function toggleVoice() {
                        if (!isConnected) return;
                        
                        if (isListening) {
                            ws.send(JSON.stringify({type: 'stop_listening'}));
                        } else {
                            ws.send(JSON.stringify({type: 'start_listening'}));
                        }
                    }
                    
                    function sendText() {
                        const input = document.getElementById('textInput');
                        const text = input.value.trim();
                        
                        if (text && isConnected) {
                            ws.send(JSON.stringify({type: 'text_input', text: text}));
                            input.value = '';
                        }
                    }
                    
                    function handleKeyPress(event) {
                        if (event.key === 'Enter') {
                            sendText();
                        }
                    }
                    
                    function addMessage(text, sender) {
                        const container = document.getElementById('chatContainer');
                        const message = document.createElement('div');
                        message.className = `message ${sender}-message`;
                        message.textContent = text;
                        container.appendChild(message);
                        container.scrollTop = container.scrollHeight;
                    }
                    
                    function updateStatus(message) {
                        document.getElementById('status').textContent = message;
                    }
                    
                    function updateVoiceButton() {
                        const btn = document.getElementById('voiceBtn');
                        if (isListening) {
                            btn.textContent = 'Stop Listening';
                            btn.className = 'listening';
                        } else {
                            btn.textContent = 'Start Listening';
                            btn.className = '';
                        }
                    }
                    
                    function clearChat() {
                        document.getElementById('chatContainer').innerHTML = '';
                        if (isConnected) {
                            ws.send(JSON.stringify({type: 'clear_history'}));
                        }
                    }
                    
                    // Initialize
                    connectWebSocket();
                </script>
            </body>
            </html>
            """
        
        @self.app.get("/api/status")
        async def get_status():
            """Get assistant status"""
            return self.assistant.get_status()
        
        @self.app.post("/api/text")
        async def process_text(request: dict):
            """Process text input"""
            text = request.get("text", "")
            if not text:
                raise HTTPException(status_code=400, detail="Text is required")
            
            response = await self.assistant.process_text_input(text)
            return {"response": response}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication"""
            await self.connect_websocket(websocket)
    
    async def connect_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Setup assistant callbacks
        self.assistant.set_callbacks(
            on_listening_start=lambda: self.broadcast_message({
                "type": "listening_start"
            }),
            on_listening_stop=lambda: self.broadcast_message({
                "type": "listening_stop"
            }),
            on_transcript=lambda text: self.broadcast_message({
                "type": "transcript",
                "text": text
            }),
            on_response=lambda text: self.broadcast_message({
                "type": "response", 
                "text": text
            }),
            on_error=lambda error: self.broadcast_message({
                "type": "error",
                "message": error
            })
        )
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                await self.handle_websocket_message(data, websocket)
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
            self.logger.info("WebSocket client disconnected")
    
    async def handle_websocket_message(self, data: dict, websocket: WebSocket):
        """Handle incoming WebSocket message"""
        message_type = data.get("type")
        
        try:
            if message_type == "start_listening":
                success = await self.assistant.start_listening()
                if not success:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to start listening"
                    })
            
            elif message_type == "stop_listening":
                transcript = await self.assistant.stop_listening()
                if transcript:
                    await websocket.send_json({
                        "type": "transcript",
                        "text": transcript
                    })
            
            elif message_type == "text_input":
                text = data.get("text", "")
                if text:
                    response = await self.assistant.process_text_input(text)
                    await websocket.send_json({
                        "type": "response",
                        "text": response
                    })
            
            elif message_type == "clear_history":
                self.assistant.clear_conversation_history()
                await websocket.send_json({
                    "type": "status",
                    "message": "Conversation history cleared"
                })
            
            elif message_type == "get_status":
                status = self.assistant.get_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": status
                })
        
        except Exception as e:
            self.logger.error(f"WebSocket message handling error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    async def start(self):
        """Start the web server"""
        config = uvicorn.Config(
            self.app,
            host=self.config.ui.host,
            port=self.config.ui.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def shutdown(self):
        """Shutdown web interface"""
        # Close all WebSocket connections
        for connection in self.active_connections:
            try:
                await connection.close()
            except:
                pass
        
        self.active_connections.clear()
        self.logger.info("Web interface shutdown complete")
