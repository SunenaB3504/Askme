"""
AskMe Voice Assistant - Main Application Entry Point

This module initializes and runs the offline voice assistant with integrated
LLM, ASR (Whisper), and TTS (Coqui TTS) components.
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path

from src.core.config import Config
from src.core.assistant import VoiceAssistant
from src.ui.web_interface import WebInterface
from src.utils.logger import setup_logger


class AskMeApp:
    """Main application class for AskMe Voice Assistant"""
    
    def __init__(self, config_path: str = None):
        self.config = Config(config_path)
        self.logger = setup_logger(__name__, self.config.log_level)
        self.assistant = None
        self.web_interface = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing AskMe Voice Assistant...")
        
        try:
            # Initialize voice assistant
            self.assistant = VoiceAssistant(self.config)
            await self.assistant.initialize()
            
            # Initialize web interface
            self.web_interface = WebInterface(self.config, self.assistant)
            await self.web_interface.initialize()
            
            self.logger.info("✓ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start(self):
        """Start the voice assistant"""
        self.logger.info("Starting AskMe Voice Assistant...")
        
        # Setup signal handlers for graceful shutdown
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)
        
        try:
            # Start web interface
            web_task = asyncio.create_task(
                self.web_interface.start()
            )
            
            # Start voice assistant
            assistant_task = asyncio.create_task(
                self.assistant.start()
            )
            
            self.logger.info(f"🎤 Voice Assistant ready!")
            self.logger.info(f"🌐 Web interface available at http://{self.config.ui.host}:{self.config.ui.port}")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel tasks
            web_task.cancel()
            assistant_task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(web_task, assistant_task, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error during execution: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down AskMe Voice Assistant...")
        
        if self.web_interface:
            await self.web_interface.shutdown()
        
        if self.assistant:
            await self.assistant.shutdown()
        
        self.logger.info("✓ Shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AskMe - Offline Privacy-Centric Voice Assistant"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Override web interface port"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="Override web interface host"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Verify configuration file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Creating default configuration...")
        
        # Create default config directory
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy example config if it exists
        example_config = config_path.parent / "config.example.yaml"
        if example_config.exists():
            import shutil
            shutil.copy(example_config, config_path)
            print(f"✓ Default configuration created at {config_path}")
        else:
            print("❌ No example configuration found. Please create a configuration file.")
            return 1
    
    try:
        # Initialize application
        app = AskMeApp(str(config_path))
        
        # Override config with command line arguments
        if args.dev:
            app.config.development_mode = True
            app.config.log_level = "DEBUG"
        
        if args.log_level:
            app.config.log_level = args.log_level
        
        if args.port:
            app.config.ui.port = args.port
        
        if args.host:
            app.config.ui.host = args.host
        
        # Initialize and start
        await app.initialize()
        await app.start()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the application
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
