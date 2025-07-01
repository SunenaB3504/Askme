"""
Package initialization for AskMe Voice Assistant
"""

__version__ = "1.0.0"
__author__ = "AskMe Development Team"
__email__ = "dev@askme-assistant.com"
__description__ = "Privacy-focused offline voice assistant with custom LLM"

from .core.config import Config
from .core.assistant import VoiceAssistant

__all__ = ["Config", "VoiceAssistant"]
