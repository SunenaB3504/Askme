"""
Logging utilities for AskMe Voice Assistant
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str, 
                level: str = "INFO",
                log_file: Optional[str] = None,
                max_file_size: str = "10MB",
                backup_count: int = 5,
                log_format: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with both console and file handlers
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        log_format: Custom log format string
    
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max file size
        max_bytes = _parse_size(max_file_size)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """Parse size string to bytes"""
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


class PrivacyFilter(logging.Filter):
    """Filter to remove sensitive information from logs"""
    
    SENSITIVE_PATTERNS = [
        # API keys and tokens
        r'(?i)(api[_-]?key|token|secret)["\s:=]+[a-zA-Z0-9_\-]+',
        # Email addresses
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # Phone numbers (basic pattern)
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        # Credit card numbers (basic pattern)
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    ]
    
    def filter(self, record):
        """Filter log record to remove sensitive information"""
        import re
        
        # Apply filters to message
        message = record.getMessage()
        
        for pattern in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, '[REDACTED]', message)
        
        # Update the record
        record.msg = message
        record.args = ()
        
        return True


def setup_privacy_logging(logger: logging.Logger):
    """Add privacy filter to logger"""
    privacy_filter = PrivacyFilter()
    
    for handler in logger.handlers:
        handler.addFilter(privacy_filter)
