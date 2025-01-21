import logging
import sys
from datetime import datetime
from typing import Optional

class ECFRLogger:
    """Custom logger for the ECFR API wrapper"""
    
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'ENDC': '\033[0m'       # End color
    }

    def __init__(self, name: str = "ECFRLogger", log_file: Optional[str] = None, level=logging.INFO):
        """
        Initialize the logger.
        
        Args:
            name (str): Logger name
            log_file (str, optional): Path to log file
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create formatters
        class MultiLineFormatter(logging.Formatter):
            def format(self, record):
                message = record.getMessage()
                if '\n' in message:
                    # Add proper indentation for multiline messages
                    message = message.replace('\n', '\n' + ' ' * 42)
                record.message = message
                return super().format(record)

        console_formatter = MultiLineFormatter(
            fmt='%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        
        # Add custom color formatting
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.COLORS['ENDC']}"
            return record
        
        logging.setLogRecordFactory(record_factory)
        
        self.logger.addHandler(console_handler)
        
        # File handler if log_file is specified
        if log_file:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

# Create default logger instance
logger = ECFRLogger()
