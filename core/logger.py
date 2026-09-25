import logging
from logging.handlers import RotatingFileHandler
import os
import sys

_INITIALIZED = False


def setup_logging(log_level: str = "INFO", log_file: str = "anima.log") -> logging.Logger:
    """Configure standard logging with console and rotating file output."""
    global _INITIALIZED
    root_logger = logging.getLogger()

    if _INITIALIZED:
        return root_logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler (5 MB, up to 3 backups)
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"[logging] could not initialize file logger: {e}", file=sys.stderr)

    _INITIALIZED = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
