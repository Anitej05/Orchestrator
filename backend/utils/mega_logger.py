"""
Mega Logger — Centralized logging for the entire Orbimesh backend.

Architecture:
  - Handlers (console + daily rotating file) are attached ONCE to the ROOT logger.
  - All named loggers (e.g. "OrchestratorAPI", "BrowserAgent") inherit these handlers
    via Python's logging propagation — NO duplicate handlers, NO duplicate lines.
  - Calling `setup_mega_logger("MyModule")` simply returns a named child logger.
"""

import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler

_init_lock = threading.Lock()
# NOTE: _initialized is intentionally NOT used as the sole guard because this module
# can be imported under two different paths (e.g. "utils.mega_logger" vs
# "backend.utils.mega_logger") when both the project root and the backend directory
# are on sys.path, giving two module objects each with _initialized=False.
# We use a sentinel attribute on the root logger object itself as the canonical guard,
# since logging.getLogger() always returns the same singleton regardless of import path.
_SENTINEL = "_mega_logger_initialized"


def _initialize_root_logger() -> None:
    """Attach console + file handlers to the root logger exactly once."""
    root = logging.getLogger()
    if getattr(root, _SENTINEL, False):
        return

    with _init_lock:
        # Double-check inside the lock (thread-safe singleton pattern)
        if getattr(root, _SENTINEL, False):
            return

        root.setLevel(logging.INFO)

        # Formatter shared by all handlers
        formatter = logging.Formatter(
            '%(asctime)s | [%(levelname)s] | <%(name)s> | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # --- Console Handler (UTF-8 for emoji safety) ---
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass  # Graceful fallback if stdout can't be reconfigured
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root.addHandler(console_handler)

        # --- Daily Rotating File Handler ---
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs', 'mega_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'mega_log.log')

        file_handler = TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root.addHandler(file_handler)

        # Silence noisy third-party loggers
        for noisy in ("urllib3", "httpcore", "httpx", "openai", "composio",
                       "uvicorn.access", "asyncio", "watchfiles"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        setattr(root, _SENTINEL, True)


def setup_mega_logger(logger_name: str = None) -> logging.Logger:
    """
    Return a named logger that inherits the global mega-log handlers.

    First call initializes the root logger's handlers; subsequent calls are free.
    Named loggers propagate to root — they do NOT get their own handlers.
    """
    _initialize_root_logger()
    return logging.getLogger(logger_name)


def mega_log_block(logger: logging.Logger, title: str, content: str, level: int = logging.INFO):
    """
    Log a large multi-line payload inside a visually distinct block.
    Useful for intermediate JSON states, canvas dumps, API request bodies.
    """
    separator = "=" * 80
    logger.log(level, f"\n{separator}\n>>> {title.upper()} <<<\n{content}\n{separator}\n")
