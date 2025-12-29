"""
Spreadsheet Agent - Modular package for spreadsheet operations.

A comprehensive modular system for spreadsheet manipulation with:
- LLM-powered natural language queries
- Code generation for transformations
- Memory/caching system for performance
- Thread-safe session management
- Standardized file management

Version: 2.0.0
"""

# Core configuration and models
from . import config
from . import models
from . import memory

# Business logic modules
from . import llm_agent
from . import code_generator
from . import session
from . import display

# Utilities
from . import utils

# Main FastAPI application
from .main import app

__version__ = '2.0.0'
__all__ = [
    'config',
    'models', 
    'memory',
    'llm_agent',
    'code_generator',
    'session',
    'display',
    'utils',
    'app'
]

