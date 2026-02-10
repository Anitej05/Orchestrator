"""
Text processing utilities for the Orbimesh backend.

This module contains common text processing functions used across agents and services.
"""

import re
from typing import Optional


def strip_think_tags(text: str) -> str:
    """
    Remove thinking/reasoning tags from LLM output.
    Handles ALL known formats from various models.
    
    Args:
        text: Input text that may contain thinking tags
        
    Returns:
        Text with all thinking tags removed
        
    Examples:
        >>> strip_think_tags("<think>reasoning</think>actual response")
        'actual response'
        >>> strip_think_tags("<|thinking|>internal thoughts</|thinking|>output")
        'output'
    """
    if not isinstance(text, str):
        return text
    
    # Pattern 1: <think>...</think> (closed tags)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 2: Minimax pipe format <|thinking|>...</|thinking|>
    text = re.sub(r'<\|thinking\|>.*?</\|thinking\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\|thinking\|>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 3: Minimax alternate format <|thought|>...</|thought|>
    text = re.sub(r'<\|thought\|>.*?</\|thought\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\|thought\|>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 4: DeepSeek format <thought>...</thought>
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 5: Chinese thinking tags
    text = re.sub(r'【thinking】.*?【/thinking】', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'【thinking】.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 6: <reasoning>...</reasoning>
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 7: Generic pipe format
    text = re.sub(r'<\|[a-z_]+\|>.*?</\|[a-z_]+\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding a suffix if truncated.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of returned text (including suffix)
        suffix: String to append if text is truncated
        
    Returns:
        Truncated text with suffix if applicable
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    actual_max = max_length - len(suffix)
    return text[:actual_max].rstrip() + suffix


def clean_whitespace(text: str) -> str:
    """
    Normalize whitespace in text (remove extra spaces, tabs, newlines).
    
    Args:
        text: Text to clean
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_code_blocks(text: str, language: Optional[str] = None) -> list[str]:
    """
    Extract code blocks from markdown-formatted text.
    
    Args:
        text: Text containing markdown code blocks
        language: Optional language filter (e.g., 'python', 'javascript')
        
    Returns:
        List of code block contents
    """
    if language:
        # Match specific language code blocks
        pattern = rf'```{language}\n(.*?)```'
    else:
        # Match all code blocks
        pattern = r'```(?:\w+)?\n(.*?)```'
    
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed filename length
        
    Returns:
        Sanitized filename safe for most filesystems
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip(' .')
    
    # Truncate if too long
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        filename = f"{name[:max_name_length]}.{ext}" if ext else name[:max_length]
    
    return filename or 'unnamed'
