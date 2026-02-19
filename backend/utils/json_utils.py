"""
JSON Utilities - Centralized JSON processing functions

This module provides robust JSON extraction, parsing, and serialization utilities
to avoid code duplication across the codebase.
"""

import json
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[Any]:
    """
    Extract and parse JSON from text that may contain surrounding content.

    Handles:
    - JSON embedded in markdown code blocks
    - JSON with trailing content
    - Multiple JSON objects (returns first valid one)
    - Nested JSON structures

    Args:
        text: String that may contain JSON

    Returns:
        Parsed JSON object or None if no valid JSON found
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Try to find the start of JSON (object or array)
    start_idx = text.find("{")
    array_start = text.find("[")

    if start_idx == -1 and array_start == -1:
        return None

    # Determine which comes first (object or array)
    if start_idx == -1:
        start_idx = array_start
    elif array_start != -1 and array_start < start_idx:
        start_idx = array_start

    text = text[start_idx:]

    # Try raw_decode first (most robust)
    decoder = json.JSONDecoder()
    try:
        obj, idx = decoder.raw_decode(text)
        return obj
    except json.JSONDecodeError:
        pass

    # Fallback: Try to find the last valid closing brace/bracket
    # by gradually truncating from the end
    for i in range(len(text), 0, -1):
        try:
            return json.loads(text[:i])
        except json.JSONDecodeError:
            continue

    return None


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with a default value on failure.

    Args:
        text: JSON string to parse
        default: Value to return if parsing fails

    Returns:
        Parsed JSON object or default value
    """
    if not text:
        return default

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON parse failed: {e}")
        return default


def safe_json_dumps(obj: Any, default: str = "{}", **kwargs) -> str:
    """
    Safely serialize object to JSON with a default value on failure.

    Args:
        obj: Object to serialize
        default: String to return if serialization fails
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, default=str, **kwargs)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed: {e}")
        return default


def extract_json_with_fallback(
    text: str, schema: Any = None
) -> Tuple[Any, Optional[str]]:
    """
    Extract JSON from text with comprehensive fallback strategies.

    Args:
        text: Text that may contain JSON
        schema: Optional Pydantic model to validate against

    Returns:
        Tuple of (parsed_object, error_message)
        If successful, error_message is None
    """
    # Strategy 1: Direct extraction
    data = extract_json_from_text(text)

    if data is None:
        return None, "No valid JSON found in text"

    # Strategy 2: Schema validation if provided
    if schema is not None:
        try:
            if hasattr(schema, "model_validate"):
                # Pydantic v2
                data = schema.model_validate(data)
            elif hasattr(schema, "parse_obj"):
                # Pydantic v1
                data = schema.parse_obj(data)
            else:
                return None, f"Unknown schema type: {type(schema)}"
        except Exception as e:
            return None, f"Schema validation failed: {str(e)}"

    return data, None


def normalize_json_string(text: str) -> str:
    """
    Normalize a string for consistent JSON comparison.
    Removes extra whitespace and normalizes Unicode.

    Args:
        text: String to normalize

    Returns:
        Normalized string
    """
    if not text:
        return ""

    # Remove extra whitespace, normalize to single spaces
    normalized = " ".join(text.split())

    return normalized


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles common non-serializable types."""

    def default(self, obj):
        # Handle sets
        if isinstance(obj, set):
            return list(obj)

        # Handle bytes
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return obj.hex()

        # Handle datetime objects
        if hasattr(obj, "isoformat"):
            return obj.isoformat()

        # Handle objects with to_dict method
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        # Handle objects with __dict__
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        # Fall back to string representation
        return str(obj)


def dumps_with_fallback(obj: Any, **kwargs) -> str:
    """
    Serialize to JSON with comprehensive fallback handling.

    Args:
        obj: Object to serialize
        **kwargs: Arguments for json.dumps

    Returns:
        JSON string
    """
    try:
        return json.dumps(obj, cls=JSONEncoder, **kwargs)
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        # Last resort: convert to string
        try:
            return json.dumps({"error": "Serialization failed", "repr": repr(obj)})
        except:
            return '{"error": "Serialization failed"}'
