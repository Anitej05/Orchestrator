"""
CodeSandboxService — Stateful Python execution environment for the orchestrator.

Provides a safe, session-based sandbox where LLM-generated Python code can run.
Each session maintains persistent state (variables) across executions within the
same conversation thread.

Key design decisions:
- HTTP identity: Pre-configured requests.Session injected as `http` (and aliased 
  as `requests`), plus global urllib opener — covers both pd.read_html(url) and
  requests.get(url) without monkeypatching.
- Matplotlib: Forced to 'Agg' backend at import time to prevent GUI thread errors
  when code runs in background executor threads.
- Execution timeout: Configurable per-call timeout prevents runaway LLM code from
  hanging the orchestrator.
- Thread-safe output capture: Uses per-call StringIO buffers instead of sys.stdout
  monkey-patching.
"""

import logging
import traceback
import io
import os
import re
import textwrap
import threading
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Pre-import heavy libraries ONCE at module level
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import json

# Force Matplotlib to non-interactive backend BEFORE pyplot import
# This prevents "main thread is not in main loop" errors when LLM code
# generates charts from executor threads.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import requests as _requests_module
import urllib.request

logger = logging.getLogger("CodeSandboxService")

# Backend directory for resolving relative paths
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Browser-like User-Agent to prevent 403 blocks from Wikipedia etc.
DEFAULT_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/131.0.0.0 Safari/537.36'
)


class _SandboxRequestsModule:
    """
    A transparent wrapper around the `requests` module that injects browser-like
    User-Agent headers on every HTTP call.
    
    WHY THIS EXISTS:
    - LLM code often does `import requests; requests.get(url)`.
    - `import requests` inside `exec()` always loads from `sys.modules`,
      overriding any pre-injected Session objects in exec_globals.
    - Monkeypatching `requests.utils.default_headers` works but is a fragile
      lambda override on an internal API.
    
    THIS APPROACH:
    - Wraps the real `requests` module with a transparent proxy.
    - Intercepts `.get()`, `.post()`, `.put()`, `.patch()`, `.delete()`, `.head()`
      to inject default headers if the caller doesn't provide their own.
    - All other attributes (Session, exceptions, models, etc.) delegate to the
      real module unchanged.
    - Installed in `sys.modules['requests']` at startup so ALL `import requests`
      calls (including inside exec()) get the wrapped version.
    """
    
    _HTTP_METHODS = ('get', 'post', 'put', 'patch', 'delete', 'head', 'options')
    
    def __init__(self, real_module, default_headers: dict):
        self._real = real_module
        self._default_headers = default_headers
    
    def __getattr__(self, name):
        """Delegate everything to the real module."""
        return getattr(self._real, name)
    
    def _make_request(self, method: str, url, **kwargs):
        """Call a requests method, injecting default headers if not provided."""
        headers = {**self._default_headers}
        if 'headers' in kwargs and kwargs['headers']:
            headers.update(kwargs['headers'])  # Caller's headers take precedence
        kwargs['headers'] = headers
        return getattr(self._real, method)(url, **kwargs)
    
    def get(self, url, **kwargs):     return self._make_request('get', url, **kwargs)
    def post(self, url, **kwargs):    return self._make_request('post', url, **kwargs)
    def put(self, url, **kwargs):     return self._make_request('put', url, **kwargs)
    def patch(self, url, **kwargs):   return self._make_request('patch', url, **kwargs)
    def delete(self, url, **kwargs):  return self._make_request('delete', url, **kwargs)
    def head(self, url, **kwargs):    return self._make_request('head', url, **kwargs)
    def options(self, url, **kwargs): return self._make_request('options', url, **kwargs)


def _configure_global_http_identity():
    """
    Set up HTTP identity at the process level. Called once at module load.
    
    Two mechanisms cover Python's split HTTP ecosystem:
    
    1. urllib.request.install_opener() — covers:
       - pd.read_html(url), pd.read_csv(url)
       - urllib.request.urlopen(url)
       
    2. SandboxRequestsModule wrapper in sys.modules — covers:
       - requests.get(url), requests.post(url)
       - Any code that does `import requests` (including inside exec())
    """
    # --- urllib ---
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', DEFAULT_USER_AGENT),
        ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
        ('Accept-Language', 'en-US,en;q=0.5'),
    ]
    urllib.request.install_opener(opener)
    
    # --- requests ---
    default_headers = {
        'User-Agent': DEFAULT_USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    wrapped = _SandboxRequestsModule(_requests_module, default_headers)
    sys.modules['requests'] = wrapped
    
    logger.info("Global HTTP identity configured (urllib opener + requests wrapper)")


# ---------------------------------------------------------------------------
# Safe builtins whitelist — everything an LLM would reasonably need
# ---------------------------------------------------------------------------
SAFE_BUILTINS = {
    # Core types
    'len': len, 'range': range, 'str': str, 'int': int, 'float': float,
    'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
    'bytes': bytes, 'bytearray': bytearray, 'frozenset': frozenset,
    'complex': complex, 'memoryview': memoryview, 'object': object,
    'slice': slice, 'type': type,
    # Iterators & generators
    'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
    'iter': iter, 'next': next, 'reversed': reversed,
    # Math & comparison
    'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
    'pow': pow, 'divmod': divmod,
    # String & formatting
    'format': format, 'chr': chr, 'ord': ord, 'hex': hex, 'oct': oct, 'bin': bin,
    'repr': repr, 'ascii': ascii,
    # Boolean & inspection
    'any': any, 'all': all, 'isinstance': isinstance, 'issubclass': issubclass,
    'callable': callable, 'hash': hash, 'id': id,
    'sorted': sorted, 'vars': vars, 'dir': dir,
    'globals': globals, 'locals': locals,
    # Attribute access
    'getattr': getattr, 'setattr': setattr, 'hasattr': hasattr, 'delattr': delattr,
    # OOP
    'super': super, 'property': property,
    'classmethod': classmethod, 'staticmethod': staticmethod,
    # IO & import
    'open': open, 'print': print, 'input': input,
    '__import__': __import__, '__build_class__': __build_class__,
    # Standard exceptions (all of them — LLM code needs to catch/raise these)
    'Exception': Exception, 'BaseException': BaseException,
    'ArithmeticError': ArithmeticError, 'LookupError': LookupError,
    'ValueError': ValueError, 'TypeError': TypeError,
    'KeyError': KeyError, 'IndexError': IndexError,
    'AttributeError': AttributeError, 'NameError': NameError,
    'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
    'NotImplementedError': NotImplementedError,
    'OSError': OSError, 'IOError': IOError,
    'FileNotFoundError': FileNotFoundError, 'FileExistsError': FileExistsError,
    'PermissionError': PermissionError,
    'IsADirectoryError': IsADirectoryError, 'NotADirectoryError': NotADirectoryError,
    'ZeroDivisionError': ZeroDivisionError, 'OverflowError': OverflowError,
    'FloatingPointError': FloatingPointError,
    'UnicodeError': UnicodeError,
    'UnicodeDecodeError': UnicodeDecodeError, 'UnicodeEncodeError': UnicodeEncodeError,
    'ImportError': ImportError, 'ModuleNotFoundError': ModuleNotFoundError,
    'SyntaxError': SyntaxError, 'IndentationError': IndentationError,
    'SystemExit': SystemExit, 'KeyboardInterrupt': KeyboardInterrupt,
    'GeneratorExit': GeneratorExit, 'AssertionError': AssertionError,
    'EOFError': EOFError, 'MemoryError': MemoryError,
    'RecursionError': RecursionError, 'TimeoutError': TimeoutError,
    'ConnectionError': ConnectionError, 'BrokenPipeError': BrokenPipeError,
    'BufferError': BufferError,
    'Warning': Warning, 'UserWarning': UserWarning,
    'DeprecationWarning': DeprecationWarning, 'FutureWarning': FutureWarning,
}


class CodeSandboxService:
    """
    Stateful Python execution sandbox for the orchestrator.
    
    Each session maintains persistent variables across calls, allowing multi-step
    code execution within a conversation thread. Sessions are isolated from each
    other.
    
    Usage:
        sandbox = CodeSandboxService()
        result = sandbox.execute_code("x = 42; print(x)", session_id="thread_123")
        # result = {"success": True, "result": None, "stdout": "42\n", "error": None}
    """

    DEFAULT_TIMEOUT = 120  # seconds — generous for web requests + data processing
    MAX_SESSIONS = 50      # prevent memory leaks from abandoned sessions

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Enable nested asyncio for code that uses await
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            logger.warning("nest_asyncio not installed — async code in sandbox may fail")
        
        logger.info("CodeSandboxService initialized")

    def _create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Create a fresh sandbox session with pre-loaded libraries.
        
        The session globals include:
        - All safe builtins (types, exceptions, iteration, etc.)
        - Data science stack: pd, np, plt, matplotlib
        - HTTP: requests (pre-configured Session with User-Agent headers)
        - Utilities: json, os, re, math, datetime, io, pathlib, textwrap
        - Custom: BACKEND_DIR, normalize_path helper
        """
        output_buffer = io.StringIO()

        def _make_print(buffer):
            """Create a print function that captures output to the buffer."""
            def custom_print(*args, **kwargs):
                kwargs['file'] = buffer
                print(*args, **kwargs)
            return custom_print

        # Build the globals dict with everything the LLM might need
        session_globals = {
            '__builtins__': {**SAFE_BUILTINS, 'print': _make_print(output_buffer)},
            '__name__': '__main__',
            
            # Data science
            'pd': pd,
            'np': np,
            'plt': plt,
            'matplotlib': matplotlib,
            
            # HTTP — sys.modules['requests'] is already our wrapped module,
            # so `import requests` inside exec() will get the wrapper.
            # We also inject it directly for code that uses requests without import.
            'requests': sys.modules['requests'],
            
            # Serialization & IO
            'json': json,
            'io': io,
            'StringIO': io.StringIO,
            'BytesIO': io.BytesIO,
            
            # Standard library essentials
            'os': os,
            're': re,
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            'pathlib': __import__('pathlib'),
            'textwrap': textwrap,
            'collections': __import__('collections'),
            
            # Project context
            'BACKEND_DIR': BACKEND_DIR,
        }

        return {
            'globals': session_globals,
            'output_buffer': output_buffer,
        }

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get existing session or create a new one. Evicts oldest if over limit."""
        if session_id not in self.sessions:
            # Evict oldest session if at capacity
            if len(self.sessions) >= self.MAX_SESSIONS:
                oldest_key = next(iter(self.sessions))
                logger.warning(f"Session limit ({self.MAX_SESSIONS}) reached, evicting: {oldest_key}")
                self._cleanup_session(oldest_key)
            
            self.sessions[session_id] = self._create_session(session_id)
        return self.sessions[session_id]

    def _cleanup_session(self, session_id: str):
        """Clean up a session's resources."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Close the output buffer
            output_buffer = session.get('output_buffer')
            if output_buffer:
                try:
                    output_buffer.close()
                except Exception:
                    pass
            del self.sessions[session_id]

    def execute_code(
        self,
        code: str,
        session_id: str = "default",
        context_vars: Dict[str, Any] = None,
        timeout: int = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code in a persistent, sandboxed session.
        
        Args:
            code: Python code string to execute.
            session_id: Session ID for state persistence across calls.
            context_vars: Additional variables to inject for this execution.
            timeout: Max execution time in seconds (default: DEFAULT_TIMEOUT).
            
        Returns:
            Dict with keys:
                success (bool): Whether code executed without exceptions.
                result (Any): Value of `result` variable if set by code.
                stdout (str): Captured print output.
                error (str|None): Error message if execution failed.
        """
        session = self._get_or_create_session(session_id)
        exec_globals = session['globals']
        output_buffer = session['output_buffer']
        timeout = timeout or self.DEFAULT_TIMEOUT

        # Reset output buffer for this execution
        output_buffer.seek(0)
        output_buffer.truncate(0)

        # Inject context variables
        if context_vars:
            exec_globals.update(context_vars)

        # Dedent the code (LLM sometimes produces indented blocks)
        code_to_execute = textwrap.dedent(code)

        # Execute with timeout protection
        result = None
        error = None
        success = False

        exec_error = [None]  # Mutable container for thread communication
        exec_done = threading.Event()

        def _run_code():
            try:
                exec(code_to_execute, exec_globals)
            except Exception as e:
                exec_error[0] = e
            finally:
                exec_done.set()

        thread = threading.Thread(target=_run_code, daemon=True)
        thread.start()
        
        finished = exec_done.wait(timeout=timeout)

        if not finished:
            error = f"Execution timed out after {timeout}s"
            logger.error(f"Sandbox execution timed out (session={session_id})")
        elif exec_error[0]:
            error = str(exec_error[0])
            logger.error(f"Sandbox execution failed: {exec_error[0]}\n{traceback.format_exception(type(exec_error[0]), exec_error[0], exec_error[0].__traceback__)}")
        else:
            success = True

        # Extract `result` variable if code set it
        if 'result' in exec_globals:
            result = self._safe_serialize(exec_globals['result'])

        # Capture stdout
        stdout = output_buffer.getvalue()

        # Warn if code produced no output
        if success and not stdout.strip() and result is None:
            stdout = "[SYSTEM WARNING] Code executed successfully but produced NO OUTPUT. Did you forget to print()?"

        return {
            "success": success,
            "result": result,
            "stdout": stdout,
            "error": error,
        }

    @staticmethod
    def _safe_serialize(obj, depth: int = 0) -> Any:
        """
        Safely serialize an object for JSON transport.
        Handles DataFrames, nested dicts/lists, and prevents infinite recursion.
        """
        if depth > 3:
            return str(obj)
        if isinstance(obj, (int, float, bool, str, type(None))):
            return obj
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return f"<{type(obj).__name__} shape={obj.shape}>"
        if isinstance(obj, dict):
            return {
                k: CodeSandboxService._safe_serialize(v, depth + 1)
                for k, v in obj.items()
                if not str(k).startswith('_')
            }
        if isinstance(obj, (list, tuple)):
            items = [CodeSandboxService._safe_serialize(i, depth + 1) for i in obj[:20]]
            if len(obj) > 20:
                items.append(f"... ({len(obj) - 20} more items)")
            return items
        return str(obj)

    def clear_session(self, session_id: str):
        """Clear a session and release its resources."""
        self._cleanup_session(session_id)


# ---------------------------------------------------------------------------
# Module initialization
# ---------------------------------------------------------------------------
_configure_global_http_identity()
code_sandbox = CodeSandboxService()
