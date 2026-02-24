
import logging
import traceback
import sys
import io
import os
import pandas as pd
import numpy as np
import requests
import json
import ast
import builtins
from typing import Dict, Any, Optional, List

logger = logging.getLogger("CodeSandboxService")

# Get backend directory for resolving relative paths
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CodeSandboxService:
    """
    A unified service for executing Python code in a stateful, semi-sandboxed environment.
    This serves as the "Hands" of the new orchestrator, allowing the LLM to write and run code.
    
    Thread-safe: Uses custom print capture instead of sys.stdout modification.
    """
    
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
        # Attribute access
        'getattr': getattr, 'setattr': setattr, 'hasattr': hasattr, 'delattr': delattr,
        # OOP
        'super': super, 'property': property,
        'classmethod': classmethod, 'staticmethod': staticmethod,
        # IO & import
        'open': open, 'print': print, 'input': input,
        '__import__': __import__, '__build_class__': __build_class__,
        # All standard exception types
        'Exception': Exception, 'BaseException': BaseException,
        'ArithmeticError': ArithmeticError, 'LookupError': LookupError,
        'ValueError': ValueError, 'TypeError': TypeError,
        'KeyError': KeyError, 'IndexError': IndexError,
        'AttributeError': AttributeError, 'NameError': NameError,
        'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
        'NotImplementedError': NotImplementedError,
        'OSError': OSError, 'IOError': IOError,
        'FileNotFoundError': FileNotFoundError,
        'FileExistsError': FileExistsError,
        'PermissionError': PermissionError,
        'IsADirectoryError': IsADirectoryError,
        'NotADirectoryError': NotADirectoryError,
        'ZeroDivisionError': ZeroDivisionError,
        'OverflowError': OverflowError,
        'FloatingPointError': FloatingPointError,
        'UnicodeError': UnicodeError,
        'UnicodeDecodeError': UnicodeDecodeError,
        'UnicodeEncodeError': UnicodeEncodeError,
        'ImportError': ImportError,
        'ModuleNotFoundError': ModuleNotFoundError,
        'SyntaxError': SyntaxError,
        'IndentationError': IndentationError,
        'SystemExit': SystemExit,
        'KeyboardInterrupt': KeyboardInterrupt,
        'GeneratorExit': GeneratorExit,
        'AssertionError': AssertionError,
        'EOFError': EOFError,
        'MemoryError': MemoryError,
        'RecursionError': RecursionError,
        'TimeoutError': TimeoutError,
        'ConnectionError': ConnectionError,
        'BrokenPipeError': BrokenPipeError,
        'BufferError': BufferError,
        'Warning': Warning,
        'UserWarning': UserWarning,
        'DeprecationWarning': DeprecationWarning,
        'FutureWarning': FutureWarning,
    }

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            logger.warning("nest_asyncio not installed, async code execution might fail if event loop is already running.")
        logger.info("CodeSandboxService initialized")

    def _normalize_file_path(self, path: str) -> str:
        """
        Normalize file paths to use forward slashes for cross-platform compatibility.
        Converts backslashes to forward slashes.
        """
        if isinstance(path, str):
            return path.replace('\\', '/')
        return path

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            output_buffer = io.StringIO()
            
            def make_print(buffer):
                def custom_print(*args, **kwargs):
                    print(*args, file=buffer, **kwargs)
                return custom_print
            
            # Create wrapper for pd.read_excel that auto-normalizes paths
            def read_excel_safe(path, *args, **kwargs):
                normalized_path = self._normalize_file_path(path)
                logger.info(f"Reading Excel file: {path} (normalized: {normalized_path})")
                return pd.read_excel(normalized_path, *args, **kwargs)
            
            self.sessions[session_id] = {
                'globals': {
                    '__builtins__': {**self.SAFE_BUILTINS, 'print': make_print(output_buffer)},
                    'pd': pd,
                    'np': np,
                    'requests': requests,
                    'json': json,
                    'os': os,  # Provide os module for path operations
                    'BACKEND_DIR': BACKEND_DIR,  # Base directory for resolving paths
                    'normalize_path': self._normalize_file_path,  # Add path normalization utility
                },
                'output_buffer': output_buffer,
                'history': []
            }
        return self.sessions[session_id]

    def execute_code(self, code: str, session_id: str = "default", context_vars: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute python code string in a persistent session.
        
        Args:
            code: The python code to execute.
            session_id: ID to persist state (variables).
            context_vars: Additional variables to inject into this execution.
            
        Returns:
            Dict containing 'success', 'result', 'stdout', 'error', 'new_vars'.
        """
        session = self._get_or_create_session(session_id)
        exec_globals = session['globals']
        output_buffer = session['output_buffer']
        
        # Clear the output buffer for this execution
        output_buffer.seek(0)
        output_buffer.truncate(0)
        
        # Inject context variables if provided
        if context_vars:
            exec_globals.update(context_vars)
            
        if '__name__' not in exec_globals:
            exec_globals['__name__'] = '__main__'
        
        # **CRITICAL FIX: Resolve relative storage paths to absolute paths**
        # This fixes FileNotFoundError when code uses relative paths like 'storage/spreadsheets/file.xlsx'
        code_preprocessed = code
        import re
        
        # 1. Normalize backslashes to forward slashes in storage paths
        code_preprocessed = re.sub(r'([\'\"])(storage[^\'\"]*)\\([^\'\"]*)(\1)', r'\1\2/\3\4', code_preprocessed)
        
        # 2. Convert relative storage paths to absolute paths
        # Match patterns like 'storage/...' or "storage/..." and prepend BACKEND_DIR
        def make_absolute(match):
            quote_open = match.group(1)
            path = match.group(2)
            quote_close = match.group(3)
            # Only convert if it's a relative path (doesn't start with / or drive letter)
            if not path.startswith('/') and not (len(path) > 1 and path[1] == ':'):
                abs_path = os.path.join(BACKEND_DIR, path).replace('\\', '/')
                logger.info(f"[CodePreprocess] Converting relative path: '{path}' -> '{abs_path}'")
                return f'{quote_open}{abs_path}{quote_close}'
            return match.group(0)
        
        code_preprocessed = re.sub(r'([\'"])(storage/[^\'\"]+)([\'"])', make_absolute, code_preprocessed)
        
        logger.info(f"[CodePreprocess] Path resolution applied")
        if code != code_preprocessed:
            logger.info(f"[CodePreprocess] Original: {code[:200]}")
            logger.info(f"[CodePreprocess] Modified: {code_preprocessed[:200]}")
            
        result = None
        error = None
        success = False
        new_vars = []
        
        try:
            import textwrap
            code_to_execute = textwrap.dedent(code_preprocessed)  # Use preprocessed code
            exec(code_to_execute, exec_globals)
            success = True
            
            def safe_serialize(obj, depth=0):
                if depth > 2: return str(obj)
                if isinstance(obj, (int, float, bool, str, type(None))):
                    return obj
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    return f"<{type(obj).__name__} shape={obj.shape}>"
                if isinstance(obj, dict):
                    return {k: safe_serialize(v, depth+1) for k, v in obj.items() if not k.startswith('_')}
                if isinstance(obj, list):
                    return [safe_serialize(i, depth+1) for i in obj[:10]]
                return str(obj)

            if 'result' in exec_globals:
                result = safe_serialize(exec_globals['result'])

            current_keys = set(exec_globals.keys())
                    
        except Exception as e:
            error = str(e)
            logger.error(f"Sandbox execution failed: {e}\n{traceback.format_exc()}")

        # Get captured output
        stdout = output_buffer.getvalue()

        # FEEDBACK: If code ran but produced no output/result, warn the LLM
        if success and not stdout.strip() and result is None:
            stdout = "[SYSTEM WARNING] Code executed successfully but produced NO OUTPUT. Did you forget to print()?"
        
        return {
            "success": success,
            "result": result,
            "stdout": stdout,
            "error": error
        }
        
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

code_sandbox = CodeSandboxService()
