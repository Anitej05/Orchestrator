
import logging
import traceback
import sys
import io
import pandas as pd
import numpy as np
import requests
import json
import ast
import builtins
from typing import Dict, Any, Optional, List

logger = logging.getLogger("CodeSandboxService")

class CodeSandboxService:
    """
    A unified service for executing Python code in a stateful, semi-sandboxed environment.
    This serves as the "Hands" of the new orchestrator, allowing the LLM to write and run code.
    
    Thread-safe: Uses custom print capture instead of sys.stdout modification.
    """
    
    SAFE_BUILTINS = {
        'len': len,
        'range': range,
        'str': str,
        'int': int,
        'float': float,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'bool': bool,
        'enumerate': enumerate,
        'zip': zip,
        'min': min,
        'max': max,
        'sum': sum,
        'abs': abs,
        'round': round,
        'isinstance': isinstance,
        'type': type,
        'sorted': sorted,
        'Exception': Exception,
        'ValueError': ValueError,
        'TypeError': TypeError,
        'map': map,
        'filter': filter,
        'any': any,
        'all': all,
        'open': open,
        '__import__': __import__,
    }

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("CodeSandboxService initialized")

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            output_buffer = io.StringIO()
            
            def make_print(buffer):
                def custom_print(*args, **kwargs):
                    print(*args, file=buffer, **kwargs)
                return custom_print
            
            self.sessions[session_id] = {
                'globals': {
                    '__builtins__': {**self.SAFE_BUILTINS, 'print': make_print(output_buffer)},
                    'pd': pd,
                    'np': np,
                    'requests': requests,
                    'json': json,
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
            
        result = None
        error = None
        success = False
        new_vars = []
        
        try:
            exec(code, exec_globals)
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
