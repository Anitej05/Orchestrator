import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import json

# Add backend to path if needed
import sys
from pathlib import Path
backend_root = Path(__file__).parent.parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from services.inference_service import inference_service

logger = logging.getLogger(__name__)

class DocumentPlan(BaseModel):
    action: str = Field(..., description="The action to perform: /analyze, /edit, /create, /extract, /display")
    params: Dict[str, Any] = Field(..., description="Parameters for the action")
    reasoning: str = Field(..., description="Reasoning for the choice")

class DocumentPlanner:
    """
    Autonomous planner for the Document Agent.
    Converts natural language prompts into structured actions and parameters.
    """
    
    def __init__(self):
        self.inference = inference_service
        self.model_id = "gpt-oss-120b" # Compatible with Cerebras

    async def plan(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> DocumentPlan:
        """
        Analyze the prompt and return a structured plan.
        """
        logger.info(f"Planning for prompt: {prompt}")
        
        system_prompt = """You are the Planner for a Document Agent.
Your goal is to map a user's natural language request to one of the following actions:

1. `/analyze`: For questions, summaries, or retrieval from documents.
2. `/edit`: For modifying an existing document (add text, format, delete).
3. `/create`: For creating a NEW document from scratch.
4. `/display`: For requesting to view/read a specific document file.
5. `/extract`: For structured data extraction (tables, fields).

Schema for Action Params:
- `/analyze`: {"query": "The question or summary request", "file_path": "optional path"}
- `/edit`: {"file_path": "path to file", "instruction": "what to change"}
- `/create`: {"file_name": "name.docx", "content": "text content", "file_type": "docx|pdf|txt"}
- `/display`: {"file_path": "path to file"}
- `/extract`: {"file_path": "path", "extraction_type": "text|tables|structured"}

Reason step-by-step then provide the JSON plan.
If file paths are mentioned in the prompt, extract them.
If the request is ambiguous, default to `/analyze` with the full prompt as query.
"""

        try:
            # Use structured output for reliability
            plan = await self.inference.generate_structured(
                prompt=prompt,
                response_model=DocumentPlan,
                system_prompt=system_prompt,
                model=self.model_id,
                temperature=0.1
            )
            
            logger.info(f"Generated Plan: {plan.action} - {plan.reasoning}")
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback: Default to analyze
            return DocumentPlan(
                action="/analyze",
                params={"query": prompt},
                reasoning="Planning failed, defaulting to analysis."
            )
