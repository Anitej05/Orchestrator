"""
Document Agent - LLM Module

Handles document analysis, instruction interpretation, and intelligent editing planning.
Supports multiple LLM providers with automatic fallback.
"""

import logging
import os
from typing import Dict, List, Any, Optional
import json
import re
from pathlib import Path
import sys

# Ensure Orbimesh root is in path for fully qualified imports
orbimesh_root = Path(__file__).resolve().parents[3]
if str(orbimesh_root) not in sys.path:
    sys.path.insert(0, str(orbimesh_root))

from backend.utils.key_manager import get_cerebras_key, report_rate_limit, key_manager

logger = logging.getLogger(__name__)


class DocumentLLMClient:
    """
    LLM client for document analysis and editing planning.
    Provider priority: Cerebras → Groq → NVIDIA → OpenAI (fallback).
    """

    # Provider configuration (priority defined in _load_llm)
    PROVIDERS = {
        'cerebras': {
            'api_key_env': 'CEREBRAS_API_KEY',
            'model': 'gpt-oss-120b',
            'timeout': 30
        },
        'groq': {
            'api_key_env': 'GROQ_API_KEY',
            'model': 'llama-3.1-70b-versatile',
            'timeout': 30
        },
        'nvidia': {
            'api_key_env': 'NVIDIA_API_KEY',
            'model': 'nvidia/nemotron-4-340b-instruct',
            'timeout': 30
        },
        'openai': {
            'api_key_env': 'OPENAI_API_KEY',
            'model': 'gpt-4o-mini',  # fallback
            'timeout': 30
        }
    }

    def __init__(self, prefer_provider: str = 'cerebras'):
        """Initialize LLM client with preferred provider priority."""
        self.prefer_provider = prefer_provider
        self.llm = None
        self._load_llm()

    def _load_llm(self) -> None:
        """Lazy load LLM with provider fallback."""
        # Priority: prefer_provider -> cerebras -> groq -> nvidia -> openai
        ordered = ['cerebras', 'groq', 'nvidia', 'openai']
        providers_to_try = [self.prefer_provider] + [p for p in ordered if p != self.prefer_provider]

        for provider in providers_to_try:
            try:
                if provider == 'cerebras' and os.getenv('CEREBRAS_API_KEY'):
                    from langchain_cerebras import ChatCerebras
                    # Use KeyManager to get an active key
                    self.llm = ChatCerebras(model=self.PROVIDERS['cerebras']['model'], api_key=get_cerebras_key())
                    logger.info("Initialized Cerebras LLM with rotated key")
                    return

                if provider == 'groq' and os.getenv('GROQ_API_KEY'):
                    from langchain_groq import ChatGroq
                    self.llm = ChatGroq(model=self.PROVIDERS['groq']['model'])
                    logger.info("Initialized Groq LLM")
                    return

                if provider == 'nvidia' and os.getenv('NVIDIA_API_KEY'):
                    from langchain_nvidia_ai_endpoints import ChatNVIDIA
                    self.llm = ChatNVIDIA(model=self.PROVIDERS['nvidia']['model'])
                    logger.info("Initialized NVIDIA LLM")
                    return

                if provider == 'openai' and os.getenv('OPENAI_API_KEY'):
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(model=self.PROVIDERS['openai']['model'], timeout=self.PROVIDERS['openai']['timeout'])
                    logger.info("Initialized OpenAI LLM (fallback)")
                    return

            except Exception as e:
                logger.warning(f"Failed to load {provider}: {e}")

        logger.warning("No LLM provider available")

    def _invoke_with_retry(self, prompt: str, max_retries: int = 3) -> Any:
        """
        Invoke LLM with specific handling for Cerebras rate limits (429).
        Rotates keys and retries immediately if limited.
        """
        for attempt in range(max_retries):
            try:
                if not self.llm:
                    raise ValueError("LLM not initialized")
                    
                response = self.llm.invoke(prompt)
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(k in error_str for k in ['429', '413', 'rate_limit', 'too many requests'])
                
                # Check for HTTP status code in exception attributes
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                     if e.response.status_code == 429:
                         is_rate_limit = True

                # Specific handling for Cerebras rotation
                if "cerebras" in str(type(self.llm)).lower() and is_rate_limit:
                    logger.warning(f"⚡ Cerebras Rate Limit (Attempt {attempt+1}/{max_retries}). Rotating key...")
                    
                    # Report current key (best effort to guess which one failed)
                    # For simplicty, just report logical limit to manager
                    # The manager is robust to reporting unknown keys or already-reported keys
                    try:
                        if hasattr(self.llm, 'api_key'):
                            report_rate_limit(self.llm.api_key)
                    except:
                        pass
                        
                    # Get new key (waits if all exhausted)
                    new_key = get_cerebras_key()
                    
                    # Re-init LLM
                    from langchain_cerebras import ChatCerebras
                    self.llm = ChatCerebras(model=self.PROVIDERS['cerebras']['model'], api_key=new_key)
                    logger.info(f"🔄 Switched to new Cerebras key. Retrying...")
                    continue
                
                # If valid non-rate-limit error or retries exhausted
                if attempt == max_retries - 1:
                    raise
                
                logger.warning(f"LLM Invoke failed: {e}. Retrying...")
        
        raise ValueError("Max retries exceeded")

    def interpret_edit_instruction(
        self,
        instruction: str,
        document_content: str,
        document_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interpret natural language instruction and return structured editing plan.
        """
        if not self.llm:
            return {
                'success': False,
                'error': 'No LLM provider available',
                'actions': []
            }

        try:
            prompt = self._build_editing_prompt(
                instruction,
                document_content[:5000],  # Limit context
                document_structure
            )

            response = self._invoke_with_retry(prompt)
            return self._parse_edit_response(response.content)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'actions': []
            }

    def analyze_document_with_query(
        self,
        document_content: str,
        query: str
    ) -> str:
        """
        Analyze document content with a query using RAG context.
        Content can be either full document or RAG-retrieved chunks.
        """
        if not self.llm:
            return "Error: No LLM provider available"

        try:
            # Determine if content is RAG-retrieved chunks or full document
            is_rag_context = "[Source" in document_content
            
            if is_rag_context:
                prompt = f"""
<SYSTEM>
You are a precise document analysis expert. The content below contains relevant chunks retrieved from vector search.
Answer the query using ONLY these retrieved sources:
- If the answer is not in the provided sources, say "No grounded answer found in the document."
- Be concise (2-5 sentences max) and cite source numbers when applicable.
- Do not invent information beyond what's in the sources.
</SYSTEM>

<RETRIEVED_SOURCES>
{document_content[:12000]}
</RETRIEVED_SOURCES>

<QUERY>
{query}
</QUERY>

Respond with a concise, grounded answer based on the retrieved sources.
"""
            else:
                prompt = f"""
<SYSTEM>
You are a precise document analysis expert. Answer strictly from the provided document content.
- If the answer is not present, say "No grounded answer found in the document."
- Be concise (2-5 sentences max) and list concrete items when applicable.
- Do not invent products or facts not in the document.
</SYSTEM>

<DOCUMENT>
{document_content[:12000]}
</DOCUMENT>

<QUERY>
{query}
</QUERY>

Respond with a concise, grounded answer. If listing products, return a bullet list of product names only.
"""

            response = self._invoke_with_retry(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return f"Error: {str(e)}"

    def extract_structured_data(
        self,
        document_content: str,
        extraction_type: str = 'structured'
    ) -> Dict[str, Any]:
        """
        Extract structured data from document.
        """
        if not self.llm:
            return {'success': False, 'error': 'No LLM provider available'}

        try:
            if extraction_type == 'tables':
                prompt = f"""Extract all tables from this document in JSON format.
{document_content[:5000]}

Return JSON with 'tables' key containing list of tables."""

            elif extraction_type == 'structured':
                prompt = f"""Extract and structure key information from this document.
{document_content[:5000]}

Return JSON with keys: 'title', 'sections', 'key_points', 'data'."""

            else:  # 'text'
                prompt = f"""Summarize the key content from this document.
{document_content[:5000]}

Provide 3-5 sentence summary."""

                response = self._invoke_with_retry(prompt)
                return {
                    'success': True,
                    'content': response.content,
                    'type': 'text'
                }

            response = self._invoke_with_retry(prompt)
            
            try:
                data = json.loads(response.content)
                return {
                    'success': True,
                    'data': data,
                    'type': extraction_type
                }
            except json.JSONDecodeError:
                return {
                    'success': True,
                    'data': response.content,
                    'type': extraction_type
                }

        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {'success': False, 'error': str(e)}

    # ========== HELPER METHODS ==========

    def _build_editing_prompt(
        self,
        instruction: str,
        document_content: str,
        structure: Dict[str, Any]
    ) -> str:
        """Build prompt for editing instruction interpretation."""
        return f"""You are a document editing expert. Interpret the following editing instruction and provide structured actions.

Document Structure:
- Total paragraphs: {structure.get('total_paragraphs', 0)}
- Tables: {structure.get('table_count', 0)}
- Headings: {len(structure.get('headings', []))}
- Styles used: {list(structure.get('styles_used', {}).keys())[:5]}

Current Document (first 500 chars):
{document_content[:500]}

Editing Instruction:
{instruction}

Provide response in JSON format:
{{
  "success": true,
  "actions": [
    {{"type": "add_paragraph", "text": "...", "style": "Normal"}},
    {{"type": "format_text", "text": "...", "bold": true}}
  ],
  "reasoning": "Why these actions were chosen"
}}"""

    def _parse_edit_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured actions."""
        try:
            # Clean the response - remove markdown code blocks
            cleaned = response.strip()
            if '```json' in cleaned:
                cleaned = re.sub(r'```json\s*', '', cleaned)
                cleaned = re.sub(r'```\s*$', '', cleaned)
            elif '```' in cleaned:
                cleaned = re.sub(r'```\s*', '', cleaned)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Remove any trailing commas before closing braces
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                return json.loads(json_str)
            
            # If no JSON found, try the entire cleaned response
            try:
                return json.loads(cleaned)
            except:
                pass
            
            # Fallback: return raw response
            logger.warning(f"Could not parse JSON, using fallback. Response: {response[:200]}")
            return {
                'success': True,
                'actions': [],
                'reasoning': response
            }
        except Exception as e:
            logger.error(f"Failed to parse response: {e}. Response: {response[:200]}")
            return {
                'success': False,
                'error': f'Failed to parse LLM response: {str(e)}',
                'actions': []
            }
