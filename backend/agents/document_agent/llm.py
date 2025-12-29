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
                    self.llm = ChatCerebras(model=self.PROVIDERS['cerebras']['model'])
                    logger.info("Initialized Cerebras LLM")
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

            response = self.llm.invoke(prompt)
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
        """
        if not self.llm:
            return "Error: No LLM provider available"

        try:
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

            response = self.llm.invoke(prompt)
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

                response = self.llm.invoke(prompt)
                return {
                    'success': True,
                    'content': response.content,
                    'type': 'text'
                }

            response = self.llm.invoke(prompt)
            
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
