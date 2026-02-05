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
from backend.services.inference_service import inference_service, InferencePriority
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class DocumentLLMClient:
    """
    LLM client for document analysis and editing planning.
    Uses centralized InferenceService.
    """

    def __init__(self):
        """Initialize LLM client."""
        pass

    async def interpret_edit_instruction(
        self,
        instruction: str,
        document_content: str,
        document_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interpret natural language instruction and return structured editing plan.
        """
        try:
            prompt = self._build_editing_prompt(
                instruction,
                document_content[:5000],  # Limit context
                document_structure
            )

            response = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                strip_markdown=True
            )
            return self._parse_edit_response(response)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'actions': []
            }


    async def analyze_document_with_query(
        self,
        document_content: str,
        query: str
    ) -> str:
        """
        Analyze document content with a query using RAG context.
        Content can be either full document or RAG-retrieved chunks.
        """
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

            response = await inference_service.generate(messages=[HumanMessage(content=prompt)])
            return response

        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return f"Error: {str(e)}"

    async def extract_structured_data(
        self,
        document_content: str,
        extraction_type: str = 'structured'
    ) -> Dict[str, Any]:
        """
        Extract structured data from document.
        """
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

                response = await inference_service.generate(messages=[HumanMessage(content=prompt)])
                return {
                    'success': True,
                    'content': response,
                    'type': 'text'
                }

            response_content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                strip_markdown=True
            )
            
            try:
                # Clean up JSON if needed
                cleaned = response_content.strip()
                if '```json' in cleaned:
                    cleaned = re.sub(r'```json\s*', '', cleaned)
                    cleaned = re.sub(r'```\s*$', '', cleaned)
                
                data = json.loads(cleaned)
                return {
                    'success': True,
                    'data': data,
                    'type': extraction_type
                }
            except json.JSONDecodeError:
                return {
                    'success': True,
                    'data': response_content,
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
