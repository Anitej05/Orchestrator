# agents/pdf_agent/llm_helpers.py
"""
PDF Agent - LLM Helper Functions

Domain-specific LLM methods for PDF operations.
All methods use inference_service directly.

Preserves ALL original functionality and adds comprehensive LLM-powered features.
"""
import json
import logging
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage
from backend.services.inference_service import inference_service, InferencePriority

logger = logging.getLogger("pdf_agent.llm")


class PDFLLMHelpers:
    """
    PDF-specific LLM helpers.
    
    Mix this into PDFAgent to get PDF-specific LLM methods.
    All methods use inference_service directly.
    """
    
    # ========================================================================
    # PDF ANALYSIS & UNDERSTANDING
    # ========================================================================
    
    async def analyze_pdf_structure(
        self,
        text_content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze PDF structure and provide overview.
        
        Args:
            text_content: Extracted text from PDF
            metadata: PDF metadata (pages, author, etc.)
            
        Returns:
            Structured analysis with document_type, sections, summary
        """
        # Limit content for analysis
        content_sample = text_content[:5000] if len(text_content) > 5000 else text_content
        
        prompt = f"""You are a document analysis expert. Analyze this PDF structure.

METADATA:
- Pages: {metadata.get('pages', 'unknown')}
- Author: {metadata.get('author', 'unknown')}
- Subject: {metadata.get('subject', 'unknown')}

CONTENT SAMPLE:
{content_sample}

Analyze and return JSON:
{{
  "document_type": "research_paper|report|manual|contract|invoice|form|other",
  "confidence": 0.0-1.0,
  "main_sections": ["section 1", "section 2", ...],
  "has_tables": true/false,
  "has_images": true/false,
  "has_charts": true/false,
  "complexity": "simple|moderate|complex",
  "summary": "2-3 sentence summary",
  "key_topics": ["topic 1", "topic 2"]
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                temperature=0.2,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"PDF structure analysis failed: {e}")
            return {
                "document_type": "other",
                "confidence": 0.5,
                "main_sections": [],
                "summary": "Document analysis unavailable",
                "key_topics": []
            }
    
    async def summarize_pdf(
        self,
        text_content: str,
        max_length: int = 500,
        style: str = "executive"
    ) -> str:
        """
        Summarize PDF content.
        
        Args:
            text_content: Extracted text from PDF
            max_length: Maximum summary length in words
            style: Summary style (executive, detailed, technical, etc.)
            
        Returns:
            Summary text
        """
        # Chunk if too large
        if len(text_content) > 8000:
            chunks = [text_content[i:i+4000] for i in range(0, len(text_content), 4000)]
            chunk_summaries = []
            for chunk in chunks:
                summary = await self._summarize_chunk(chunk, max_length // len(chunks), style)
                chunk_summaries.append(summary)
            return "\n\n".join(chunk_summaries)
        
        return await self._summarize_chunk(text_content, max_length, style)
    
    async def _summarize_chunk(
        self,
        text: str,
        max_length: int,
        style: str
    ) -> str:
        """Summarize a single chunk of text."""
        style_instructions = {
            "executive": "Focus on key findings, decisions, and action items. Use bullet points.",
            "detailed": "Provide comprehensive summary covering all main points.",
            "technical": "Emphasize technical details, methodologies, and specifications.",
            "simple": "Explain in simple terms suitable for general audience."
        }
        
        prompt = f"""You are a professional document summarizer.

STYLE: {style}
{style_instructions.get(style, style_instructions['executive'])}
MAXIMUM LENGTH: {max_length} words

TEXT TO SUMMARIZE:
{text[:4000]}

Provide a clear, accurate summary that captures the essential information."""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                temperature=0.3,
                max_tokens=max_length * 2,
            )
            
            return content.strip()
        except Exception as e:
            logger.error(f"Chunk summarization failed: {e}")
            return f"Summary unavailable: {e}"
    
    async def extract_key_information(
        self,
        text_content: str,
        extraction_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Extract key information from PDF.
        
        Args:
            text_content: Extracted text from PDF
            extraction_type: Type of extraction (general, financial, legal, technical, etc.)
            
        Returns:
            Extracted information structured by type
        """
        extraction_prompts = {
            "financial": "Extract: amounts, dates, parties, payment terms, financial metrics",
            "legal": "Extract: parties, obligations, terms, conditions, dates, jurisdictions",
            "technical": "Extract: specifications, parameters, requirements, standards",
            "research": "Extract: hypothesis, methodology, results, conclusions, limitations",
            "general": "Extract: key facts, important dates, main entities, critical information"
        }
        
        prompt = f"""You are an information extraction expert.

DOCUMENT TYPE: {extraction_type}
INSTRUCTIONS: {extraction_prompts.get(extraction_type, extraction_prompts['general'])}

TEXT:
{text_content[:5000]}

Return JSON with extracted information organized by category.
Include confidence scores (0.0-1.0) for each extraction."""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                temperature=0.2,
                json_mode=True,
                max_tokens=2000,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Information extraction failed: {e}")
            return {"error": str(e), "extracted_data": {}}
    
    # ========================================================================
    # PDF Q&A AND QUERYING
    # ========================================================================
    
    async def answer_questions_about_pdf(
        self,
        text_content: str,
        question: str,
        context_window: int = 4000
    ) -> Dict[str, Any]:
        """
        Answer questions about PDF content.
        
        Args:
            text_content: Extracted text from PDF
            question: User's question
            context_window: Context window size
            
        Returns:
            Answer with confidence and source references
        """
        # Find relevant context (simple keyword matching for now)
        question_words = set(question.lower().split())
        sentences = text_content.split('.')
        
        # Score sentences by relevance
        scored_sentences = []
        for i, sent in enumerate(sentences):
            score = len(set(sent.lower().split()) & question_words)
            if score > 0:
                scored_sentences.append((i, score, sent.strip()))
        
        # Get top relevant sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        relevant_context = '. '.join([s[2] for s in scored_sentences[:10]])
        
        prompt = f"""You are a precise document Q&A assistant. Answer questions based ONLY on the provided text.

QUESTION: {question}

RELEVANT CONTEXT:
{relevant_context[:context_window]}

INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. If the answer is not in the context, say "The answer is not in the provided document"
3. Provide confidence score (0.0-1.0)
4. Quote relevant passages when possible

Return JSON:
{{
  "answer": "...",
  "confidence": 0.0-1.0,
  "source_quotes": ["quote 1", "quote 2"],
  "in_document": true/false
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                temperature=0.1,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Q&A failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "in_document": False
            }
    
    # ========================================================================
    # PDF GENERATION & ENHANCEMENT
    # ========================================================================
    
    async def suggest_pdf_improvements(
        self,
        text_content: str,
        document_type: str
    ) -> Dict[str, Any]:
        """
        Suggest improvements for a PDF document.
        
        Args:
            text_content: Extracted text from PDF
            document_type: Type of document
            
        Returns:
            Improvement suggestions with priorities
        """
        prompt = f"""You are a document quality expert. Review this {document_type} and suggest improvements.

DOCUMENT TYPE: {document_type}

CONTENT:
{text_content[:3000]}

Review for:
1. Clarity and readability
2. Structure and organization
3. Completeness
4. Professional tone
5. Formatting suggestions

Return JSON:
{{
  "overall_score": 0-10,
  "strengths": ["strength 1", "strength 2"],
  "improvements": [
    {{
      "category": "clarity|structure|completeness|tone|formatting",
      "priority": "high|medium|low",
      "suggestion": "Specific suggestion",
      "impact": "Expected impact"
    }}
  ],
  "missing_sections": ["section 1", "section 2"]
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                temperature=0.3,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Improvement suggestions failed: {e}")
            return {
                "overall_score": 5,
                "strengths": [],
                "improvements": [],
                "missing_sections": []
            }
    
    async def generate_pdf_metadata(
        self,
        text_content: str
    ) -> Dict[str, str]:
        """
        Generate enhanced metadata for PDF.
        
        Args:
            text_content: Extracted text from PDF
            
        Returns:
            Enhanced metadata suggestions
        """
        prompt = f"""You are a metadata extraction expert. Analyze this document and generate enhanced metadata.

TEXT:
{text_content[:3000]}

Extract and return JSON:
{{
  "suggested_title": "Concise, descriptive title",
  "keywords": ["keyword 1", "keyword 2", "keyword 3"],
  "category": "Business|Technical|Legal|Financial|Academic|Other",
  "language": "en|es|fr|de|etc",
  "audience": "Executive|Technical|General|Legal|Financial",
  "abstract": "2-3 sentence abstract"
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                temperature=0.2,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {
                "suggested_title": "Untitled Document",
                "keywords": [],
                "category": "Other"
            }
