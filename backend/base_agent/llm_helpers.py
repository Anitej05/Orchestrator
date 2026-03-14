"""
Agent LLM Helpers - Standardized LLM methods for all agents.

This mixin provides common LLM operations that every agent needs:
- Text generation
- JSON structured output
- Summarization
- Classification
- Extraction
- Code generation

All methods use the centralized inference_service for consistency.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.inference_service import inference_service, InferencePriority

logger = logging.getLogger(__name__)


def strip_think_tags(text: str) -> str:
    """Remove thinking/reasoning tags from LLM output."""
    if not isinstance(text, str):
        return text
    
    patterns = [
        (r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
        (r'<think>.*$', re.DOTALL | re.IGNORECASE),
        (r'<\|thinking\|>.*?</\|thinking\|>', re.DOTALL | re.IGNORECASE),
        (r'<\|thought\|>.*?</\|thought\|>', re.DOTALL | re.IGNORECASE),
        (r'<thought>.*?</thought>', re.DOTALL | re.IGNORECASE),
        (r'【thinking】.*?【/thinking】', re.DOTALL | re.IGNORECASE),
        (r'<reasoning>.*?</reasoning>', re.DOTALL | re.IGNORECASE),
    ]
    
    for pattern, flags in patterns:
        text = re.sub(pattern, '', text, flags=flags)
    
    return text.strip()


class AgentLLMHelpers:
    """
    Mixin providing standardized LLM methods for all agents.
    
    Usage in any agent:
        class MyAgent(BaseAgent, AgentLLMHelpers):
            async def do_something(self):
                # Generate text
                text = await self.llm_generate("Do something...")
                
                # Get structured JSON
                data = await self.llm_generate_json("Extract data...")
                
                # Summarize content
                summary = await self.llm_summarize(long_text)
    """
    
    # ========================================================================
    # CORE GENERATION METHODS
    # ========================================================================
    
    async def llm_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False,
        priority: InferencePriority = InferencePriority.SPEED,
    ) -> str:
        """
        Generate text using the centralized inference service.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            json_mode: If True, forces JSON output
            priority: Provider selection priority
            
        Returns:
            Generated text (stripped of think tags)
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        response = await inference_service.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            priority=priority,
            strip_markdown=True,
        )
        
        return strip_think_tags(response) if response else ""
    
    async def llm_generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.
        
        Args:
            prompt: User prompt describing what JSON to generate
            system_prompt: Optional system instructions with JSON schema
            temperature: Low temperature for deterministic output
            max_tokens: Maximum tokens
            
        Returns:
            Parsed JSON as dictionary
        """
        if not system_prompt:
            system_prompt = "You are a JSON generator. Return ONLY valid JSON, no explanations."
        
        response = await self.llm_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        
        # Robust JSON parsing
        return self._parse_json_robustly(response)
    
    async def llm_generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> BaseModel:
        """
        Generate structured output using Pydantic schema.
        
        Args:
            prompt: User prompt
            schema: Pydantic model class defining the output structure
            system_prompt: Optional system instructions
            temperature: Low temperature for deterministic output
            
        Returns:
            Pydantic model instance
        """
        from backend.services.inference_service import inference_service as inf_svc
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        result = await inf_svc.generate_structured(
            messages=messages,
            schema=schema,
            temperature=temperature,
        )
        
        return result
    
    # ========================================================================
    # DOMAIN-SPECIFIC HELPER METHODS
    # ========================================================================
    
    async def llm_summarize(
        self,
        text: str,
        max_length: int = 500,
        bullet_points: bool = False,
    ) -> str:
        """
        Summarize text content.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in characters
            bullet_points: If True, return as bullet points
            
        Returns:
            Summary text
        """
        if not text:
            return ""
        
        # For very long text, use chunked summarization
        if len(text) > 4000:
            return await self._summarize_chunked(text, max_length, bullet_points)
        
        format_instruction = (
            "Return as bullet points." if bullet_points 
            else "Return as a concise paragraph."
        )
        
        system_prompt = f"""You are an expert summarizer. Create a clear, concise summary.
{format_instruction}
Maximum {max_length} characters."""
        
        return await self.llm_generate(
            prompt=f"Summarize this text:\n\n{text}",
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=max_length // 4,
        )
    
    async def _summarize_chunked(
        self,
        text: str,
        max_length: int,
        bullet_points: bool,
    ) -> str:
        """Summarize long text using map-reduce."""
        chunk_size = 3500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Map: Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = await self.llm_generate(
                prompt=f"Summarize this text:\n\n{chunk}",
                system_prompt="Create a concise summary.",
                temperature=0.3,
                max_tokens=500,
            )
            chunk_summaries.append(summary)
        
        # Reduce: Combine summaries
        combined = "\n\n".join(chunk_summaries)
        return await self.llm_summarize(combined, max_length, bullet_points)
    
    async def llm_extract(
        self,
        text: str,
        fields: List[str],
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Extract specific fields from text.
        
        Args:
            text: Text to extract from
            fields: List of field names to extract
            description: Optional description of what to extract
            
        Returns:
            Dictionary with extracted fields
        """
        fields_str = ", ".join(fields)
        
        system_prompt = f"""You are an information extraction expert.
Extract ONLY the following fields from the text: {fields_str}
{description}
Return JSON with exactly these fields."""
        
        return await self.llm_generate_json(
            prompt=f"Extract fields from:\n\n{text}",
            system_prompt=system_prompt,
            temperature=0.1,
        )
    
    async def llm_classify(
        self,
        text: str,
        categories: List[str],
        description: str = "",
    ) -> str:
        """
        Classify text into one of the given categories.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            description: Optional description of classification task
            
        Returns:
            Selected category
        """
        categories_str = ", ".join(categories)
        
        system_prompt = f"""You are a classification expert.
Classify the text into ONE of these categories: {categories_str}
{description}
Return ONLY the category name, nothing else."""
        
        return await self.llm_generate(
            prompt=f"Classify this text:\n\n{text}",
            system_prompt=system_prompt,
            temperature=0.0,  # Deterministic
            max_tokens=50,
        )
    
    async def llm_generate_code(
        self,
        instruction: str,
        context: str = "",
        language: str = "python",
        constraints: List[str] = None,
    ) -> str:
        """
        Generate code based on instructions.
        
        Args:
            instruction: What code to generate
            context: Optional context (existing code, data structures, etc.)
            language: Programming language
            constraints: List of constraints/rules to follow
            
        Returns:
            Generated code
        """
        constraints_str = ""
        if constraints:
            constraints_str = "\n\nCONSTRAINTS:\n" + "\n".join(f"- {c}" for c in constraints)
        
        system_prompt = f"""You are an expert {language} developer.
Generate clean, efficient, well-commented code.
Follow best practices for {language}.
Output ONLY the code, no explanations.{constraints_str}"""
        
        context_str = f"\n\nCONTEXT:\n{context}" if context else ""
        
        return await self.llm_generate(
            prompt=f"Generate {language} code for:\n\n{instruction}{context_str}",
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000,
        )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _parse_json_robustly(self, text: str) -> Dict[str, Any]:
        """
        Robustly parse JSON from LLM output.
        
        Handles:
        - Markdown code fences
        - Extra whitespace
        - Partial JSON in text
        """
        if not text or not isinstance(text, str):
            return {}
        
        # Clean markdown fences
        clean_text = re.sub(r'```(?:json)?\s*', '', text)
        clean_text = re.sub(r'\s*```', '', clean_text)
        clean_text = clean_text.strip()
        
        if not clean_text:
            return {}
        
        # Try standard parse
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # Try to extract JSON-like structure
            try:
                match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            except Exception:
                logger.warning(f"Failed to parse JSON from: {text[:200]}...")
                return {}
        
        return {}
