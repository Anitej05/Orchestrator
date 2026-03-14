# agents/mail_agent/llm.py
"""
Mail Agent - LLM Helper Functions

These are domain-specific helpers that build on top of BaseAgent's llm_* methods.
They add Gmail-specific prompts and parsing logic.
"""
import json
import re
import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger("mail_agent")


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


class GmailLLMHelpers:
    """
    Gmail-specific LLM helpers.

    Mix this into MailAgent to get Gmail-specific LLM methods.
    All methods use BaseAgent's llm_* methods internally.
    """
    
    async def generate_optimized_query(self, vague_query: str) -> str:
        """Generate an optimized Gmail search query from natural language"""
        if not vague_query or vague_query.strip() in ["", "''", '""']:
            logger.warning("[LLM] Empty or invalid query detected")
            return "label:inbox"

        system_prompt = """You are an expert Gmail search assistant with deep knowledge of Gmail operators and search patterns.

=== GMAIL SEARCH OPERATORS ===
- from:email@example.com, to:email@example.com, cc:name, bcc:name
- subject:"exact phrase", "exact phrase anywhere"
- has:attachment, filename:pdf
- after:2024-01-15, before:2024-06-30, newer_than:7d, older_than:1y
- is:unread, is:read, is:starred, is:important
- in:inbox, in:spam, in:trash, label:work
- size:5000000, larger:10M, smaller:1M

=== GUIDELINES ===
1. NO "body:" operator - use plain keywords for body search
2. OR must be UPPERCASE
3. Use quotes for exact phrases when needed
4. Combine operators intelligently for complex queries
5. Return ONLY the optimized query (no explanations)

EXAMPLES:
- "Demo Request" → subject:"Demo Request"
- "emails from John about project" → from:John project
- "unread emails with PDFs" → is:unread has:attachment filename:pdf"""

        response = await self.llm_generate(
            prompt=f'Convert this request into an optimal Gmail search query: "{vague_query}"\n\nQuery:',
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=200,
        )
        
        # Get last non-empty line (usually the query)
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not any(x in line.lower() for x in ['user wants', 'we need', 'they want']):
                return line.strip().strip('"\'`')
        
        return vague_query  # Fallback

    async def summarize_email_content(self, email_text: str) -> str:
        """Generate a summary using recursive map-reduce for large content"""
        if not email_text:
            return "Empty email content."

        if len(email_text) > 4000:
            logger.info(f"[LLM] Email too large ({len(email_text)} chars). Using recursive map-reduce...")
            chunks = [email_text[i:i+3500] for i in range(0, len(email_text), 3500)]
            chunk_results = await asyncio.gather(*[
                self._base_summarize(chunk, is_leaf=True) for chunk in chunks
            ])

            combined_summary = "\n\n".join(chunk_results)
            if len(combined_summary) > 4000:
                return await self.summarize_email_content(combined_summary)
            else:
                return await self._base_summarize(combined_summary, is_leaf=False)
        else:
            return await self._base_summarize(email_text, is_leaf=True)

    async def _base_summarize(self, text: str, is_leaf: bool = True) -> str:
        """Internal summarization logic"""
        instruction = (
            "Create 3-7 insightful bullet points highlighting key information, actions, and context."
            if is_leaf 
            else "Synthesize these partial summaries into a coherent, comprehensive report."
        )
        
        system_prompt = f"""You are an expert email summarizer. Extract key information while preserving important nuances, tone, and context.
        
{instruction}"""

        response = await self.llm_generate(
            prompt=f'Summarize this email content:\n\n"""\n{text}\n"""\n\nSummary:',
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=500,
        )
        
        return strip_think_tags(response) if response else "Failed to summarize."

    async def draft_email_reply(self, thread_content: str, intent: str, sender_name: str = "Sender") -> Dict[str, str]:
        """Draft a reply based on thread context"""
        if not thread_content:
            return {"subject": "Re: Email", "body": "No thread context provided.", "is_html": False}

        if len(thread_content) > 6000:
            thread_context = await self.summarize_email_content(thread_content)
        else:
            thread_context = thread_content

        system_prompt = """You are an expert email communication assistant.
Craft clear, professional emails that match the tone and formality of the thread.

INSTRUCTIONS:
1. Write the email body in CLEAN HTML format (use <p>, <br>, <ul>, <li>, <strong>).
2. Do NOT use Markdown.
3. Address the intent completely and thoughtfully.
4. Return JSON with fields: "subject", "body" (HTML string), "is_html" (boolean true)."""

        response = await self.llm_generate_json(
            prompt=f"""Draft a thoughtful reply to this email thread.

Intent: "{intent}"
Sender: {sender_name}

Thread Context:
\"\"\"
{thread_context}
\"\"\"

JSON:""",
            system_prompt=system_prompt,
            temperature=0.7,
        )
        
        if "subject" in response and "body" in response:
            return response
        
        return {"subject": "Re: Email", "body": "Could not generate draft.", "is_html": False}

    async def extract_actions(self, email_texts: List[str]) -> List[Dict[str, Any]]:
        """Extract action items from emails"""
        if not email_texts:
            return []

        combined = "\n---\n".join(email_texts)

        if len(combined) > 4050:
            chunks = [combined[i:i+4000] for i in range(0, len(combined), 4000)]
            chunk_results = await asyncio.gather(*[
                self._base_extract_actions(chunk) for chunk in chunks
            ])

            all_actions = []
            seen_descriptions = set()
            for chunk_list in chunk_results:
                for action in chunk_list:
                    desc = action.get("description", "").lower().strip()
                    if desc and desc not in seen_descriptions:
                        all_actions.append(action)
                        seen_descriptions.add(desc)
            return all_actions
        else:
            return await self._base_extract_actions(combined)

    async def _base_extract_actions(self, text: str) -> List[Dict[str, Any]]:
        """Internal extraction logic"""
        system_prompt = """You are an expert at analyzing emails and identifying actionable items.

Extract all action items with context and nuance. Focus on:
- Explicit and implicit deadlines
- Direct and indirect requests
- Meeting invitations and scheduling
- Delegated or self-assigned tasks
- Follow-ups and commitments

Return JSON with "actions" list. Each action should include:
- description: Clear action description
- type: todo/meeting/deadline/followup
- priority: high/medium/low (inferred from context)
- source: Email subject or sender
- context: Brief note on why this is important (optional)"""

        response = await self.llm_generate_json(
            prompt=f"Analyze these emails and extract all action items:\n\n\"\"\"\n{text}\n\"\"\"\n\nJSON:",
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1500,
        )
        
        return response.get("actions", []) if response else []
