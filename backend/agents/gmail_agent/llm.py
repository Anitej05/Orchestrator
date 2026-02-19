# agents/gmail_agent/llm.py
import os
import json
import re
import asyncio
import logging
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("gmail_agent")

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

def strip_think_tags(text: str) -> str:
    """
    Remove thinking/reasoning tags from LLM output.
    Handles ALL known formats from various models.
    """
    if not isinstance(text, str):
        return text
    
    # Pattern 1: <think>...</think> (closed tags)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 3: Minimax pipe format <|thinking|>...</|thinking|>
    text = re.sub(r'<\|thinking\|>.*?</\|thinking\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\|thinking\|>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 4: Minimax alternate format <|thought|>...</|thought|>
    text = re.sub(r'<\|thought\|>.*?</\|thought\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\|thought\|>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 5: DeepSeek format <thought>...</thought>
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 6: Chinese thinking tags
    text = re.sub(r'【thinking】.*?【/thinking】', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'【thinking】.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 7: <reasoning>...</reasoning>
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Pattern 8: Generic pipe format
    text = re.sub(r'<\|[a-z_]+\|>.*?</\|[a-z_]+\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()

class LLMClient:
    """LLM client for Gmail Agent - handles summarization, drafting, extraction"""
    
    def __init__(self):
        self.clients = []
        
        # Provider order: Cerebras → NVIDIA → Groq
        if CEREBRAS_API_KEY:
            self.clients.append({
                "name": "Cerebras",
                "client": AsyncOpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1"),
                "model": "zai-glm-4.7",
                "summary_model": "llama-3.3-70b"
            })
            
        if NVIDIA_API_KEY:
            self.clients.append({
                "name": "NVIDIA",
                "client": AsyncOpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1"),
                "model": "minimaxai/minimax-m2",
                "summary_model": "llama-3.1-405b-instruct"
            })
            
        if GROQ_API_KEY:
            self.clients.append({
                "name": "Groq",
                "client": AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
                "model": "openai/gpt-oss-120b",
                "summary_model": "llama-3.3-70b-versatile"
            })

    async def generate_optimized_query(self, vague_query: str) -> str:
        """Generate an optimized Gmail search query from natural language"""
        
        if not vague_query or vague_query.strip() in ["", "''", '""']:
            logger.warning("[LLM] Empty or invalid query detected")
            return "label:inbox"

        prompt = f"""You are a Gmail search query generator. Convert the user's request into a valid Gmail search query.

USER REQUEST: "{vague_query}"

=== GMAIL SEARCH OPERATORS ===
- from:email@example.com, to:email@example.com, cc:name, bcc:name
- subject:"exact phrase", "exact phrase anywhere"
- has:attachment, filename:pdf
- after:2024-01-15, before:2024-06-30, newer_than:7d, older_than:1y
- is:unread, is:read, is:starred, is:important
- in:inbox, in:spam, in:trash, label:work
- size:5000000, larger:10M, smaller:1M

=== RULES ===
1. NO "body:" operator - use plain keywords for body search
2. OR must be UPPERCASE
3. Use quotes for exact phrases
4. Prefer simple queries
5. Return ONLY the query string, NO explanation

EXAMPLES:
- "Demo Request" → subject:"Demo Request"
- "emails from John" → from:John
- "unread emails" → is:unread

Query:"""

        for provider in self.clients:
            try:
                response = await provider['client'].chat.completions.create(
                    model=provider['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                
                content = response.choices[0].message.content
                if content:
                    content = strip_think_tags(content)
                    content = re.sub(r'<[^>]*>', '', content)
                    lines = content.strip().split('\n')
                    
                    best_line = None
                    for line in reversed(lines):
                        line = line.strip()
                        if not line or any(x in line.lower() for x in ['user wants', 'we need', 'they want']):
                            continue
                        best_line = line
                        break
                    
                    if best_line:
                        content = best_line
                    
                    content = re.sub(r'^(Query:|Here is|The query|Gmail query:?)\s*', '', content, flags=re.IGNORECASE)
                    cleaned_query = content.strip().strip('"\'`')
                    
                    if len(cleaned_query) > 300 or '<' in cleaned_query:
                        continue
                    
                    logger.info(f"[LLM] Generated query: {cleaned_query}")
                    return cleaned_query
                                
            except Exception as e:
                logger.warning(f"[WARNING] {provider['name']} query generation failed: {e}")
                continue
        
        return vague_query

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
        prompt = f"""Summarize this text. Focus on facts, actions, and key identifiers.
{'Use 3-5 high-density bullet points.' if is_leaf else 'Synthesize the following partial summaries into a final coherent report.'}

Text:
\"\"\"
{text}
\"\"\"

Summary:"""

        for provider in self.clients:
            try:
                model = provider.get('summary_model', provider['model'])
                response = await provider['client'].chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                
                content = response.choices[0].message.content
                if content:
                    return strip_think_tags(content)
            except Exception as e:
                logger.warning(f"[WARNING] {provider['name']} summarize failed: {e}")
                continue
        return "Failed to summarize."

    async def summarize_text_batch(self, texts: List[str]) -> str:
        """Summarize a batch of texts hierarchically"""
        if not texts:
            return "No content to summarize."
        
        all_raw = "\n\n---\n\n".join(texts)
        return await self.summarize_email_content(all_raw)

    async def draft_email_reply(self, thread_content: str, intent: str, sender_name: str = "Sender") -> Dict[str, str]:
        """Draft a reply based on thread context"""
        if not thread_content:
            return {"subject": "Re: Email", "body": "No thread context provided.", "is_html": False}
            
        if len(thread_content) > 6000:
            thread_context = await self.summarize_email_content(thread_content)
        else:
            thread_context = thread_content
            
        prompt = f"""You are a professional Email Agent. Draft a reply to this email.

Intent: "{intent}"
Sender: {sender_name}

Thread Context:
\"\"\"
{thread_context}
\"\"\"

INSTRUCTIONS:
1. Write the email body in CLEAN HTML format (use <p>, <br>, <ul>, <li>, <strong>).
2. Do NOT use Markdown.
3. Be professional but natural.
4. Return JSON with fields: "subject", "body" (HTML string), "is_html" (boolean true).

JSON:"""

        for provider in self.clients:
            try:
                response = await provider['client'].chat.completions.create(
                    model=provider['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                result = json.loads(strip_think_tags(content))
                
                if "subject" not in result or "body" not in result:
                    continue
                
                return result
            except Exception as e:
                logger.warning(f"[WARNING] {provider['name']} draft failed: {e}")
                continue
        
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
        prompt = f"""Extract action items from these emails.
Focus on: deadlines, requests, meetings, and delegated tasks.

Emails:
\"\"\"
{text}
\"\"\"

Return JSON with "actions" list. Each: description, type (todo/meeting/deadline), priority (high/medium/low), source (subject).

JSON:"""

        for provider in self.clients:
            try:
                response = await provider['client'].chat.completions.create(
                    model=provider.get('summary_model', provider['model']),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                data = json.loads(strip_think_tags(content))
                
                if not isinstance(data.get("actions"), list):
                    continue
                
                return data.get("actions", [])
            except Exception as e:
                logger.warning(f"[WARNING] {provider['name']} extract failed: {e}")
                continue
        
        return []

# Global instance
llm_client = LLMClient()
