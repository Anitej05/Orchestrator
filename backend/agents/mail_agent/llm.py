# agents/mail_agent/llm.py
import json
import re
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from backend.services.inference_service import inference_service, InferencePriority, ProviderType

load_dotenv()
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Refactored properties:
    - Uses Centralized InferenceService for ALL generation
    - Removing manual retry loops (handled by service)
    - Removing manual think-tag stripping (handled by service)
    """
    
    def _parse_json_robustly(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Robustly parses JSON from LLM output, handling markdown fences and empty strings.
        """
        if not text or not isinstance(text, str):
            return None
        
        # 1. Clean markdown fences
        clean_text = re.sub(r'```(?:json)?\s*', '', text)
        clean_text = re.sub(r'\s*```', '', clean_text)
        clean_text = clean_text.strip()
        
        if not clean_text:
            return None
            
        # 2. Try standard load
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # 3. Try to find JSON-like structure if it still fails
            try:
                match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            except:
                pass
        
        logger.warning(f"Failed to parse JSON robustly from: {text[:100]}...")
        return None

    def __init__(self):
        # No client init needed - service is singleton
        pass

    async def generate_optimized_query(self, vague_query: str) -> str:
        """Generate a single optimized Gmail search query combining multiple relevant keywords with OR"""
        
        if not vague_query or vague_query.strip() in ["", "''", '""']:
            return "label:inbox"  # Safe fallback

        prompt = f"""You are a Gmail search query generator. Convert the user's request into a valid Gmail search query.

USER REQUEST: "{vague_query}"

=== GMAIL SEARCH OPERATOR REFERENCE ===
- from:email@example.com
- to:email@example.com
- subject:"exact phrase"
- "exact phrase" (body)
- word1 OR word2 (OR must be uppercase)
- has:attachment
- after:YYYY-MM-DD
- is:unread

=== CRITICAL RULES ===
1. NO "body:" operator.
2. Output ONLY the query string.
3. NO quotes around the output unless part of search syntax.

EXAMPLES:
- "emails from John" -> from:John
- "unread" -> is:unread
"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                temperature=0.2,
                strip_think_tags=True
            )
            
            # Post-processing cleanup (legacy robust logic preserved where simple stripping isn't enough)
            content = re.sub(r'^(Query:|Here is|The query|Gmail query:?|Search query:?|Output:?)\s*', '', content, flags=re.IGNORECASE)
            cleaned = content.strip().strip('`').strip()
            
            # Auto-balance quotes fallback
            if cleaned.count('"') % 2 != 0:
                 if cleaned.rstrip().endswith('"'):
                     cleaned = cleaned.rstrip()[:-1].rstrip()
                 else:
                     cleaned = cleaned + '"'
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return vague_query

    async def summarize_email_content(self, email_text: str, target_length: int = 1500) -> str:
        """Generate a lossless summary using Recursive Map-Reduce if content is large"""
        if not email_text:
            return "Empty email content."
            
        if len(email_text) > 4000:
            logger.info(f"[LOSSLESS] Email too large ({len(email_text)} chars). Using Recursive Map-Reduce...")
            chunks = [email_text[i:i+3500] for i in range(0, len(email_text), 3500)]
            
            chunk_results = await asyncio.gather(*[
                self._base_summarize(chunk, is_leaf=True) for chunk in chunks
            ])
            
            combined_summary = "\n\n".join(chunk_results)
            if len(combined_summary) > 4000:
                return await self.summarize_email_content(combined_summary, target_length)
            else:
                return await self._base_summarize(combined_summary, is_leaf=False)
        else:
            return await self._base_summarize(email_text, is_leaf=True)

    async def _base_summarize(self, text: str, is_leaf: bool = True) -> str:
        prompt = f"""Summarize this text. Focus on facts, actions, and key identifiers.
        {'Use 3-5 high-density bullet points.' if is_leaf else 'Synthesis the following partial summaries into a final coherent report.'}
        
        Text:
        \"\"\"
        {text}
        \"\"\"
        
        Summary:"""
        
        try:
            return await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED, 
                temperature=0.3
            )
        except Exception:
            return "Failed to summarize text."

    async def analyze_urgency(self, email_text: str) -> str:
        """Dedicated pass to extract ONLY deadlines and high-priority requests"""
        if len(email_text) > 8000:
             email_text = email_text[:4000] + "\n[...]\n" + email_text[-4000:]
             
        prompt = f"""SCANNED TEXT:
        \"\"\"
        {email_text}
        \"\"\"
        
        TASK: Extract ONLY explicit deadlines, urgent requests, or pending questions for me.
        If none found, return "No urgent action items detected."
        
        Immediate Attention:"""

        try:
            return await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED, # Reasoning models handled by service config
                temperature=0.1
            )
        except Exception:
            return "Urgency analysis unavailable."

    async def summarize_text_batch(self, texts: List[str]) -> str:
        if not texts:
            return "No content to summarize."
        
        all_raw = "\n\n".join(texts)
        urgent_items = await self.analyze_urgency(all_raw)
        final_summary = await self.summarize_email_content(all_raw)
        
        return f"""=== IMMEDIATE ATTENTION ===
{urgent_items}

=== SUMMARY ===
{final_summary}"""

    async def draft_email_reply(self, thread_content: str, intent: str, sender_name: str) -> Dict[str, str]:
        """Draft a reply based on thread context"""
        if not thread_content:
            return {"subject": "Re: Email", "body": "No thread context provided."}
            
        if len(thread_content) > 6000:
            thread_context = await self.summarize_email_content(thread_content, target_length=3000)
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
2. Do NOT use Markdown (no **, no #, no \\n for newlines in HTML).
3. Be professional but natural.
4. Return JSON with fields: "subject", "body" (the HTML string), and "is_html" (boolean true).
JSON:"""

        # Retry loop for robustness
        for attempt in range(2):
            try:
                content = await inference_service.generate(
                    messages=[HumanMessage(content=prompt)],
                    priority=InferencePriority.QUALITY,
                    temperature=0.7,
                    json_mode=True
                )
                
                data = self._parse_json_robustly(content)
                if data:
                    return data
                
                logger.warning(f"Drafting attempt {attempt + 1} returned unparsable content.")
            except Exception as e:
                logger.error(f"Drafting failed on attempt {attempt + 1}: {e}")
        
        return {"subject": "Re: Email", "body": "Could not generate draft."}

    async def extract_actions(self, email_texts: List[str]) -> List[Dict[str, Any]]:
        if not email_texts: return []
        
        combined = "\n---\n".join(email_texts)
        
        if len(combined) > 4050:
            chunks = [combined[i:i+4000] for i in range(0, len(combined), 4000)]
            chunk_results = await asyncio.gather(*[
                self._base_extract_actions(chunk) for chunk in chunks
            ])
            all_actions = []
            seen = set()
            for lst in chunk_results:
                for item in lst:
                    desc = item.get("description", "").lower().strip()
                    if desc and desc not in seen:
                        all_actions.append(item)
                        seen.add(desc)
            return all_actions
        else:
            return await self._base_extract_actions(combined)

    async def _base_extract_actions(self, text: str) -> List[Dict[str, Any]]:
        prompt = f"""Extract action items from these emails.
        Focus on: deadlines, requests, meetings, and delegated tasks.
        
        Emails:
        \"\"\"
        {text}
        \"\"\"
        
        Return JSON with "actions" list. Each: description, type (todo/meeting/deadline), priority (high/medium/low), source (subject).
        JSON:"""
        
        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                temperature=0.1,
                json_mode=True
            )
            data = self._parse_json_robustly(content)
            return data.get("actions", []) if data else []
        except Exception:
            return []

    async def analyze_attachment_importance(self, email_body: str, attachment_names: List[str]) -> Dict[str, Any]:
        if not attachment_names:
            return {"is_critical": False, "reason": "No attachments"}
        
        # Keyword scan first
        body_lower = email_body.lower()
        critical_keywords = ["see attached", "review proposal", "attached file", "attached document", "invoice attached"]
        if any(kw in body_lower for kw in critical_keywords):
             return {"is_critical": True, "reason": "Direct attachment reference found in body"}
             
        context_body = await self.summarize_email_content(email_body, target_length=2000) if len(email_body) > 4000 else email_body
        
        prompt = f"""Are attachments CRITICAL to understand this email?
        Body: "{context_body}"
        Attachments: {', '.join(attachment_names)}
        Return JSON: {{"is_critical": bool, "reason": "short"}}"""
        
        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                json_mode=True
            )
            return json.loads(content)
        except:
             if "attach" in email_body.lower():
                 return {"is_critical": True, "reason": "Keyword fallback"}
             return {"is_critical": False, "reason": "Analysis failed"}

    async def check_ambiguity(self, query: str) -> Dict[str, Any]:
        if not query or len(query) < 4: return {"is_ambiguous": False}
        
        prompt = f"""Analyze this user query for genuine ambiguity in an email context.
Query: "{query}"

GUIDELINES:
1. ONLY flag "is_ambiguous": true if the request is high-risk OR literally impossible to execute (e.g., "delete his email" when multiple men are in the thread).
2. If one interpretation is standard (e.g., "subject:test" usually implies a search filter), assume that intent and proceed.
3. Ignore theoretical or pedantic ambiguities. If you can make a 90% confident guess on intent, do NOT flag.
4. If a query is clear but broad (e.g., "find emails about work"), that is NOT ambiguous; it is just broad. Proceed with the search.

Return JSON:
{{
  "is_ambiguous": boolean,
  "question": "If true, provide a brief clarifying question",
  "options": ["Specific Option 1", "Specific Option 2"],
  "reasoning": "Internal logic for the decision"
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                json_mode=True
            )
            return json.loads(content)
        except:
            return {"is_ambiguous": False}

    async def decompose_complex_request(self, prompt: str, error_context: Optional[str] = None) -> Dict[str, Any]:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        decomposition_prompt = f"""You are a task planner for an email management pipeline.
Current Time: {current_time}
User Request: "{prompt}"
{f'PREVIOUS ERROR: {error_context}' if error_context else ''}

AVAILABLE ACTIONS:
- search: Search for emails (params: query, max_results)
- summarize: Summarize specific emails (params: message_ids, use_history=True)
- draft_reply: Draft a reply (params: message_id, intent)
- send_email: Send an email (params: to, subject, body)
- manage: Organize emails (params: action="archive"|"delete"|"mark_read"|"star", message_ids, use_history=True)
- extract_actions: Find to-dos in emails (params: use_history=True)
Return JSON with "steps" and "reasoning".
JSON:"""
        
        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=decomposition_prompt)],
                priority=InferencePriority.SPEED,
                json_mode=True
            )
            return json.loads(content)
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return {"steps": [], "reasoning": "Failed to decompose"}

# Global instances
llm_client = LLMClient()
