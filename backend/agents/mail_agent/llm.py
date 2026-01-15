# agents/mail_agent/llm.py
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
logger = logging.getLogger(__name__)

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

def strip_think_tags(text: str) -> str:
    """
    Remove thinking/reasoning tags from LLM output.
    This mirrors the Orchestrator's approach for consistency.
    """
    if not isinstance(text, str):
        return text
    
    # Pattern 1: <think>...</think> (closed tags)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Pattern 2: <think>... (unclosed - GREEDY to end, handles all cases)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Pattern 3: Chinese thinking tags
    text = re.sub(r'【thinking】.*?【/thinking】', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'【thinking】.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Pattern 4: Other common patterns
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()

class LLMClient:
    """Simple LLM client for generating Gmail search keywords"""
    
    def __init__(self):
        # Initialize clients with fallbacks
        # Using gpt-oss-120b (reasoning model) - strip_think_tags handles it perfectly
        self.clients = []
        
        if CEREBRAS_API_KEY:
            self.clients.append({
                "name": "Cerebras",
                "client": AsyncOpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1"),
                "model": "gpt-oss-120b"  # Reasoning model - strip_think_tags handles it
            })
            
        if GROQ_API_KEY:
            self.clients.append({
                "name": "Groq",
                "client": AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
                "model": "openai/gpt-oss-120b"  # Same model via Groq
            })
            
        if NVIDIA_API_KEY:
            self.clients.append({
                "name": "NVIDIA",
                "client": AsyncOpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1"),
                "model": "minimaxai/minimax-m2"  # Has intermediate reasoning - strip_think_tags handles it
            })

    async def generate_optimized_query(self, vague_query: str) -> str:
        """Generate a single optimized Gmail search query combining multiple relevant keywords with OR"""
        
        # Comprehensive Gmail search syntax reference (based on official Google documentation)
        prompt = f"""You are a Gmail search query generator. Convert the user's request into a valid Gmail search query.

USER REQUEST: "{vague_query}"

=== GMAIL SEARCH OPERATOR REFERENCE ===

SENDER/RECIPIENT OPERATORS:
- from:email@example.com    → Emails from this sender
- to:email@example.com      → Emails sent to this recipient  
- cc:name                   → Emails where someone is CC'd
- bcc:name                  → Emails where someone is BCC'd

CONTENT OPERATORS:
- subject:word              → Word appears in subject line
- subject:"exact phrase"    → Exact phrase in subject
- "exact phrase"            → Exact phrase anywhere in email
- word1 word2               → Both words appear (implicit AND)
- word1 OR word2            → Either word appears (OR must be uppercase)
- -word                     → Exclude emails containing this word
- +word                     → Word must appear exactly as written
- word1 AROUND n word2      → Words appear within n words of each other

ATTACHMENT OPERATORS:
- has:attachment            → Has any attachment
- filename:pdf              → Attachment with this extension
- filename:report.pdf       → Specific attachment filename
- has:drive                 → Has Google Drive attachment
- has:document              → Has Google Docs attachment
- has:spreadsheet           → Has Google Sheets attachment

DATE OPERATORS:
- after:2024/01/15          → Emails after this date (YYYY/MM/DD)
- before:2024/06/30         → Emails before this date
- older_than:1y             → Older than 1 year (d=day, m=month, y=year)
- newer_than:7d             → Newer than 7 days

STATUS OPERATORS:
- is:unread                 → Unread emails
- is:read                   → Read emails
- is:starred                → Starred emails
- is:important              → Marked as important
- is:snoozed                → Snoozed emails

LOCATION OPERATORS:
- in:inbox                  → In inbox
- in:spam                   → In spam folder
- in:trash                  → In trash
- in:anywhere               → Search all folders including spam/trash
- label:work                → Has this label
- category:promotions       → In this category (primary, social, promotions, updates, forums)

SIZE OPERATORS:
- size:5000000              → Larger than 5MB (in bytes)
- larger:10M                → Larger than 10MB
- smaller:1M                → Smaller than 1MB

MAILING LIST:
- list:info@mailinglist.com → From this mailing list

=== CRITICAL RULES ===
1. There is NO "body:" operator. To search email body, use plain keywords without any operator.
2. OR must be UPPERCASE
3. Use quotes for exact phrases
4. Combine operators: from:boss@company.com subject:urgent has:attachment
5. Group alternatives with OR: from:alice OR from:bob
6. Use parentheses for complex queries: (from:alice OR from:bob) subject:meeting

=== OUTPUT FORMAT ===
- Return ONLY the Gmail search query string
- NO explanation, NO preamble, NO thinking
- NO quotes around the output
- Example output: idiom OR subject:idiom OR "common idiom"

Query:"""

        for provider in self.clients:
            try:
                logger.info(f"🤖 Generating optimized query using {provider['name']}...")
                # No timeout - let LLM respond fully, switch provider only on error
                response = await provider['client'].chat.completions.create(
                    model=provider['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                
                content = response.choices[0].message.content
                logger.debug(f"🔍 Raw LLM response from {provider['name']}: {content[:200] if content else 'None'}...")
                
                if content:
                    # STEP 0: Use centralized strip_think_tags FIRST (matches Orchestrator)
                    content = strip_think_tags(content)
                    
                    # STEP 1: Remove any remaining XML-like tags
                    content = re.sub(r'<[^>]*>', '', content)
                    
                    # STEP 2: Handle multi-line output - find the actual query
                    lines = content.strip().split('\n')
                    best_line = None
                    for line in reversed(lines):
                        line = line.strip()
                        # Skip empty lines and lines with thinking keywords
                        if not line:
                            continue
                        if any(x in line.lower() for x in ['user wants', 'we need', 'they want', 'so the query', 'the request']):
                            continue
                        # Found a potential query line
                        best_line = line
                        break
                    
                    if best_line:
                        content = best_line
                    
                    # STEP 3: Remove common preambles
                    content = re.sub(r'^(Query:|Here is|The query|Gmail query:?|Search query:?|Output:?)\s*', '', content, flags=re.IGNORECASE)
                    
                    # STEP 4: Strip outer quotes
                    cleaned_query = content.strip().strip('"').strip("'").strip('`')
                    
                    # STEP 5: Validate query quality
                    is_garbage = (
                        len(cleaned_query) > 200 or
                        '<' in cleaned_query or
                        'user wants' in cleaned_query.lower() or
                        'we need' in cleaned_query.lower() or
                        cleaned_query.startswith('The ') or
                        len(cleaned_query) < 3
                    )
                    
                    if is_garbage:
                        logger.warning(f"⚠️ {provider['name']} produced garbage query, trying next provider: {cleaned_query[:100]}...")
                        continue  # TRY NEXT PROVIDER instead of returning fallback
                    
                    logger.info(f"✅ Generated single query: {cleaned_query}")
                    return cleaned_query
                            
            except Exception as e:
                logger.warning(f"⚠️ {provider['name']} query generation failed: {e}")
                continue
                
        # Hard fallback
        return vague_query

    async def summarize_email_content(self, email_text: str) -> str:
        """Generate a concise summary of the provided email content"""
        # Truncate to 2500 chars (~600 tokens) to save costs
        truncated = email_text[:2500]
        
        prompt = f"""Summarize in 3-5 bullet points. Focus on key info, actions, dates.

Content:
\"\"\"
{truncated}
\"\"\"

Summary:"""

        for provider in self.clients:
            try:
                logger.info(f"🤖 Summarizing email using {provider['name']}...")
                # No timeout - let LLM respond fully, switch provider only on error
                response = await provider['client'].chat.completions.create(
                    model=provider['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                
                content = response.choices[0].message.content
                if content:
                    # Clean up thinking content - handle CLOSED tags
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'【thinking】.*?【/thinking】', '', content, flags=re.DOTALL | re.IGNORECASE)
                    
                    # Handle UNCLOSED <think> tags - remove everything from <think> to end
                    content = re.sub(r'<think>.*', '', content, flags=re.DOTALL | re.IGNORECASE)
                    
                    # Remove any remaining XML-like tags
                    content = re.sub(r'<[^>]+>', '', content)
                    
                    return content.strip()
                        
            except Exception as e:
                logger.warning(f"⚠️ {provider['name']} summarization failed: {e}")
                continue
                
        return "Failed to generate summary."

    async def summarize_text_batch(self, texts: List[str]) -> str:
        """
        Hierarchically summarize a batch of texts (e.g., multiple emails).
        Uses a Map-Reduce approach if content is large.
        """
        if not texts:
            return "No content to summarize."
        
        # 1. Map: Summarize each text individually if it's long, or chunk them
        # For simplicity, we'll join them until they hit a limit, then summarize chunks
        
        combined_text = "\n\n---\n\n".join(texts)
        
        # If total length is manageable (e.g. < 12000 chars ~ 3k tokens), summarize directly
        if len(combined_text) < 12000:
            return await self.summarize_email_content(combined_text)
        
        # 2. Reduce: Chunking strategy
        logger.info(f"📚 Batch too large ({len(combined_text)} chars). Performing hierarchical summarization...")
        
        chunk_summaries = []
        current_chunk = ""
        
        for text in texts:
            if len(current_chunk) + len(text) > 8000:
                # Summarize current chunk
                summary = await self.summarize_email_content(current_chunk)
                chunk_summaries.append(summary)
                current_chunk = text
            else:
                current_chunk += "\n\n" + text
                
        # Process last chunk
        if current_chunk:
            summary = await self.summarize_email_content(current_chunk)
            chunk_summaries.append(summary)
            
        # 3. Final Reduce: Summarize the summaries
        combined_summaries = "Here are the summaries of the individual parts:\n\n" + "\n\n".join(chunk_summaries)
        return await self.summarize_email_content(combined_summaries)

    async def draft_email_reply(self, thread_content: str, intent: str, sender_name: str) -> Dict[str, str]:
        """Draft a reply based on thread context and user intent"""
        # Truncate thread to ~600 tokens to save costs
        truncated = thread_content[:2500]
        
        prompt = f"""You are a professional Email Agent. Draft a reply to this email.
        
Intent: "{intent}"
Sender: {sender_name}

Thread:
\"\"\"
{truncated}
\"\"\"

INSTRUCTIONS:
1. Write the email body in CLEAN HTML format (use <p>, <br>, <ul>, <li>, <strong>).
2. Do NOT use Markdown (no **, no #, no \n for newlines in HTML).
3. Be professional but natural.
4. Return JSON with fields: "subject", "body" (the HTML string), and "is_html" (boolean true).

JSON:"""

        for provider in self.clients:
            try:
                # No timeout - let LLM respond fully, switch provider only on error
                response = await provider['client'].chat.completions.create(
                    model=provider['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                logger.warning(f"Drafting failed with {provider['name']}: {e}")
                continue
        
        return {"subject": "Re: Email", "body": "Could not generate draft."}

    async def extract_actions(self, email_texts: List[str]) -> List[Dict[str, Any]]:
        """Extract action items, deadlines, and meetings from emails"""
        # Truncate to ~750 tokens
        combined = "\n---\n".join(email_texts)[:3000]
        
        prompt = f"""Extract action items from these emails.

Emails:
\"\"\"
{combined}
\"\"\"

Return JSON with "actions" list. Each: description, type (todo/meeting/deadline), priority (high/medium/low), source (subject).

JSON:"""

        for provider in self.clients:
            try:
                # No timeout - let LLM respond fully, switch provider only on error
                response = await provider['client'].chat.completions.create(
                    model=provider['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                data = json.loads(content)
                return data.get("actions", [])
            except Exception:
                continue
        return []

    async def analyze_attachment_importance(self, email_body: str, attachment_names: List[str]) -> Dict[str, Any]:
        """Analyze if attachments are critical to the email's purpose"""
        if not attachment_names:
            return {"is_critical": False, "reason": "No attachments"}
        
        # Truncate to ~500 tokens
        truncated = email_body[:2000]
        
        prompt = f"""Are attachments CRITICAL to understand this email?

Body:
\"\"\"
{truncated}
\"\"\"

Attachments: {', '.join(attachment_names)}

Critical if: "see attached", "review proposal", empty body with just file.
Not critical if: body summarizes content, attachments are logos/signatures.

Return JSON: {{"is_critical": bool, "reason": "short"}}
JSON:"""

        for provider in self.clients:
            try:
                # No timeout - let LLM respond fully, switch provider only on error
                response = await provider['client'].chat.completions.create(
                    model=provider['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception:
                continue
        
        # Fallback: aggressive default if LLM fails but "attach" is in text
        if "attach" in email_body.lower():
            return {"is_critical": True, "reason": "Keyword fallback detection"}
        return {"is_critical": False, "reason": "Analysis failed, assuming non-critical"}

    async def decompose_complex_request(self, prompt: str, error_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Decompose a complex natural language request into executable steps.
        
        Example input: "Find emails from John, summarize them, and mark important ones"
        Example output: {
            "steps": [
                {"action": "search", "params": {"query": "from:John"}},
                {"action": "summarize", "params": {"use_history": true}},
                {"action": "mark_important", "params": {"use_history": true}}
            ]
        }
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        decomposition_prompt = f"""You are a task decomposition expert for an email management system.

Current Time: {current_time}

Break down this complex request into simple, sequential steps that can be executed one at a time.

User Request: "{prompt}"

{f'PREVIOUS PLAN FAILED. ERROR: {error_context}. Please adjust your plan to avoid this error.' if error_context else ''}

Available actions:
- search: Find emails (params: query, max_results)
- summarize: Summarize emails (params: message_ids or use_history=true for previous search)
- mark_important: Star/mark emails as important (params: message_ids or use_history=true)
- add_labels: Add labels to emails (params: message_ids, labels)
- draft_reply: Draft a reply (params: message_id, intent)
- extract_actions: Extract action items (params: message_ids or use_history=true)
- archive_emails: Archive emails (params: message_ids or use_history=true)
- download_attachments: Download attachments from email (params: message_id or use_history=true)
- send_email: Send email with optional attachments (params: to, subject, body, attachment_paths)

Rules:
1. Start with 'search' if the user wants to find/filter emails
2. Use 'use_history: true' to reference results from the previous step
3. Keep steps atomic - one action per step
4. Extract specific parameters from the user's request (names, keywords, etc.)
5. If user asks for "all", "everything", or a specific number of emails, set 'max_results' in the search step accordingly (default to 10 if unspecified, map "all" to 50).

Return JSON:
{{
    "steps": [
        {{"action": "action_name", "params": {{"param1": "value1"}}}},
        ...
    ],
    "reasoning": "Brief explanation of the decomposition"
}}

JSON:"""

        for provider in self.clients:
            try:
                response = await provider['client'].chat.completions.create(
                    model=provider['model'],
                    messages=[{"role": "user", "content": decomposition_prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                # Strip thinking tags if present
                content = strip_think_tags(content)
                result = json.loads(content)
                logger.info(f"🧠 [DECOMPOSE] Successfully decomposed into {len(result.get('steps', []))} steps")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ [DECOMPOSE] JSON parse error from {provider['name']}: {e}")
                continue
            except Exception as e:
                logger.warning(f"⚠️ [DECOMPOSE] Provider {provider['name']} failed: {e}")
                continue
        
        # Fallback: Simple keyword-based decomposition
        logger.warning("⚠️ [DECOMPOSE] All providers failed, using keyword fallback")
        steps = []
        prompt_lower = prompt.lower()
        
        # Detect search intent
        if any(kw in prompt_lower for kw in ["find", "search", "get", "from", "emails"]):
            # Extract sender name if present
            query = prompt  # Use full prompt as fallback query
            steps.append({"action": "search", "params": {"query": query, "max_results": 10}})
        
        # Detect summarize intent
        if any(kw in prompt_lower for kw in ["summarize", "summary", "summarise"]):
            steps.append({"action": "summarize", "params": {"use_history": True}})
        
        # Detect mark/label intent
        if any(kw in prompt_lower for kw in ["mark", "important", "star", "label", "flag"]):
            steps.append({"action": "mark_important", "params": {"use_history": True}})
        
        return {"steps": steps, "reasoning": "Keyword-based fallback decomposition"}

# Global instance
llm_client = LLMClient()
