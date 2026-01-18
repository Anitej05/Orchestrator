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
    Handles ALL known formats from various models including:
    - Standard: <think>...</think>
    - Minimax: <|thinking|>...</|thinking|> and similar pipe formats
    - DeepSeek: <thought>...</thought>
    - Chinese: 【thinking】...【/thinking】
    - Reasoning: <reasoning>...</reasoning>
    """
    if not isinstance(text, str):
        return text
    
    # Pattern 1: <think>...</think> (closed tags)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Pattern 2: <think>... (unclosed - GREEDY to end)
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
    
    # Pattern 8: Generic pipe format <|any_tag|>...</|any_tag|> for intermediate reasoning
    text = re.sub(r'<\|[a-z_]+\|>.*?</\|[a-z_]+\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()

class LLMClient:
    """Simple LLM client for generating Gmail search keywords"""
    
    def __init__(self):
        # Initialize clients with fallbacks
        # Using gpt-oss-120b (reasoning model) - strip_think_tags handles it perfectly
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
        """Generate a single optimized Gmail search query combining multiple relevant keywords with OR"""
        
        # Issue #38: Early validation for empty or invalid queries
        if not vague_query or vague_query.strip() in ["", "''", '""']:
            logger.warning("[LLM] Empty or invalid query detected")
            return "label:inbox"  # Safe fallback

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
- after:2024-01-15          → Emails after this date (YYYY-MM-DD format with DASHES)
- before:2024-06-30         → Emails before this date (YYYY-MM-DD format with DASHES)
- older_than:1y             → Older than 1 year (d=day, m=month, y=year)
- newer_than:7d             → Newer than 7 days
- NOTE: Gmail REQUIRES dashes (-) not slashes (/) in dates!

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
3. Use quotes for exact phrases with spaces
4. PREFER SIMPLE QUERIES: For specific requests like "Demo Request", just use subject:"Demo Request"
5. Only use OR patterns when the user asks for multiple things
6. The cc: operator alone returns all emails. Gmail has NO operator for "multiple CC recipients"

=== OUTPUT FORMAT ===
- Return ONLY the Gmail search query string
- NO explanation, NO preamble, NO thinking
- NO outer quotes around the output
- Keep it simple - don't over-engineer the query

EXAMPLES:
- "Demo Request" → subject:"Demo Request"
- "emails from John" → from:John
- "unread emails" → is:unread
- "invoices or receipts" → subject:invoice OR subject:receipt

Query:"""

        MAX_RETRIES = 3
        previous_error = None
        
        for attempt in range(MAX_RETRIES):
            for provider in self.clients:
                try:
                    # Build prompt with error feedback if this is a retry
                    current_prompt = prompt
                    if previous_error:
                        current_prompt = f"""{prompt}

IMPORTANT: Your previous attempt produced an invalid query with this error: {previous_error}
Please avoid this mistake and return a valid, complete Gmail query."""
                    
                    logger.info(f"[LLM] Generating optimized query using {provider['name']} (attempt {attempt + 1}/{MAX_RETRIES})...")
                    response = await provider['client'].chat.completions.create(
                        model=provider['model'],
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=0.2
                    )
                    
                    content = response.choices[0].message.content
                    finish_reason = response.choices[0].finish_reason
                    logger.info(f"[DEBUG] Raw LLM response from {provider['name']} (finish_reason={finish_reason}, len={len(content) if content else 0}): {repr(content)}")
                    
                    if content:
                        # STEP 0: Use centralized strip_think_tags FIRST (matches Orchestrator)
                        content = strip_think_tags(content)
                        logger.debug(f"[DEBUG] After strip_think_tags: {repr(content[:200] if content else 'None')}")
                        
                        # STEP 1: Remove any remaining XML-like tags
                        content = re.sub(r'<[^>]*>', '', content)
                        logger.debug(f"[DEBUG] After XML strip: {repr(content[:200] if content else 'None')}")
                        
                        # STEP 2: Handle multi-line output - find the actual query
                        lines = content.strip().split('\n')
                        logger.debug(f"[DEBUG] Split into {len(lines)} lines: {[l[:50] for l in lines]}")
                        
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
                            logger.info(f"[DEBUG] Selected line: {repr(best_line)}")
                            content = best_line
                        
                        # STEP 3: Remove common preambles
                        content = re.sub(r'^(Query:|Here is|The query|Gmail query:?|Search query:?|Output:?)\s*', '', content, flags=re.IGNORECASE)
                        
                        # STEP 4: Strip outer quotes (but be careful not to break balanced quotes)
                        cleaned_query = content.strip()
                        # Only strip if the entire string is wrapped in quotes
                        if (cleaned_query.startswith('"') and cleaned_query.endswith('"')) or \
                           (cleaned_query.startswith("'") and cleaned_query.endswith("'")):
                            cleaned_query = cleaned_query[1:-1].strip()
                        cleaned_query = cleaned_query.strip('`')
                        
                        # STEP 5: Auto-fix unbalanced quotes (don't reject - just fix)
                        if cleaned_query.count('"') % 2 != 0:
                            logger.info(f"[LLM] Auto-fixing unbalanced quotes in: {cleaned_query[:50]}...")
                            # Remove trailing incomplete quoted phrase or add closing quote
                            if cleaned_query.rstrip().endswith('"'):
                                # Has trailing quote - remove it
                                cleaned_query = cleaned_query.rstrip()[:-1].rstrip()
                            else:
                                # Missing closing quote - add it
                                cleaned_query = cleaned_query + '"'
                        
                        # STEP 6: Validate query quality
                        is_garbage = (
                            len(cleaned_query) > 300 or
                            '<' in cleaned_query or
                            'user wants' in cleaned_query.lower() or
                            'we need' in cleaned_query.lower() or
                            cleaned_query.startswith('The ') or
                            len(cleaned_query) < 3
                        )
                        
                        if is_garbage:
                            previous_error = f"Query was too long or contained invalid content: '{cleaned_query[:50]}...'"
                            logger.warning(f"[WARNING] {provider['name']} produced garbage query, retrying: {cleaned_query[:100]}...")
                            continue  # Try next provider or next attempt
                        
                        logger.info(f"[SUCCESS] Generated single query: {cleaned_query}")
                        return cleaned_query
                                
                except Exception as e:
                    previous_error = str(e)
                    logger.warning(f"[WARNING] {provider['name']} query generation failed: {e}")
                    continue
        
        # Final fallback after all retries exhausted - try simple quote fix
        logger.warning(f"[FALLBACK] All {MAX_RETRIES} retries exhausted. Using original query with quote fix if needed.")
        fallback = vague_query
        if fallback.count('"') % 2 != 0:
            fallback += '"'
        return fallback

    async def summarize_email_content(self, email_text: str, target_length: int = 1500) -> str:
        """Generate a lossless summary using Recursive Map-Reduce if content is large"""
        if not email_text:
            return "Empty email content."
            
        # Implementation of Lossless Recursive Summarization (Issue: Context limits)
        if len(email_text) > 4000:
            logger.info(f"[LOSSLESS] Email too large ({len(email_text)} chars). Using Recursive Map-Reduce...")
            
            # Step 1: Split into manageable chunks (~3500 chars to leave buffer)
            chunks = [email_text[i:i+3500] for i in range(0, len(email_text), 3500)]
            
            # Step 2: Map - Summarize each chunk
            logger.info(f"[MAP] Summarizing {len(chunks)} chunks...")
            chunk_results = await asyncio.gather(*[
                self._base_summarize(chunk, is_leaf=True) for chunk in chunks
            ])
            
            # Step 3: Reduce - Combine and summarize again recursively
            combined_summary = "\n\n".join(chunk_results)
            if len(combined_summary) > 4000:
                return await self.summarize_email_content(combined_summary, target_length)
            else:
                return await self._base_summarize(combined_summary, is_leaf=False)
        else:
            return await self._base_summarize(email_text, is_leaf=True)

    async def _base_summarize(self, text: str, is_leaf: bool = True) -> str:
        """Internal low-level summarization logic"""
        prompt = f"""Summarize this text. Focus on facts, actions, and key identifiers.
        {'Use 3-5 high-density bullet points.' if is_leaf else 'Synthesis the following partial summaries into a final coherent report.'}
        
        Text:
        \"\"\"
        {text}
        \"\"\"
        
        Summary:"""

        for provider in self.clients:
            try:
                # Use Llama 3.3 70b for speed and efficiency as requested
                model = provider.get('summary_model', provider['model'])
                response = await provider['client'].chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=None # Drastically increase/remove as requested
                )
                
                content = response.choices[0].message.content
                if content:
                    return strip_think_tags(content)
            except Exception as e:
                logger.warning(f"[WARNING] {provider['name']} base summarize failed: {e}")
                continue
        return "Failed to summarize part of the text."

    async def analyze_urgency(self, email_text: str) -> str:
        """Dedicated pass to extract ONLY deadlines and high-priority requests (Lossless)"""
        logger.info("[URGENCY] Performing multi-pass extraction for immediate attention items...")
        
        # We also need to process large text for urgency pass
        if len(email_text) > 8000:
             # Fast scan of first and last chunks (where deadlines usually live)
             email_text = email_text[:4000] + "\n[...]\n" + email_text[-4000:]
             
        prompt = f"""SCANNED TEXT:
        \"\"\"
        {email_text}
        \"\"\"
        
        TASK: Extract ONLY explicit deadlines, urgent requests, or pending questions for me.
        If none found, return "No urgent action items detected."
        
        Immediate Attention:"""

        for provider in self.clients:
            try:
                response = await provider['client'].chat.completions.create(
                    model=provider['model'], # Use reasoning model for high precision
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
                if content:
                    return strip_think_tags(content)
            except Exception:
                continue
        return "Urgency analysis unavailable."

    async def summarize_text_batch(self, texts: List[str]) -> str:
        """
        Hierarchically summarize a batch of texts with Immediate Attention pass.
        """
        if not texts:
            return "No content to summarize."
        
        # Pass 1: "Immediate Attention" on the whole raw text
        all_raw = "\n\n".join(texts)
        urgent_items = await self.analyze_urgency(all_raw)
        
        # Pass 2: Lossless Summarization
        final_summary = await self.summarize_email_content(all_raw)
        
        report = f"""=== IMMEDIATE ATTENTION ===
{urgent_items}

=== SUMMARY ===
{final_summary}"""
        return report

    async def draft_email_reply(self, thread_content: str, intent: str, sender_name: str) -> Dict[str, str]:
        """Draft a reply based on thread context (Lossless)"""
        if not thread_content:
            return {"subject": "Re: Email", "body": "No thread context provided."}
            
        # Lossless Context Management for Drafting
        if len(thread_content) > 6000:
            logger.info(f"[LOSSLESS] Thread content too large ({len(thread_content)} chars). Summarizing for context...")
            # Use lossless summarization as the context provider
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
2. Do NOT use Markdown (no **, no #, no \n for newlines in HTML).
3. Be professional but natural.
4. Return JSON with fields: "subject", "body" (the HTML string), and "is_html" (boolean true).

JSON:"""

        MAX_RETRIES = 3
        previous_error = None
        
        for attempt in range(MAX_RETRIES):
            for provider in self.clients:
                try:
                    # Build prompt with error feedback if this is a retry
                    current_prompt = prompt
                    if previous_error:
                        current_prompt = f"""{prompt}

IMPORTANT: Your previous attempt failed with this error: {previous_error}
Please fix this issue and return valid JSON with "subject", "body", and "is_html" fields."""
                    
                    logger.info(f"[DRAFT] Generating reply using {provider['name']} (attempt {attempt + 1}/{MAX_RETRIES})...")
                    response = await provider['client'].chat.completions.create(
                        model=provider['model'],
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=0.7,
                        response_format={"type": "json_object"}
                    )
                    content = response.choices[0].message.content
                    result = json.loads(strip_think_tags(content))
                    
                    # Validate required fields
                    if "subject" not in result or "body" not in result:
                        previous_error = f"Missing required fields. Got keys: {list(result.keys())}"
                        logger.warning(f"[DRAFT] {previous_error} - will retry.")
                        continue
                    
                    return result
                except json.JSONDecodeError as e:
                    previous_error = f"Invalid JSON: {str(e)[:100]}"
                    logger.warning(f"[DRAFT] {previous_error} - will retry.")
                    continue
                except Exception as e:
                    previous_error = str(e)
                    logger.warning(f"[DRAFT] Failed with {provider['name']}: {e}")
                    continue
        
        logger.warning(f"[DRAFT] All {MAX_RETRIES} retries exhausted. Returning fallback.")
        return {"subject": "Re: Email", "body": "Could not generate draft."}

    async def extract_actions(self, email_texts: List[str]) -> List[Dict[str, Any]]:
        """Extract action items from emails without LOSING info (Lossless)"""
        if not email_texts:
            return []
            
        combined = "\n---\n".join(email_texts)
        
        # Recursive Extraction strategy (Issue: Scrapping info)
        if len(combined) > 4050:
            logger.info(f"[LOSSLESS] Extraction batch too large ({len(combined)} chars). Using Recursive Map-Reduce...")
            chunks = [combined[i:i+4000] for i in range(0, len(combined), 4000)]
            
            logger.info(f"[MAP] Extracting from {len(chunks)} chunks...")
            chunk_results = await asyncio.gather(*[
                self._base_extract_actions(chunk) for chunk in chunks
            ])
            
            # Reduce: Flatten and deduplicate
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
        """Internal low-level extraction logic"""
        prompt = f"""Extract action items from these emails.
        Focus on: deadlines, requests, meetings, and delegated tasks.
        
        Emails:
        \"\"\"
        {text}
        \"\"\"
        
        Return JSON with "actions" list. Each: description, type (todo/meeting/deadline), priority (high/medium/low), source (subject).
        
        JSON:"""

        MAX_RETRIES = 3
        previous_error = None
        
        for attempt in range(MAX_RETRIES):
            for provider in self.clients:
                try:
                    # Build prompt with error feedback if this is a retry
                    current_prompt = prompt
                    if previous_error:
                        current_prompt = f"""{prompt}

IMPORTANT: Your previous attempt failed with this error: {previous_error}
Please fix this issue and return valid JSON with an "actions" list."""
                    
                    logger.info(f"[EXTRACT] Extracting actions using {provider['name']} (attempt {attempt + 1}/{MAX_RETRIES})...")
                    response = await provider['client'].chat.completions.create(
                        model=provider.get('summary_model', provider['model']),
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    content = response.choices[0].message.content
                    data = json.loads(strip_think_tags(content))
                    
                    # Validate structure
                    if not isinstance(data.get("actions"), list):
                        previous_error = f"Response missing 'actions' list. Got keys: {list(data.keys())}"
                        logger.warning(f"[EXTRACT] {previous_error} - will retry.")
                        continue
                    
                    return data.get("actions", [])
                except json.JSONDecodeError as e:
                    previous_error = f"Invalid JSON: {str(e)[:100]}"
                    logger.warning(f"[EXTRACT] {previous_error} - will retry.")
                    continue
                except Exception as e:
                    previous_error = str(e)
                    logger.warning(f"[EXTRACT] {provider['name']} base extract failed: {e}")
                    continue
        
        logger.warning(f"[EXTRACT] All {MAX_RETRIES} retries exhausted. Returning empty list.")
        return []

    async def analyze_attachment_importance(self, email_body: str, attachment_names: List[str]) -> Dict[str, Any]:
        """Analyze if attachments are critical without losing context (Lossless)"""
        if not attachment_names:
            return {"is_critical": False, "reason": "No attachments"}
        
        # Lossless keyword scan on whole body first
        body_lower = email_body.lower()
        critical_keywords = ["see attached", "review proposal", "attached file", "attached document", "invoice attached"]
        if any(kw in body_lower for kw in critical_keywords):
             return {"is_critical": True, "reason": "Direct attachment reference found in body (Lossless Scan)"}

        # If keyword scan is ambiguous, use LLM but summarize if needed
        context_body = email_body
        if len(email_body) > 4000:
             context_body = await self.summarize_email_content(email_body, target_length=2000)
        
        prompt = f"""Are attachments CRITICAL to understand this email?
 
Body:
\"\"\"
{context_body}
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

        decomposition_prompt = f"""You are a task planner for an email management pipeline.

Current Time: {current_time}

User Request: "{prompt}"

{f'PREVIOUS PLAN FAILED. ERROR: {error_context}. Adjust your plan accordingly.' if error_context else ''}

=== SELF-RESOLVING PIPELINE ===

Each step can FETCH ITS OWN DATA. You have two options:

**Option A: Search First Pattern (Recommended for multi-step workflows)**
- Step 1: search → finds emails and saves IDs to memory
- Step 2+: use_history: true → uses IDs from Step 1

**Option B: Self-Sufficient Steps (Best for single actions or when history may be irrelevant)**
- Use target_query in ANY step → step fetches its own data inline
- Example: draft_reply with target_query: "Demo Request" → finds the email and replies

The system NEVER fails due to missing IDs - it fetches what it needs.

=== AVAILABLE ACTIONS ===

1. **search** - Find emails (use as Step 1 in multi-step workflows)
   - Params: query (Gmail search), max_results (default 10)
   - Effect: Saves found IDs to memory for subsequent steps

2. **summarize** - Generate a summary
   - Params: use_history: true OR target_query: "search term"

3. **extract_actions** - Extract data points, action items, deadlines
   - Params: use_history: true OR target_query: "search term"
   - Use for: "extract X", "find amounts", "get action items"

4. **draft_reply** - Draft a reply
   - Params: intent: "what to say"
   - Plus: use_history: true OR target_query: "search term"

5. **mark_important** - Star emails
   - Params: use_history: true OR target_query: "search term"

6. **add_labels** - Add Gmail labels
   - Params: labels: ["label1"], use_history: true OR target_query

7. **archive_emails** - Archive emails
   - Params: use_history: true OR target_query

8. **download_attachments** - Download files
   - Params: use_history: true OR target_query

9. **send_email** - Send new email
   - Params: to, subject, body

=== CHOOSING THE RIGHT APPROACH ===

| Scenario | Best Approach |
|----------|---------------|
| "Find X emails and summarize them" | search → summarize(use_history) |
| "Reply to the Demo Request email" | draft_reply(target_query: "Demo Request") |
| "Summarize emails from John" | summarize(target_query: "from:John") |
| "Download attachments from invoice emails" | download(target_query: "invoice has:attachment") |

=== RULES ===

1. For multi-step workflows: Use search first, then use_history: true
2. For single actions: Use target_query for self-sufficient execution
3. NEVER use placeholders like {{{{result[0].id}}}} - use target_query or use_history
4. Extract the user's intent accurately for draft_reply
5. "extract X from emails" = extract_actions (NOT summarize)

=== EXAMPLES ===

Example 1 - Multi-step:
Request: "Find unread emails and summarize them"
{{
  "steps": [
    {{"action": "search", "params": {{"query": "is:unread"}}}},
    {{"action": "summarize", "params": {{"use_history": true}}}}
  ],
  "reasoning": "Search first, then summarize results"
}}

Example 2 - Self-sufficient:
Request: "Reply to the Demo Request email with thank you"
{{
  "steps": [
    {{"action": "draft_reply", "params": {{"target_query": "Demo Request", "intent": "Thank you for your interest"}}}}
  ],
  "reasoning": "Single action - step fetches its own data"
}}

Example 3 - Complex:
Request: "Get emails from John, extract action items, and mark important"
{{
  "steps": [
    {{"action": "search", "params": {{"query": "from:John"}}}},
    {{"action": "extract_actions", "params": {{"use_history": true}}}},
    {{"action": "mark_important", "params": {{"use_history": true}}}}
  ],
  "reasoning": "Multi-step workflow using history"
}}

Return JSON:
{{
    "steps": [...],
    "reasoning": "..."
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
                
                # Handle None or empty content
                if not content:
                    logger.warning(f"[WARNING] [DECOMPOSE] {provider['name']} returned empty content, trying next provider")
                    continue
                
                # Strip thinking tags if present
                content = strip_think_tags(content)
                
                # Check again after stripping (in case all content was thinking)
                if not content or not content.strip():
                    logger.warning(f"[WARNING] [DECOMPOSE] {provider['name']} content was all thinking tags, trying next provider")
                    continue
                
                result = json.loads(content)
                logger.info(f"[DECOMPOSE] Successfully decomposed into {len(result.get('steps', []))} steps")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"[WARNING] [DECOMPOSE] JSON parse error from {provider['name']}: {e}")
                continue
            except Exception as e:
                logger.warning(f"[WARNING] [DECOMPOSE] Provider {provider['name']} failed: {e}")
                continue
        
        # Fallback: Simple keyword-based decomposition
        logger.warning("[WARNING] [DECOMPOSE] All providers failed, using keyword fallback")
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
