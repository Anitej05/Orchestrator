"""
Spreadsheet Agent v3.0 - LLM Client

LLM integration with task decomposition and pandas code generation.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any

from .config import LLM_PROVIDERS, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
from .schemas import ExecutionPlan, StepPlan

logger = logging.getLogger("spreadsheet_agent.llm")


class LLMClient:
    """
    LLM client with task decomposition and code generation.
    Supports multiple providers with fallback.
    """
    
    def __init__(self):
        self.providers = self._init_providers()
        logger.info(f"LLMClient initialized with {len(self.providers)} providers")
    
    def _init_providers(self) -> List[Dict]:
        """Initialize available LLM providers."""
        available = []
        
        for provider in LLM_PROVIDERS:
            if provider.get('api_key'):
                available.append(provider)
                logger.info(f"LLM provider available: {provider['name']}")
        
        return available
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    async def decompose_request(
        self,
        prompt: str,
        context: Dict[str, Any],
        error_context: str = None
    ) -> ExecutionPlan:
        """
        Decompose a complex request into executable steps.
        
        Example input: "Upload sales.csv, calculate revenue by region, add totals"
        Example output: ExecutionPlan with steps for load, aggregate, add_totals
        """
        system_prompt = self._build_decomposition_prompt(context, error_context)
        
        response = await self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])
        
        plan = self._parse_plan(response)
        
        # DEBUG: Log what plan was generated
        logger.info(f"[LLM DECOMPOSE] Prompt: {prompt[:100]}...")
        logger.info(f"[LLM DECOMPOSE] Plan: {len(plan.steps)} steps, needs_clarification={plan.needs_clarification}")
        if plan.steps:
            logger.info(f"[LLM DECOMPOSE] Steps: {[s.action for s in plan.steps]}")
        
        return plan
    
    async def generate_pandas_code(
        self,
        instruction: str,
        df_context: str
    ) -> str:
        """
        Generate safe pandas code for a transformation.
        """
        system_prompt = f"""You are a pandas code generator. Generate ONLY executable pandas code.

DATAFRAME CONTEXT:
{df_context}

RULES:
1. The DataFrame is available as 'df'
2. Output ONLY the code, no explanations
3. Always assign result back to 'df' if modifying
4. Use .copy() when creating subsets to avoid SettingWithCopyWarning
5. Handle missing values gracefully
6. Never use inplace=True for operations that return None
7. DO NOT assume existence of other variables (like df_last, df2) - only 'df' exists
8. To SAVE a file, use: `path = save_spreadsheet(df, 'filename.xlsx')` (returns the path)
9. If saving, assign the path to 'result' (e.g. `result = save_spreadsheet(df, 'file.xlsx')`)

OUTPUT FORMAT:
```python
# Your code here
df = ...
```"""
        
        response = await self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ])
        
        return self._extract_code(response)
    
    async def answer_question(
        self,
        question: str,
        df_context: str,
        history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Answer a question about the data.
        """
        system_prompt = f"""You are a data analyst. Answer questions about the spreadsheet data.

DATAFRAME CONTEXT:
{df_context}

RULES:
1. Be concise and accurate
2. If you need to compute something, provide the pandas code in the "code" field
3. If the data doesn't contain the answer, say so
4. Format numbers nicely (e.g., $1,234.56, 45.2%)
5. For aggregations, show the calculation in the code field
6. **CRITICAL**: The "answer" field MUST contain the ACTUAL answer or result, NOT instructions on how to get it
7. If you provide code that computes a value, the "answer" should state what you expect the result to be
8. NEVER say "you can get X by running..." - instead say "X is [value]" or provide the code to compute it

OUTPUT FORMAT (JSON):
{{
    "answer": "The answer or result here (NOT instructions)",
    "code": "result = df['column'].nunique()  # Use 'result' variable for computed values",
    "confidence": 0.0-1.0
}}

EXAMPLES:
- Question: "How many unique products?" -> answer: "There are 250 unique products", code: "result = df['item_name'].nunique()"
- Question: "What is the total revenue?" -> answer: "Total revenue is $1,234,567.89", code: "result = df['gross'].sum()"
- Question: "Show top 5 items" -> answer: "Top 5 items by revenue:", code: "result = df.nlargest(5, 'gross')[['item_name', 'gross']]"
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history if provided
        if history:
            for h in history[-3:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": h.get('question', '')})
                messages.append({"role": "assistant", "content": h.get('answer', '')})
        
        messages.append({"role": "user", "content": question})
        
        response = await self._call_llm(messages)
        
        return self._parse_answer(response, question)
    
    
    async def analyze_query_intent(
        self,
        query: str,
        schema_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM-powered query intent analysis - NO HARDCODED KEYWORDS!
        
        The LLM decides:
        1. What search/filter terms are relevant
        2. Whether to filter the dataset or use full data
        3. Which sampling strategy to use
        
        This replaces brittle keyword matching with intelligent analysis.
        """
        system_prompt = f"""You are a data query analyzer. Analyze the user's query and the dataset schema to determine how to find relevant data.

DATASET SCHEMA:
{json.dumps(schema_info, indent=2)}

Your task: Analyze the query and extract:
1. Search terms: Specific values/keywords the user is looking for in the data
2. Filtering needed: Whether we need to filter the dataset or use all rows
3. Sampling strategy: How to sample the data if needed

OUTPUT AS JSON:
{{
    "search_terms": ["term1", "term2"],  // Specific values to search for (empty if none)
    "target_columns": ["col1", "col2"],  // Which columns to search in (empty for all)
    "needs_filtering": true/false,       // Whether to filter dataset
    "row_references": [],                // Specific row numbers mentioned (0-indexed)
    "sampling_strategy": "full|distribution|stratified",  // How to sample
    "reasoning": "Brief explanation of your analysis"
}}

IMPORTANT:
- Only include search terms if the user explicitly mentions specific values
- "sum", "count", "average" are NOT search terms - they're operations
- For aggregations (sum/count/avg), use sampling_strategy: "distribution"
- For searches ("find X"), use needs_filtering: true and list search terms
- For lookups ("row 5"), use row_references
"""

        response = await self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ])
        
        # Parse JSON response
        try:
            # Extract JSON from response (may be wrapped in ```json```)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                json_str = json_match.group(0) if json_match else '{}'
            
            analysis = json.loads(json_str)
            logger.info(f"LLM query analysis: {analysis.get('reasoning', 'No reasoning')}")
            return analysis
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM query analysis: {e}")
            # Fallback: conservative approach - no filtering
            return {
                "search_terms": [],
                "target_columns": [],
                "needs_filtering": False,
                "row_references": [],
                "sampling_strategy": "stratified",
                "reasoning": "Parse error, using safe defaults"
            }
    

    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    def _build_decomposition_prompt(
        self,
        context: Dict[str, Any],
        error_context: str = None
    ) -> str:
        """Build the system prompt for task decomposition."""
        
        has_data = context.get('has_data', False)
        columns = context.get('columns', [])
        data_preview = context.get('data_preview')  # NEW: Actual data!
        history = context.get('history', [])
        
        # Build data context section
        if data_preview:
            # LLM sees ACTUAL DATA - can make intelligent decisions!
            # CRITICAL: Explicitly tell the LLM that data is ALREADY loaded
            data_context = f"""
**DATA IS ALREADY LOADED - DO NOT USE load_file ACTION**

DATA PREVIEW:
{data_preview}

This shows you ACTUAL data from the spreadsheet that is already loaded in memory.
Use this to understand what columns exist and what values they contain.
Proceed directly with 'process' steps - the file is already available.
"""
        else:
            # Fallback to column names only
            if has_data:
                data_context = f"""
**DATA IS ALREADY LOADED - DO NOT USE load_file ACTION**
- Available columns: {columns[:20] if columns else 'None'}
"""
            else:
                data_context = f"""
- Has loaded data: {has_data}
- Available columns: {columns[:20] if columns else 'None'}
"""
        
        prompt = f"""You are a spreadsheet task planner. Break complex requests into simple executable steps.

CONTEXT:
{data_context}
- Recent operations: {[h.get('action') for h in history[-5:]] if history else 'None'}

AVAILABLE ACTIONS:
- load_file: Load a spreadsheet file (params: filename) - ONLY use if no data is loaded!
- process: Perform ANY data operation using pandas. Describe what you want in natural language. (params: instruction)
  Examples: "count unique values in column X", "filter rows where A > 100", "calculate sum of B grouped by C", "add new column D = A * B", "sort by date descending"
- export: Save file (params: filename, format)

RULES:
1. Use 'process' for ALL data operations - it has full pandas freedom
2. **CRITICAL**: If data is ALREADY LOADED (see DATA PREVIEW above), DO NOT include a load_file step. Skip directly to 'process'.
3. Only use load_file if the context explicitly says no data is loaded
4. If unclear, set needs_clarification=true with a question
5. For destructive operations, set requires_confirmation=true
6. **CRITICAL: ONLY use columns that exist in the data.** Refer to the DATA PREVIEW to see actual column names.
7. **MANDATORY**: If data IS loaded, your plan must ONLY contain 'process' (or 'export') steps, NOT 'load_file'.
8. Look for column synonyms: 'revenue' might be 'gross', 'sales', 'total', 'amount'; 'products' might be 'item_name', 'product_name', 'name'.

OUTPUT FORMAT (JSON):
{{
    "steps": [
        {{"action": "action_name", "params": {{}}, "description": "What this does"}}
    ],
    "reasoning": "Why this plan",
    "needs_clarification": false,
    "question": null,
    "options": null
}}"""
        
        if error_context:
            prompt += f"\n\nPREVIOUS ERROR (retry with different approach):\n{error_context}"
        
        return prompt
    
    def _parse_plan(self, response: str) -> ExecutionPlan:
        """Parse LLM response into ExecutionPlan."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            steps = [
                StepPlan(
                    action=s.get('action', 'unknown'),
                    params=s.get('params', {}),
                    description=s.get('description')
                )
                for s in data.get('steps', [])
            ]
            
            return ExecutionPlan(
                steps=steps,
                reasoning=data.get('reasoning'),
                needs_clarification=data.get('needs_clarification', False),
                question=data.get('question'),
                options=data.get('options')
            )
        except Exception as e:
            logger.warning(f"Failed to parse plan: {e}")
            # Return single step with original prompt as transform
            return ExecutionPlan(
                steps=[StepPlan(action='transform', params={'instruction': response})],
                reasoning="Fallback to direct transform"
            )
    
    def _parse_answer(self, response: str, question: str) -> Dict[str, Any]:
        """Parse LLM answer response."""
        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except:
            return {
                "answer": response.strip(),
                "code": None,
                "confidence": 0.7
            }
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other content."""
        if not text:
            return ""
        
        # Remove thinking tags first
        text = strip_think_tags(text)
        
        # Try to find JSON in code blocks first
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Find the first { and extract JSON using bracket counting
        start = text.find('{')
        if start == -1:
            return text
        
        # Count brackets to find matching closing brace
        depth = 0
        in_string = False
        escape = False
        
        for i, char in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
                
            if char == '\\' and in_string:
                escape = True
                continue
                
            if char == '"' and not escape:
                in_string = not in_string
                continue
                
            if in_string:
                continue
                
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        
        # If we get here, brackets didn't match - return from start to end
        return text[start:]
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from text."""
        # Try to find code in code blocks
        code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Return as-is if no code block
        return text.strip()
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None
    ) -> str:
        """Call LLM with fallback across providers."""
        max_tokens = max_tokens or LLM_MAX_TOKENS
        
        for provider in self.providers:
            try:
                return await self._call_provider(provider, messages, max_tokens)
            except Exception as e:
                logger.warning(f"Provider {provider['name']} failed: {e}")
                continue
        
        raise RuntimeError("All LLM providers failed")
    
    async def _call_provider(
        self,
        provider: Dict,
        messages: List[Dict[str, str]],
        max_tokens: int
    ) -> str:
        """Call a specific LLM provider."""
        name = provider['name']
        api_key = provider['api_key']
        model = provider['model']
        base_url = provider['base_url']
        
        if name == 'google':
            return await self._call_google(api_key, model, messages, max_tokens)
        elif name == 'anthropic':
            return await self._call_anthropic(api_key, model, messages, max_tokens)
        else:
            return await self._call_openai_compatible(api_key, model, base_url, messages, max_tokens)
    
    async def _call_openai_compatible(
        self,
        api_key: str,
        model: str,
        base_url: str,
        messages: List[Dict[str, str]],
        max_tokens: int
    ) -> str:
        """Call OpenAI-compatible API (OpenAI, Groq, Cerebras, etc.)."""
        import httpx
        
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": LLM_TEMPERATURE
                }
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
    
    async def _call_google(
        self,
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int
    ) -> str:
        """Call Google Gemini API."""
        import httpx
        
        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
            else:
                role = 'user' if msg['role'] == 'user' else 'model'
                contents.append({
                    "role": role,
                    "parts": [{"text": msg['content']}]
                })
        
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": api_key},
                json={
                    "contents": contents,
                    "systemInstruction": {"parts": [{"text": system_instruction}]} if system_instruction else None,
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": LLM_TEMPERATURE
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
    
    async def _call_anthropic(
        self,
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int
    ) -> str:
        """Call Anthropic Claude API."""
        import httpx
        
        # Extract system message
        system = None
        filtered_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system = msg['content']
            else:
                filtered_messages.append(msg)
        
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system or "You are a helpful assistant.",
                    "messages": filtered_messages
                }
            )
            response.raise_for_status()
            data = response.json()
            return data['content'][0]['text']
    
    # ========================================================================
    # EXCEL PREPROCESSING (LLM-DRIVEN)
    # ========================================================================
    
    async def analyze_excel_structure(
        self,
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze Excel file structure and determine preprocessing strategy.
        """
        structure_json = json.dumps(structure, indent=2, default=str)[:8000]  # Limit size
        
        prompt = f"""Analyze this Excel file structure and determine how to extract clean tabular data.

FILE STRUCTURE:
{structure_json}

LEGEND:
- [B] = Bold text (often headers)
- [BG] = Has background color (often headers or titles)
- Empty strings = empty cells

TASK: Analyze the structure and determine:
1. Which sheet contains the main data table?
2. What row number contains the column headers?
3. What row number does the actual data start?
4. Are there metadata/title rows to skip at the top?
5. Are there any columns that should be skipped (empty, index numbers)?
6. How should merged cells be handled?
7. Does the sheet contain multiple separate tables?

OUTPUT JSON:
{{
    "target_sheet": "name of sheet with main data",
    "header_row": 1,
    "data_start_row": 2,
    "skip_top_rows": 0,
    "skip_columns": [],
    "merged_cell_strategy": "fill_value",
    "has_multiple_tables": false,
    "detected_issues": ["list of any data quality issues noticed"],
    "preprocessing_notes": "Brief description of what preprocessing is needed"
}}

Return ONLY the JSON, no explanation."""

        response = await self._call_llm([
            {"role": "user", "content": prompt}
        ])
        
        try:
            return json.loads(self._extract_json(response))
        except Exception as e:
            logger.warning(f"Failed to parse Excel analysis: {e}")
            return {
                "target_sheet": structure.get("active_sheet", "Sheet1"),
                "header_row": 1,
                "data_start_row": 2,
                "skip_top_rows": 0,
                "skip_columns": [],
                "merged_cell_strategy": "fill_value",
                "has_multiple_tables": False,
                "preprocessing_notes": "Using defaults due to analysis failure"
            }
    
    async def generate_preprocessing_plan(
        self,
        analysis: Dict[str, Any],
        error_context: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a preprocessing plan based on structured analysis.
        
        If error_context is provided (from previous failed attempts), 
        includes it in the prompt so LLM can learn from mistakes.
        """
        from .excel_tools import get_tools_prompt
        
        tools_doc = get_tools_prompt()
        
        # Build focused prompt from analysis
        main_sheet = analysis.get('main_sheet', {})
        structure = analysis.get('structure', {})
        samples = analysis.get('samples', {})
        columns = analysis.get('columns', [])
        hints = analysis.get('preprocessing_hints', [])
        
        # Format samples for context
        top_rows_str = "\n".join([" | ".join(row[:10]) for row in samples.get('top_rows', [])[:8]])
        first_data_str = "\n".join([" | ".join(row[:10]) for row in samples.get('first_data', [])[:3]])
        last_rows_str = "\n".join([" | ".join(row[:10]) for row in samples.get('last_rows', [])[-3:]])
        
        # Format columns
        columns_str = ", ".join([f"{c.get('header', 'Col')} ({c.get('inferred_type', '?')})" for c in columns[:10]])
        
        # Build error context section if retrying
        error_section = ""
        if error_context:
            error_lines = ["PREVIOUS ATTEMPTS FAILED - Learn from these errors:"]
            for err in error_context:
                error_lines.append(f"  Attempt {err.get('attempt')}: {err.get('error')}")
                if err.get('steps_tried'):
                    error_lines.append(f"    Steps tried: {', '.join(err.get('steps_tried', []))}")
            error_section = "\n".join(error_lines) + "\n\nYou MUST try a DIFFERENT approach this time.\n"
        
        prompt = f"""Generate a preprocessing plan for this Excel file based on the analysis.

{error_section}FILE INFO:
- Sheet: {main_sheet.get('name', 'Sheet1')}
- Size: {main_sheet.get('total_rows', 0)} rows × {main_sheet.get('total_cols', 0)} columns
- Has merged cells: {structure.get('has_merged', False)}
- Detected header row: {structure.get('detected_header_row', 1)}
- Header confidence: {structure.get('header_confidence', 0.5):.0%}

TOP ROWS (title/headers area):
{top_rows_str}

FIRST DATA ROWS:
{first_data_str}

LAST ROWS (check for totals):
{last_rows_str}

COLUMNS DETECTED:
{columns_str}

PREPROCESSING HINTS (from analysis):
{chr(10).join(['- ' + h for h in hints]) if hints else '- No issues detected'}

{tools_doc}

TASK: Create a preprocessing plan to convert this file into a clean DataFrame.

OUTPUT JSON:
{{
    "sheet": "{main_sheet.get('name')}" or null,
    "steps": [
        {{"function": "function_name", "params": {{}}}},
        ...
    ],
    "reasoning": "Brief explanation"
}}

IMPORTANT RULES:
1. If hints mention MERGED_CELLS, call unmerge_and_fill FIRST
2. If hints mention TITLE_ROWS or header is not row 1, use set_header_row
3. If hints mention TOTALS_ROW, call remove_totals_row
4. Always end with cleanup: remove_empty_rows, strip_whitespace
5. Return ONLY valid JSON

JSON:"""

        response = await self._call_llm([
            {"role": "user", "content": prompt}
        ])
        
        try:
            extracted = self._extract_json(response)
            
            result = json.loads(extracted)
            
            # Check if LLM returned empty plan - use fallback
            if not result.get('steps'):
                logger.info("LLM returned empty plan, using intelligent fallback")
                result = self._build_fallback_plan(structure, main_sheet)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to parse preprocessing plan: {e}")
            return self._build_fallback_plan(structure, main_sheet)
    
    def _build_fallback_plan(self, structure: Dict, main_sheet: Dict) -> Dict:
        """Build intelligent fallback plan from detected structure."""
        steps = []
        
        if structure.get('has_merged'):
            steps.append({"function": "unmerge_and_fill", "params": {}})
        
        header_row = structure.get('detected_header_row', 1)
        if header_row > 1:
            steps.append({"function": "set_header_row", "params": {"row_number": header_row}})
        
        steps.append({"function": "remove_empty_rows", "params": {}})
        steps.append({"function": "strip_whitespace", "params": {}})
        
        return {
            "sheet": main_sheet.get('name'),
            "steps": steps,
            "reasoning": "Fallback plan based on detected structure"
        }
    
    async def adjust_step(
        self,
        action: str,
        original_params: Dict[str, Any],
        error: str,
        error_history: str,
        df_context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Ask LLM to adjust a step's parameters based on failure.
        
        Returns adjusted action and params, or None if no adjustment possible.
        """
        prompt = f"""A spreadsheet operation step failed. Adjust the parameters to fix it.

ACTION: {action}
ORIGINAL PARAMS: {json.dumps(original_params, default=str)}

ERROR: {error}

PREVIOUS ATTEMPTS:
{error_history}

{df_context}

TASK: Suggest adjusted parameters that might fix the error.
- If column name is wrong, suggest the correct one
- If value type is wrong, suggest conversion
- If operation is impossible, return action: null

OUTPUT JSON:
{{
    "action": "{action}" (or different action, or null if impossible),
    "params": {{ adjusted parameters }},
    "reasoning": "Why this adjustment should work"
}}

Return ONLY JSON:"""

        try:
            response = await self._call_llm([
                {"role": "user", "content": prompt}
            ])
            
            result = json.loads(self._extract_json(response))
            
            if result.get('action') is None:
                return None
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to adjust step: {e}")
            return None


def strip_think_tags(text: str) -> str:
    """Remove thinking/reasoning tags from LLM output."""
    if not isinstance(text, str):
        return text
    
    # Standard: <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Minimax: <|thinking|>...</|thinking|>
    text = re.sub(r'<\|thinking\|>.*?</\|thinking\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|thinking\|>.*$', '', text, flags=re.DOTALL)
    
    # DeepSeek: <thought>...</thought>
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Reasoning: <reasoning>...</reasoning>
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()


# ========================================================================
# GLOBAL INSTANCE
# ========================================================================

llm_client = LLMClient()

