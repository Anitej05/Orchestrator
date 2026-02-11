# agents/spreadsheet_agent/llm.py

import json
import logging
import re
from typing import Dict, List, Optional, Any

# Import Centralized Service
from backend.services.inference_service import inference_service, InferencePriority
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .agent_schemas import ExecutionPlan, StepPlan

logger = logging.getLogger("spreadsheet_agent.llm")


def strip_think_tags(text: str) -> str:
    """Helper to strip think tags (can also rely on inference_service)."""
    if not text: return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


class LLMClient:
    """
    Adapter Client: Delegates all valid LLM calls to the Unified InferenceService.
    Preserves existing method signatures for compatibility with SpreadsheetAgent.
    """
    
    def __init__(self):
        logger.info("LLMClient initialized (Delegating to Unified InferenceService)")
    
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
        """
        system_prompt = self._build_decomposition_prompt(context, error_context)
        
        # Use simple Generation + Parsing (Plan decomposition is complex, strict JSON schema sometimes fails)
        response = await inference_service.generate(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ],
            priority=InferencePriority.SPEED,
            temperature=0.2,
            json_mode=True
        )
        
        plan = self._parse_plan(response)
        
        logger.info(f"[LLM DECOMPOSE] Plan: {len(plan.steps)} steps")
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
        
        response = await inference_service.generate(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=instruction)
            ],
            priority=InferencePriority.SPEED,
            temperature=0.1
        )
        
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
}}"""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add history if provided
        if history:
            for h in history[-3:]:  # Last 3 exchanges
                messages.append(HumanMessage(content=h.get('question', '')))
                messages.append(AIMessage(content=h.get('answer', '')))
        
        messages.append(HumanMessage(content=question))
        
        response = await inference_service.generate(
            messages=messages,
            priority=InferencePriority.SPEED,
            temperature=0.2,
            json_mode=True
        )
        
        return self._parse_answer(response, question)
    
    
    async def analyze_query_intent(
        self,
        query: str,
        schema_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM-powered query intent analysis.
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
    "search_terms": ["term1", "term2"],
    "target_columns": ["col1", "col2"],
    "needs_filtering": true/false,
    "row_references": [],
    "sampling_strategy": "full|distribution|stratified",
    "reasoning": "Brief explanation of your analysis"
}}"""

        response = await inference_service.generate(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {query}")
            ],
            priority=InferencePriority.SPEED,
            json_mode=True
        )
        
        # Parse JSON response
        try:
            return json.loads(self._extract_json(response))
        except Exception as e:
            logger.warning(f"Failed to parse LLM query analysis: {e}")
            return {
                "search_terms": [],
                "target_columns": [],
                "needs_filtering": False,
                "row_references": [],
                "sampling_strategy": "stratified",
                "reasoning": "Parse error, using safe defaults"
            }
    
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
}}"""

        response = await inference_service.generate(
            messages=[HumanMessage(content=prompt)],
            priority=InferencePriority.SPEED,
            json_mode=True
        )
        
        try:
            return json.loads(self._extract_json(response))
        except Exception as e:
            logger.warning(f"Failed to parse Excel analysis: {e}")
            return {
                "target_sheet": structure.get("active_sheet", "Sheet1"),
                "header_row": 1,
                "data_start_row": 2,
                "preprocessing_notes": "Using defaults due to analysis failure"
            }
            
    async def generate_preprocessing_plan(
        self,
        analysis: Dict[str, Any],
        error_context: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a preprocessing plan based on structured analysis.
        """
        from .excel_tools import get_tools_prompt
        tools_doc = get_tools_prompt()
        
        # Build prompt logic (simplified for brevity, assume helper/structure identical)
        main_sheet = analysis.get('main_sheet', {})
        structure = analysis.get('structure', {})
        
        prompt = f"""Generate a preprocessing plan for this Excel file based on the analysis.
        
Analysis: {json.dumps(analysis, default=str)[:2000]}
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
}}"""

        response = await inference_service.generate(
            messages=[HumanMessage(content=prompt)],
            priority=InferencePriority.SPEED,
            json_mode=True
        )
        
        try:
            extracted = self._extract_json(response)
            result = json.loads(extracted)
             # Check if LLM returned empty plan - use fallback
            if not result.get('steps'):
                logger.info("LLM returned empty plan, using intelligent fallback")
                result = self._build_fallback_plan(structure, main_sheet)
            return result
        except Exception:
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
        """
        prompt = f"""A spreadsheet operation step failed. Adjust the parameters to fix it.

ACTION: {action}
ORIGINAL PARAMS: {json.dumps(original_params, default=str)}
ERROR: {error}
PREVIOUS ATTEMPTS:
{error_history}
{df_context}

TASK: Suggest adjusted parameters that might fix the error.

OUTPUT JSON:
{{
    "action": "{action}",
    "params": {{ adjusted parameters }},
    "reasoning": "Why this adjustment should work"
}}"""

        try:
            response = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                json_mode=True
            )
            return json.loads(self._extract_json(response))
        except Exception:
            return None

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
        data_preview = context.get('data_preview')
        history = context.get('history', [])
        
        data_context = f"""
**DATA IS ALREADY LOADED**
- Available columns: {columns[:20] if columns else 'None'}
"""
        if data_preview:
            data_context = f"""
**DATA IS ALREADY LOADED**
DATA PREVIEW:
{data_preview}
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
        """Extract JSON from text."""
        if not text: return ""
        text = strip_think_tags(text)
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match: return code_match.group(1)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1: return text[start:end+1]
        return text
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from text."""
         # Try to find code in code blocks
        code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text.strip()

# Initialize and Export
llm_client = LLMClient()
