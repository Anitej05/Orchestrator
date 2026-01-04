"""
LLM-powered query agent for natural language spreadsheet operations.

This module provides the SpreadsheetQueryAgent class that converts natural language
queries into pandas operations using a ReAct-style reasoning loop.
"""

import logging
import json
import time
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

from .config import (
    GROQ_API_KEY,
    CEREBRAS_API_KEY,
    NVIDIA_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GROQ_MODEL,
    CEREBRAS_MODEL,
    NVIDIA_MODEL,
    GOOGLE_MODEL,
    OPENAI_MODEL,
    ANTHROPIC_MODEL,
    GROQ_BASE_URL,
    CEREBRAS_BASE_URL,
    NVIDIA_BASE_URL,
    GOOGLE_BASE_URL,
    OPENAI_BASE_URL,
    ANTHROPIC_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS_QUERY
)
from .models import QueryResult
from .memory import spreadsheet_memory

logger = logging.getLogger(__name__)


class SpreadsheetQueryAgent:
    """
    LLM-powered sub-agent that converts natural language queries to pandas operations.
    Uses a ReAct-style reasoning loop to iteratively query and analyze data.
    """
    
    def __init__(self):
        self.providers = []
        self._init_providers()
        self._start_time = time.time()
        self.metrics = {
            "queries": {
                "total": 0,
                "successful": 0,
                "failed": 0
            },
            "llm_calls": {
                "total": 0,
                "groq": 0,
                "cerebras": 0,
                "nvidia": 0,
                "google": 0,
                "openai": 0,
                "anthropic": 0,
                "retries": 0,
                "failures": 0
            },
            "tokens": {
                "input_total": 0,
                "output_total": 0
            },
            "performance": {
                "total_latency_ms": 0,
                "avg_latency_ms": 0,
                "llm_latency_ms": 0,
                "execution_latency_ms": 0,
                "queries_completed": 0
            },
            "cache": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0
            },
            "retry": {
                "total_retries": 0,
                "successful_retries": 0,
                "retry_success_rate": 0.0
            },
            "resource": {
                "peak_memory_mb": 0,
                "avg_cpu_percent": 0,
                "current_memory_mb": 0
            }
        }
    
    def _init_providers(self):
        """Initialize LLM providers with fallback chain"""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed. LLM features disabled.")
            return
        
        # Build provider chain: Cerebras → Groq → NVIDIA → Google → OpenAI → Anthropic
        if CEREBRAS_API_KEY:
            self.providers.append({
                "name": "cerebras",
                "client": OpenAI(api_key=CEREBRAS_API_KEY, base_url=CEREBRAS_BASE_URL),
                "model": CEREBRAS_MODEL,
                "max_tokens": LLM_MAX_TOKENS_QUERY
            })

        if GROQ_API_KEY:
            self.providers.append({
                "name": "groq",
                "client": OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL),
                "model": GROQ_MODEL,
                "max_tokens": LLM_MAX_TOKENS_QUERY
            })

        if NVIDIA_API_KEY:
            self.providers.append({
                "name": "nvidia",
                "client": OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL),
                "model": NVIDIA_MODEL,
                "max_tokens": LLM_MAX_TOKENS_QUERY
            })
        
        # Continue with other providers
        if GOOGLE_API_KEY:
            # Google uses OpenAI-compatible API
            self.providers.append({
                "name": "google",
                "client": OpenAI(api_key=GOOGLE_API_KEY, base_url=GOOGLE_BASE_URL),
                "model": GOOGLE_MODEL,
                "max_tokens": LLM_MAX_TOKENS_QUERY
            })
        
        if OPENAI_API_KEY:
            self.providers.append({
                "name": "openai",
                "client": OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
                "model": OPENAI_MODEL,
                "max_tokens": LLM_MAX_TOKENS_QUERY
            })
        
        if ANTHROPIC_API_KEY:
            # Anthropic uses different API, add placeholder for now
            # Would need anthropic SDK for full implementation
            self.providers.append({
                "name": "anthropic",
                "client": OpenAI(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL),
                "model": ANTHROPIC_MODEL,
                "max_tokens": LLM_MAX_TOKENS_QUERY
            })
        
        if self.providers:
            provider_names = ' → '.join([p['name'] for p in self.providers])
            logger.info(f"🔧 Query Agent initialized with providers: {provider_names}")
        else:
            logger.warning("⚠️ No LLM providers configured. Agent will not be functional.")
    
    def _get_completion(self, messages: List[Dict], temperature: float = 0.1, provider_offset: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Get LLM completion with fallback and metrics tracking
        
        Returns:
            Tuple of (response_text, metrics_dict)
        
        Note: Temperature is hardcoded to 0.1 (matching old working agent) for consistency.
              Do not change this unless thoroughly testing the new temperature's JSON output.
        provider_offset lets callers skip providers that produced repeated parse errors.
        """
        if not self.providers:
            raise RuntimeError("No LLM providers available")
        
        # HARDCODED: Always use 0.1 to match old agent's tested behavior
        
        llm_start = time.time()
        call_metrics = {
            "provider_used": None,
            "retries": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "latency_ms": 0
        }
        
        last_error = None
        for idx, provider in enumerate(self.providers[provider_offset:], start=provider_offset):
            try:
                logger.info(f"🤖 Using {provider['name'].upper()} for query analysis")
                self.metrics["llm_calls"]["total"] += 1
                self.metrics["llm_calls"][provider['name']] += 1
                
                if idx > 0:
                    self.metrics["llm_calls"]["retries"] += 1
                    self.metrics["retry"]["total_retries"] += 1
                    call_metrics["retries"] = idx
                
                response = provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=provider["max_tokens"]
                )
                
                # Track token usage
                if hasattr(response, 'usage'):
                    call_metrics["tokens_input"] = response.usage.prompt_tokens
                    call_metrics["tokens_output"] = response.usage.completion_tokens
                    self.metrics["tokens"]["input_total"] += response.usage.prompt_tokens
                    self.metrics["tokens"]["output_total"] += response.usage.completion_tokens
                    
                    logger.info(f"📊 Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")
                
                call_metrics["provider_used"] = provider['name']
                call_metrics["latency_ms"] = (time.time() - llm_start) * 1000
                self.metrics["performance"]["llm_latency_ms"] += call_metrics["latency_ms"]
                
                if idx > 0:
                    self.metrics["retry"]["successful_retries"] += 1
                
                return response.choices[0].message.content.strip(), call_metrics
                
            except Exception as e:
                logger.warning(f"⚠️ {provider['name'].upper()} failed: {str(e)[:100]}")
                last_error = e
                if idx == len(self.providers) - 1:
                    self.metrics["llm_calls"]["failures"] += 1
                continue
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    def _get_dataframe_context(self, df: pd.DataFrame, file_id: str = None) -> str:
        """Generate context about the dataframe for the LLM (with caching)
        
        Note: The LLM should be reminded that pandas (as pd) is already imported
        in the execution environment and no import statements should be generated.
        """
        # Try to get from cache if file_id provided
        if file_id:
            cached_context = spreadsheet_memory.get_df_metadata(file_id)
            if cached_context and 'context_string' in cached_context:
                return cached_context['context_string']
        
        context = f"""DataFrame Information:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Data Types:
{df.dtypes.to_string()}

Sample Data (first 5 rows):
{df.head().to_string()}

Column Statistics:
"""
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                context += f"\n{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}"
            else:
                unique_count = df[col].nunique()
                context += f"\n{col}: {unique_count} unique values"
                if unique_count <= 10:
                    context += f", values: {df[col].unique().tolist()}"
        
        # Cache the context if file_id provided
        if file_id:
            spreadsheet_memory.cache_df_metadata(file_id, {
                'context_string': context,
                'shape': df.shape,
                'columns': list(df.columns)
            })
        
        return context
    
    def _build_system_prompt(self, df_context: str, session_context: str = "", all_files_context: str = "") -> str:
        """Build the system prompt for the query agent
        
        NOTE: Simplified to match old working agent's prompt structure.
        Removed 'ALL FILES' section which confused Llama 3.1 models.
        """
        return f"""You are a powerful data analysis assistant that helps users query and analyze spreadsheet data using pandas.

=== CURRENT DATAFRAME STATE ===
{df_context}

=== SESSION HISTORY (Previous Operations) ===
{session_context}

=== YOUR CAPABILITIES ===
You can perform ANY pandas operation including:
- FILTERING: Select rows matching conditions (df.query(), boolean indexing)
- AGGREGATION: Calculate statistics (sum, mean, count, max, min, std)
- GROUPING: Group by columns and aggregate (df.groupby())
- SORTING: Sort by one or more columns (df.sort_values())
- ANALYSIS: Find patterns, outliers, trends

=== IMPORTANT RULES ===
1. ALWAYS respond with valid JSON in the exact format specified below
2. Use the DataFrame variable 'df' for all operations
3. Handle column names with spaces using backticks in query() OR bracket notation df['col name']
4. If column names look combined (e.g., "YearsExperience,Salary"), split them into separate columns and drop empty/unnamed columns before analysis
5. If the question is unclear, make reasonable assumptions and explain them
6. For multi-step analysis, break it down clearly with needs_more_steps=true
7. ALWAYS provide a helpful, human-readable final_answer

=== RESPONSE FORMAT (JSON) ===
{{
    "thinking": "Your step-by-step reasoning about what the user wants",
    "needs_more_steps": true/false,
    "pandas_code": "df.query('column > value')",
    "explanation": "Clear explanation of what this code does",
    "is_final_answer": true/false,
    "final_answer": "Human-readable answer (REQUIRED if is_final_answer is true)"
}}

=== EXAMPLES ===

User: "Show me all people older than 30"
Response: {{"thinking": "User wants to filter rows where age > 30", "needs_more_steps": false, "pandas_code": "df.query('age > 30')", "explanation": "Filtering to show only rows where age column exceeds 30", "is_final_answer": true, "final_answer": "Here are all people older than 30"}}

User: "What's the average salary by department?"
Response: {{"thinking": "User wants a grouped aggregation - group by department and calculate mean salary", "needs_more_steps": false, "pandas_code": "df.groupby('department')['salary'].mean()", "explanation": "Grouping by department and calculating the average salary for each", "is_final_answer": true, "final_answer": "Here is the average salary broken down by department"}}

User: "Who earns the most in the sales department?"
Response: {{"thinking": "Need to filter to Sales department first, then find the row with maximum salary", "needs_more_steps": false, "pandas_code": "df[df['department'] == 'Sales'].nlargest(1, 'salary')", "explanation": "Filter to Sales department only, then get the row with the highest salary", "is_final_answer": true, "final_answer": "The highest earner in the Sales department is shown above"}}

User: "How many rows have Feature1 greater than 50?"
Response: {{"thinking": "Count rows where Feature1 > 50", "needs_more_steps": false, "pandas_code": "len(df[df['Feature1'] > 50])", "explanation": "Counting rows where Feature1 exceeds 50", "is_final_answer": true, "final_answer": "There are X rows where Feature1 is greater than 50"}}
"""
    
    def _safe_execute_pandas(self, df: pd.DataFrame, code: str) -> Tuple[Any, pd.DataFrame, Optional[str]]:
        """Safely execute pandas code and return result or error"""
        try:
            # Create a safe execution environment
            local_vars = {"df": df, "pd": pd}
            
            # Execute the code
            exec(code, {"__builtins__": {}}, local_vars)
            
            # Retrieve the (possibly modified) dataframe
            updated_df = local_vars.get("df", df)
            
            # If the specific variable 'result' exists, return it
            if "result" in local_vars:
                return local_vars["result"], updated_df, None
                
            # Otherwise, return the df itself as the result (e.g. after filtering)
            return updated_df, updated_df, None
            
        except Exception as e:
            return None, df, str(e)
    
    async def query(
        self, 
        df: pd.DataFrame, 
        question: str, 
        max_iterations: int = 5, 
        session_context: str = "", 
        all_files_context: str = "",
        file_id: str = None,
        thread_id: str = "default"
    ) -> QueryResult:
        """
        Process a natural language query against the dataframe.
        Uses a ReAct-style loop to iteratively reason and execute.
        
        Args:
            df: DataFrame to query
            question: Natural language question
            max_iterations: Maximum reasoning iterations
            session_context: Previous operations context
            all_files_context: Context about all files in conversation
            file_id: File ID for caching
            thread_id: Thread ID for query isolation
        
        Returns:
            QueryResult with answer, steps, and final data
        """
        # Track query start time and resources
        query_start = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.metrics["queries"]["total"] += 1
        
        query_metrics = {
            "llm_calls": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "retries": 0,
            "cache_hit": False,
            "iterations": 0,
            "latency_ms": 0
        }
        
        if not self.providers:
            self.metrics["queries"]["failed"] += 1
            return QueryResult(
                question=question,
                answer="LLM providers not available. Please check API keys.",
                steps_taken=[],
                success=False,
                error="No LLM providers configured"
            )
        
        # TEMPORARILY DISABLED CACHING: Query caching disabled until JSON parsing is stable
        # Previous issues with cache returning malformed queries. Re-enable after validation.
        # cached_result = spreadsheet_memory.get_cached_query(question, file_id, thread_id)
        # if cached_result:
        #     logger.info(f"✨ Using cached result for query: {question[:50]}...")
        #     self.metrics["cache"]["hits"] += 1
        #     query_metrics["cache_hit"] = True
        #     query_metrics["latency_ms"] = (time.time() - query_start) * 1000
        #     cached_result.execution_metrics = query_metrics
        #     self._log_execution_metrics(query_metrics, True)
        #     return cached_result
        
        self.metrics["cache"]["misses"] += 1
        
        # Working copy of dataframe for this session
        current_df = df.copy()
        
        df_context = self._get_dataframe_context(current_df, file_id)
        system_prompt = self._build_system_prompt(df_context, session_context, all_files_context)
        
        steps_taken = []
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        current_result = None
        iteration = 0
        parse_failures = 0
        provider_offset = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"📊 Query iteration {iteration}/{max_iterations}")
            
            try:
                # Get LLM response with metrics
                response_text, call_metrics = self._get_completion(conversation, provider_offset=provider_offset)
                query_metrics["llm_calls"] += 1
                query_metrics["tokens_input"] += call_metrics["tokens_input"]
                query_metrics["tokens_output"] += call_metrics["tokens_output"]
                query_metrics["retries"] += call_metrics["retries"]
                query_metrics["iterations"] = iteration
                
                # Parse JSON response
                try:
                    # Handle potential markdown code blocks
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]
                    
                    # HARDENED: Normalize whitespace before parsing
                    response_text = response_text.strip()
                    response_text = "\n".join([line.strip() for line in response_text.split("\n")])

                    def _extract_json_object(text: str) -> str:
                        start = text.find("{")
                        end = text.rfind("}")
                        return text[start:end + 1] if start != -1 and end != -1 and end > start else text

                    try:
                        response = json.loads(response_text)
                    except Exception:
                        fallback = _extract_json_object(response_text)
                        response = json.loads(fallback)

                    # VALIDATION: Ensure required fields exist in parsed JSON
                    required_fields = ["thinking", "pandas_code", "is_final_answer"]
                    missing_fields = [f for f in required_fields if f not in response]
                    if missing_fields:
                        raise ValueError(f"JSON missing required fields: {missing_fields}")

                    parse_failures = 0

                except (json.JSONDecodeError, ValueError) as e:
                    parse_failures += 1
                    logger.warning(f"Failed to parse LLM response as JSON (attempt {parse_failures}): {e}")
                    steps_taken.append({
                        "iteration": iteration,
                        "error": f"JSON parse error: {e}",
                        "raw_response": response_text[:500]
                    })
                    # Ask LLM to fix the response with stricter instruction
                    conversation.append({"role": "assistant", "content": response_text[:500]})
                    conversation.append({
                        "role": "user", 
                        "content": "Please respond with valid JSON only (no markdown), using keys thinking, pandas_code, is_final_answer, final_answer."
                    })
                    if parse_failures >= 2 and provider_offset < len(self.providers) - 1:
                        failures_before_switch = parse_failures
                        provider_offset += 1
                        parse_failures = 0
                        logger.info(f"Switching to next LLM provider after {failures_before_switch} consecutive parse failures")
                    continue
                
                # Extract fields
                thinking = response.get("thinking", "")
                pandas_code = response.get("pandas_code", "")
                explanation = response.get("explanation", "")
                is_final = response.get("is_final_answer", False)
                final_answer = response.get("final_answer", "")
                
                step_info = {
                    "iteration": iteration,
                    "thinking": thinking,
                    "code": pandas_code,
                    "explanation": explanation
                }
                
                # Execute pandas code if provided
                if pandas_code:
                    result, updated_df, error = self._safe_execute_pandas(current_df, pandas_code)
                    
                    if error:
                        step_info["error"] = error
                        step_info["success"] = False
                        steps_taken.append(step_info)
                        
                        # Feed error back to LLM
                        conversation.append({"role": "assistant", "content": response_text})
                        conversation.append({
                            "role": "user", 
                            "content": f"The code raised an error: {error}\nPlease fix the code and try again."
                        })
                        continue
                        
                    # Update our working dataframe
                    current_df = updated_df
                    
                    # Convert result to serializable format
                    if isinstance(result, pd.DataFrame):
                        # Convert to records and ensure native types
                        records = result.head(10).to_dict(orient="records")
                        # Sanitize records
                        safe_records = json.loads(json.dumps(records, default=str))
                        step_info["result_preview"] = safe_records
                        step_info["result_shape"] = result.shape
                        current_result = result.to_dict(orient="records")
                    elif isinstance(result, pd.Series):
                        # Convert to dict and ensure native types
                        series_dict = result.head(10).to_dict()
                        # Sanitize dict
                        safe_dict = json.loads(json.dumps(series_dict, default=str))
                        step_info["result_preview"] = safe_dict
                        current_result = result.to_dict()
                    else:
                        step_info["result_preview"] = str(result)
                        current_result = result
                    
                    step_info["success"] = True
                
                steps_taken.append(step_info)
                
                # Check if we have a final answer
                if is_final:
                    # Format the final answer with data
                    if current_result:
                        if isinstance(current_result, list) and len(current_result) > 0:
                            final_data = current_result[:50]  # Limit to 50 rows
                        elif isinstance(current_result, dict):
                            final_data = [current_result]
                        else:
                            final_data = None
                    else:
                        final_data = None
                    
                    # Create query result first
                    query_result = QueryResult(
                        question=question,
                        answer=final_answer,
                        steps_taken=steps_taken,
                        final_data=final_data,
                        success=True,
                        final_dataframe=current_df  # Return the modified dataframe
                    )
                    
                    # Calculate final metrics
                    query_metrics["latency_ms"] = (time.time() - query_start) * 1000
                    end_memory = process.memory_info().rss / 1024 / 1024
                    query_metrics["memory_used_mb"] = end_memory - start_memory
                    
                    # Update session metrics
                    self.metrics["queries"]["successful"] += 1
                    self.metrics["performance"]["total_latency_ms"] += query_metrics["latency_ms"]
                    self.metrics["performance"]["queries_completed"] += 1
                    
                    if self.metrics["performance"]["queries_completed"] > 0:
                        self.metrics["performance"]["avg_latency_ms"] = round(
                            self.metrics["performance"]["total_latency_ms"] / 
                            self.metrics["performance"]["queries_completed"], 2
                        )
                    
                    # Update resource metrics
                    self.metrics["resource"]["peak_memory_mb"] = max(
                        self.metrics["resource"]["peak_memory_mb"], end_memory
                    )
                    self.metrics["resource"]["current_memory_mb"] = end_memory
                    
                    # Calculate rates
                    total_cache = self.metrics["cache"]["hits"] + self.metrics["cache"]["misses"]
                    if total_cache > 0:
                        self.metrics["cache"]["hit_rate"] = round(
                            self.metrics["cache"]["hits"] / total_cache * 100, 1
                        )
                    
                    if self.metrics["retry"]["total_retries"] > 0:
                        self.metrics["retry"]["retry_success_rate"] = round(
                            self.metrics["retry"]["successful_retries"] / 
                            self.metrics["retry"]["total_retries"] * 100, 1
                        )
                    
                    # Add metrics to result
                    query_result.execution_metrics = query_metrics
                    
                    # Log metrics
                    self._log_execution_metrics(query_metrics, True)
                    
                    # Cache the result
                    spreadsheet_memory.cache_query_result(question, query_result, file_id, thread_id)
                    
                    return query_result
                
                # Continue conversation for multi-step queries
                conversation.append({"role": "assistant", "content": response_text})
                
                # Provide result context for next iteration
                if current_result:
                    result_summary = f"Previous step result: {str(current_result)[:500]}"
                    conversation.append({"role": "user", "content": f"{result_summary}\n\nContinue analysis or provide final answer."})
                
            except Exception as e:
                logger.error(f"Query iteration failed: {e}", exc_info=True)
                steps_taken.append({
                    "iteration": iteration,
                    "error": str(e),
                    "success": False
                })
                break
        
        # Max iterations reached
        self.metrics["queries"]["failed"] += 1
        return QueryResult(
            question=question,
            answer="Could not complete the query within the maximum iterations.",
            steps_taken=steps_taken,
            final_data=current_result if isinstance(current_result, list) else None,
            success=False,
            error="Max iterations reached"
        )
    
    def _log_execution_metrics(self, exec_metrics: Dict[str, Any], success: bool):
        """Log detailed execution metrics with visual formatting."""
        status_emoji = "✅" if success else "❌"
        
        logger.info("=" * 80)
        logger.info(f"{status_emoji} SPREADSHEET AGENT EXECUTION METRICS")
        logger.info("=" * 80)
        logger.info(f"📊 Performance:")
        logger.info(f"  ⏱️  Total Latency:        {exec_metrics.get('latency_ms', 0):.2f} ms")
        logger.info(f"  🔄 Iterations:           {exec_metrics.get('iterations', 0)}")
        logger.info(f"  💾 Cache Hit:            {'Yes' if exec_metrics.get('cache_hit') else 'No'}")
        logger.info(f"")
        logger.info(f"📈 LLM Statistics:")
        logger.info(f"  🤖 API Calls:            {exec_metrics.get('llm_calls', 0)}")
        logger.info(f"  🔄 Retries:              {exec_metrics.get('retries', 0)}")
        logger.info(f"  📝 Tokens Input:         {exec_metrics.get('tokens_input', 0)}")
        logger.info(f"  📤 Tokens Output:        {exec_metrics.get('tokens_output', 0)}")
        logger.info(f"  💰 Total Tokens:         {exec_metrics.get('tokens_input', 0) + exec_metrics.get('tokens_output', 0)}")
        logger.info(f"")
        
        # Session totals
        logger.info(f"🎯 Session Totals:")
        logger.info(f"  📝 Total Queries:        {self.metrics['queries']['total']}")
        logger.info(f"  ✅ Successful:           {self.metrics['queries']['successful']}")
        logger.info(f"  ❌ Failed:               {self.metrics['queries']['failed']}")
        logger.info(f"  ⏱️  Avg Latency:          {self.metrics['performance']['avg_latency_ms']:.2f} ms")
        logger.info(f"  💾 Cache Hit Rate:       {self.metrics['cache']['hit_rate']:.1f}%")
        logger.info(f"  🔄 Retry Success:        {self.metrics['retry']['retry_success_rate']:.1f}%")
        logger.info(f"  🧠 Memory Used:          {exec_metrics.get('memory_used_mb', 0):.1f} MB")
        logger.info(f"  📊 Peak Memory:          {self.metrics['resource']['peak_memory_mb']:.1f} MB")
        logger.info("=" * 80)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with computed values."""
        uptime_seconds = time.time() - self._start_time
        
        # Update CPU usage
        try:
            process = psutil.Process(os.getpid())
            self.metrics["resource"]["avg_cpu_percent"] = process.cpu_percent(interval=0.1)
        except:
            pass
        
        return {
            **self.metrics,
            "uptime_seconds": round(uptime_seconds, 2),
            "success_rate": round(
                self.metrics["queries"]["successful"] / self.metrics["queries"]["total"] * 100, 1
            ) if self.metrics["queries"]["total"] > 0 else 0
        }


# Global query agent instance
query_agent = SpreadsheetQueryAgent()
