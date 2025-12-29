"""
LLM-powered query agent for natural language spreadsheet operations.

This module provides the SpreadsheetQueryAgent class that converts natural language
queries into pandas operations using a ReAct-style reasoning loop.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

from .config import (
    CEREBRAS_API_KEY,
    GROQ_API_KEY,
    CEREBRAS_MODEL,
    GROQ_MODEL,
    CEREBRAS_BASE_URL,
    GROQ_BASE_URL,
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
    
    def _init_providers(self):
        """Initialize LLM providers with fallback chain"""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed. LLM features disabled.")
            return
        
        # Build provider chain: Cerebras → Groq
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
        
        if self.providers:
            logger.info(f"🔧 Query Agent initialized with providers: {' → '.join([p['name'] for p in self.providers])}")
        else:
            logger.warning("⚠️ No LLM providers available for Query Agent")
    
    def _get_completion(self, messages: List[Dict], temperature: float = None) -> str:
        """Get LLM completion with fallback"""
        if not self.providers:
            raise RuntimeError("No LLM providers available")
        
        if temperature is None:
            temperature = LLM_TEMPERATURE
        
        last_error = None
        for provider in self.providers:
            try:
                logger.info(f"🤖 Using {provider['name'].upper()} for query analysis")
                response = provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=provider["max_tokens"]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"⚠️ {provider['name'].upper()} failed: {str(e)[:100]}")
                last_error = e
                continue
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    def _get_dataframe_context(self, df: pd.DataFrame, file_id: str = None) -> str:
        """Generate context about the dataframe for the LLM (with caching)"""
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
        """Build the system prompt for the query agent"""
        return f"""You are a powerful data analysis assistant that helps users query and analyze spreadsheet data using pandas.

=== ALL FILES IN THIS CONVERSATION ===
{all_files_context if all_files_context else "Currently working with one file."}

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
4. If the question is unclear, make reasonable assumptions and explain them
5. For multi-step analysis, break it down clearly with needs_more_steps=true
6. ALWAYS provide a helpful, human-readable final_answer

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
        if not self.providers:
            return QueryResult(
                question=question,
                answer="LLM providers not available. Please check API keys.",
                steps_taken=[],
                success=False,
                error="No LLM providers configured"
            )
        
        # Check cache for similar query
        cached_result = spreadsheet_memory.get_cached_query(question, file_id, thread_id)
        if cached_result:
            logger.info(f"✨ Using cached result for query: {question[:50]}...")
            return cached_result
        
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
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"📊 Query iteration {iteration}/{max_iterations}")
            
            try:
                # Get LLM response
                response_text = self._get_completion(conversation)
                
                # Parse JSON response
                try:
                    # Handle potential markdown code blocks
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]
                    
                    response = json.loads(response_text.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response as JSON: {e}")
                    steps_taken.append({
                        "iteration": iteration,
                        "error": f"JSON parse error: {e}",
                        "raw_response": response_text[:500]
                    })
                    # Ask LLM to fix the response
                    conversation.append({"role": "assistant", "content": response_text})
                    conversation.append({"role": "user", "content": "Please respond with valid JSON only, no markdown or extra text."})
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
                    
                    query_result = QueryResult(
                        question=question,
                        answer=final_answer,
                        steps_taken=steps_taken,
                        final_data=final_data,
                        success=True,
                        final_dataframe=current_df  # Return the modified dataframe
                    )
                    
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
        return QueryResult(
            question=question,
            answer="Could not complete the query within the maximum iterations.",
            steps_taken=steps_taken,
            final_data=current_result if isinstance(current_result, list) else None,
            success=False,
            error="Max iterations reached"
        )


# Global query agent instance
query_agent = SpreadsheetQueryAgent()
