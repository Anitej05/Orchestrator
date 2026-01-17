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
import numpy as np
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
from .models import QueryResult, UserChoice, AnomalyDetails
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
                
                # Prefer structured JSON output where supported (OpenAI-compatible clients).
                # Some providers reject unknown params; fall back cleanly without failing the provider.
                create_kwargs = {
                    "model": provider["model"],
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": provider["max_tokens"],
                }

                response = None
                try:
                    create_kwargs["response_format"] = {"type": "json_object"}
                    response = provider["client"].chat.completions.create(**create_kwargs)
                except Exception as e:
                    # Retry once without response_format for providers that don't support it.
                    create_kwargs.pop("response_format", None)
                    response = provider["client"].chat.completions.create(**create_kwargs)
                
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
    
    def _get_dataframe_context(self, df: pd.DataFrame, file_id: str = None, thread_id: str = "default") -> str:
        """Generate intelligent context about the dataframe for the LLM (with caching)
        
        Uses intelligent parsing to provide structured document understanding,
        metadata extraction, and context building for better LLM comprehension.
        
        Note: The LLM should be reminded that pandas (as pd) is already imported
        in the execution environment and no import statements should be generated.
        """
        # Try to get from context cache if file_id provided
        if file_id:
            cached_context = spreadsheet_memory.get_context(thread_id, file_id)
            if cached_context:
                return cached_context
        
        # Try intelligent parsing first
        try:
            from agents.spreadsheet_agent.spreadsheet_parser import spreadsheet_parser
            
            # Parse the dataframe with intelligent analysis
            parsed_spreadsheet = spreadsheet_parser.parse_dataframe(df, file_id or "temp", "Sheet1")
            
            # Build structured context
            structured_context = spreadsheet_parser.build_context(parsed_spreadsheet, max_tokens=6000)
            
            # Convert structured context to LLM-friendly format
            context = self._format_intelligent_context(structured_context, df)
            
            logger.info(f"✅ Using intelligent context for {file_id} (confidence: {parsed_spreadsheet.parsing_confidence:.2f})")
            
        except Exception as e:
            logger.warning(f"⚠️ Intelligent parsing failed for {file_id}: {e}, falling back to basic context")
            # Fallback to basic context
            context = self._build_basic_context(df)
        
        # Cache the context if file_id provided
        if file_id:
            spreadsheet_memory.store_context(thread_id, file_id, context)
            spreadsheet_memory.cache_df_metadata(file_id, {
                'shape': df.shape,
                'columns': list(df.columns)
            })
        
        return context
    
    def _format_intelligent_context(self, structured_context, df: pd.DataFrame) -> str:
        """Format structured context into LLM-friendly text format."""
        try:
            context_dict = structured_context.to_dict()
            
            context = f"""INTELLIGENT DOCUMENT ANALYSIS:

Document Type: {context_dict['document_type'].upper()}
Parsing Confidence: {structured_context.metadata.validation_checksums.get('confidence', 'N/A')}

DataFrame Shape: {df.shape[0]} rows × {df.shape[1]} columns
Columns: {list(df.columns)}

DOCUMENT STRUCTURE:
"""
            
            # Add section information
            for section_name, section_data in context_dict['sections'].items():
                context += f"\n--- {section_name.upper()} SECTION ---\n"
                
                if isinstance(section_data, dict):
                    if section_data.get('type') == 'table':
                        context += f"Type: Data Table\n"
                        if 'schema' in section_data:
                            context += f"Headers: {section_data['schema']}\n"
                        if 'row_count' in section_data:
                            context += f"Rows: {section_data['row_count']}\n"
                        if 'data' in section_data:
                            context += f"Sample Data: {section_data['data'][:3]}\n"
                    
                    elif section_data.get('type') == 'metadata':
                        context += f"Type: Metadata\n"
                        if 'content' in section_data:
                            for key, value in section_data['content'].items():
                                context += f"{key}: {value}\n"
                    
                    elif section_data.get('type') == 'calculations':
                        context += f"Type: Summary/Calculations\n"
                        if 'content' in section_data:
                            for key, value in section_data['content'].items():
                                context += f"{key}: {value}\n"
            
            # Add data types and statistics
            context += f"\nDATA TYPES:\n{df.dtypes.to_string()}\n"
            
            context += f"\nSAMPLE DATA (first 5 rows):\n{df.head().to_string()}\n"
            
            # Add column statistics
            context += "\nCOLUMN STATISTICS:\n"
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    context += f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
                else:
                    unique_count = df[col].nunique()
                    context += f"{col}: {unique_count} unique values"
                    if unique_count <= 10:
                        context += f", values: {df[col].unique().tolist()}"
                    context += "\n"
            
            # Add anti-hallucination markers
            if structured_context.metadata.validation_checksums:
                context += "\nVALIDATION CHECKSUMS (for accuracy):\n"
                for key, value in structured_context.metadata.validation_checksums.items():
                    context += f"{key}: {value}\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to format intelligent context: {e}")
            return self._build_basic_context(df)
    
    def _build_basic_context(self, df: pd.DataFrame) -> str:
        """Build basic context as fallback when intelligent parsing fails."""
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
        
        return context
    
    def _enhance_answer_with_data(self, template_answer: str, current_result: Any) -> str:
        """
        Enhance a templated answer (e.g. "X%", "Y%", "[calculated value]") with actual computed values.
        Extracts real values from result and inserts them into various placeholder formats.
        """
        if not current_result or not template_answer:
            return template_answer
        
        try:
            # Parse result into key-value pairs and values
            kv_pairs = {}  # For dict-like results: {"key": value}
            values_list = []  # For ordered values
            
            if isinstance(current_result, dict):
                # Dict or Series as dict
                for key, val in list(current_result.items())[:20]:
                    # Format the value
                    if isinstance(val, float) and val == int(val):
                        formatted_val = str(int(val))
                    elif isinstance(val, float):
                        formatted_val = f"{val:.2f}"
                    else:
                        formatted_val = str(val)
                    
                    kv_pairs[str(key)] = formatted_val
                    values_list.append(formatted_val)
                    
            elif isinstance(current_result, list):
                # List of dicts or values
                if current_result and isinstance(current_result[0], dict):
                    # List of dicts - extract values from first item or all
                    for item in current_result[:10]:
                        for key, val in item.items():
                            if isinstance(val, (int, float)):
                                if isinstance(val, float) and val == int(val):
                                    formatted_val = str(int(val))
                                elif isinstance(val, float):
                                    formatted_val = f"{val:.2f}"
                                else:
                                    formatted_val = str(val)
                                values_list.append(formatted_val)
                                if str(key) not in kv_pairs:
                                    kv_pairs[str(key)] = formatted_val
                else:
                    # List of scalars
                    for item in current_result[:20]:
                        if isinstance(item, (int, float)):
                            if isinstance(item, float) and item == int(item):
                                formatted_val = str(int(item))
                            elif isinstance(item, float):
                                formatted_val = f"{item:.2f}"
                            else:
                                formatted_val = str(item)
                            values_list.append(formatted_val)
                        else:
                            values_list.append(str(item))
            else:
                # Scalar value
                val = current_result
                if isinstance(val, float) and val == int(val):
                    values_list.append(str(int(val)))
                elif isinstance(val, float):
                    values_list.append(f"{val:.2f}")
                else:
                    values_list.append(str(val))
            
            # Replace multiple types of placeholders in answer
            enhanced = template_answer
            
            # 1. Replace X, Y, Z placeholders ONLY (safer than A, B, C which appear in words)
            placeholders = ["X", "Y", "Z"]
            for idx, placeholder in enumerate(placeholders):
                if placeholder in enhanced and idx < len(values_list):
                    # Use word boundary to avoid replacing letters inside words
                    import re
                    # Replace X, Y, Z only when they're standalone (not part of a word)
                    pattern = r'\b' + placeholder + r'\b'
                    enhanced = re.sub(pattern, values_list[idx], enhanced, count=1)
            
            # 2. Replace [calculated value] style placeholders
            import re
            calculated_pattern = r'\[calculated value\]|\[value\]|\[result\]'
            matches = re.findall(calculated_pattern, enhanced, re.IGNORECASE)
            for idx, match in enumerate(matches):
                if idx < len(values_list):
                    enhanced = enhanced.replace(match, values_list[idx], 1)
            
            # 3. Replace [Month], [Category] style placeholders with actual key names
            bracket_pattern = r'\[(\w+)\]'
            bracket_matches = re.findall(bracket_pattern, enhanced)
            for key_name in bracket_matches:
                if key_name in kv_pairs:
                    enhanced = enhanced.replace(f"[{key_name}]", kv_pairs[key_name], 1)
            
            # 4. Replace {} style placeholders
            if "{}" in enhanced and values_list:
                for val in values_list[:5]:
                    enhanced = enhanced.replace("{}", val, 1)
            
            return enhanced
        except Exception as e:
            logger.debug(f"Could not enhance answer with data: {e}")
            return template_answer

    def _find_similar_columns(self, df: pd.DataFrame, search_terms: List[str]) -> Dict[str, List[str]]:
        """
        Find columns that might match the search terms using fuzzy matching.
        
        Args:
            df: DataFrame to search in
            search_terms: List of terms to search for (e.g., ['category', 'categories'])
        
        Returns:
            Dict mapping search terms to list of matching column names
        """
        from difflib import get_close_matches
        import re
        
        available_columns = list(df.columns)
        matches = {}
        
        for term in search_terms:
            term_lower = term.lower()
            column_matches = []
            
            # Exact match (case insensitive)
            for col in available_columns:
                if term_lower == col.lower():
                    column_matches.append(col)
            
            # Partial match (term appears in column name)
            if not column_matches:
                for col in available_columns:
                    if term_lower in col.lower() or col.lower() in term_lower:
                        column_matches.append(col)
            
            # Fuzzy match using difflib
            if not column_matches:
                fuzzy_matches = get_close_matches(term_lower, [col.lower() for col in available_columns], n=3, cutoff=0.6)
                for fuzzy in fuzzy_matches:
                    # Find the original column name
                    for col in available_columns:
                        if col.lower() == fuzzy:
                            column_matches.append(col)
                            break
            
            # Word boundary matching (e.g., "Product Category" matches "category")
            if not column_matches:
                for col in available_columns:
                    col_words = re.findall(r'\b\w+\b', col.lower())
                    if term_lower in col_words:
                        column_matches.append(col)
            
            if column_matches:
                matches[term] = column_matches
        
        return matches

    def _enhance_query_with_column_suggestions(self, question: str, df: pd.DataFrame) -> str:
        """
        Enhance the user's question by suggesting actual column names when they use generic terms.
        
        Args:
            question: Original user question
            df: DataFrame to analyze
        
        Returns:
            Enhanced question with column suggestions
        """
        # Common terms users might use that need column mapping
        category_terms = ['category', 'categories', 'type', 'types', 'kind', 'group']
        quantity_terms = ['quantity', 'quantities', 'amount', 'amounts', 'count', 'total', 'sum']
        
        # Find matching columns
        category_matches = self._find_similar_columns(df, category_terms)
        quantity_matches = self._find_similar_columns(df, quantity_terms)
        
        enhanced_question = question
        suggestions = []
        
        # Add category column suggestions
        for term, matches in category_matches.items():
            if term.lower() in question.lower():
                if matches:
                    suggestions.append(f"For '{term}', I found column(s): {', '.join(matches)}")
        
        # Add quantity column suggestions
        for term, matches in quantity_matches.items():
            if term.lower() in question.lower():
                if matches:
                    suggestions.append(f"For '{term}', I found column(s): {', '.join(matches)}")
        
        if suggestions:
            enhanced_question += "\n\nColumn suggestions based on your question:\n" + "\n".join(suggestions)
            enhanced_question += f"\n\nAvailable columns: {', '.join(df.columns)}"
        
        return enhanced_question

    def _check_percentage_invariants(self, answer: str, current_result: Any) -> Optional[str]:
        """Lightweight sanity check: if answer expresses percentages, verify they sum ~100."""
        if not answer or "%" not in answer:
            return None

        # Extract numeric values from result
        numbers = []
        try:
            if isinstance(current_result, dict):
                numbers = [float(v) for v in current_result.values() if isinstance(v, (int, float))]
            elif isinstance(current_result, list):
                if current_result and isinstance(current_result[0], dict):
                    for item in current_result:
                        for v in item.values():
                            if isinstance(v, (int, float)):
                                numbers.append(float(v))
                else:
                    numbers = [float(v) for v in current_result if isinstance(v, (int, float))]
        except Exception:
            numbers = []

        if not numbers:
            return None

        total = sum(numbers)
        if total < 90 or total > 110:
            return f"Sanity check: percentage totals look off (sum={total:.1f}). Review calculation."
        return None
    
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
1. ALWAYS respond with valid JSON in the exact format specified below (no markdown)
2. Use the DataFrame variable 'df' for all operations
3. Handle column names with spaces using backticks in query() OR bracket notation df['col name']
4. If column names look combined (e.g., "YearsExperience,Salary"), split them into separate columns and drop empty/unnamed columns before analysis
5. Keep "thinking" brief (1-2 sentences). Do NOT include long step-by-step reasoning.
6. "pandas_code" MUST be a single-line string. Do not include raw newlines inside JSON strings.
7. ALWAYS provide a helpful, human-readable final_answer when is_final_answer is true

=== CRITICAL CONSTRAINTS ===
- You are ONLY for analytical questions (why/how/anomaly detection/calculations)
- DO NOT answer summary/preview/schema requests (those go to /get_summary)
- ASSUME: Orchestrator already classified this as 'qa' task type
- ASSUME: User question is already validated for analytics
- FOCUS: Execute the analysis, don't second-guess intent

=== RESPONSE FORMAT (JSON) ===
{{
    "thinking": "Brief intent (1-2 sentences)",
    "needs_more_steps": true/false,
    "pandas_code": "df.query('column > value')",
    "explanation": "Clear explanation of what this code does",
    "is_final_answer": true/false,
    "final_answer": "Human-readable answer (REQUIRED if is_final_answer is true)"
}}

=== ANSWER GENERATION GUIDELINES ===
When you set is_final_answer to true, the final_answer MUST:
1. Be a clear, human-readable statement answering the user's question
2. Use PLACEHOLDERS for values that will be computed: Use X, Y, Z, A, B, C for numeric/categorical values
3. Example Template Answers:
   - For grouped results: "Category X has Y transactions with Z average price"
   - For percentages: "Beauty: X%, Clothing: Y%, Electronics: Z%"
   - For single value: "The total is X"
   - For multiple stats: "Sales: X, Average: Y, Count: Z"
4. If the result is a dict/Series with key-value pairs, show all keys with X, Y, Z placeholders
5. Do NOT include actual values in your answer - the system will fill in placeholders with real computed values

=== EXAMPLES ===

User: "Show me all people older than 30"
Response: {{"thinking": "User wants to filter rows where age > 30", "needs_more_steps": false, "pandas_code": "df.query('age > 30')", "explanation": "Filtering to show only rows where age column exceeds 30", "is_final_answer": true, "final_answer": "Here are the X people older than 30 (results shown above)"}}

User: "What's the average salary by department?"
Response: {{"thinking": "User wants grouped aggregation by department", "needs_more_steps": false, "pandas_code": "df.groupby('department')['salary'].mean()", "explanation": "Grouping by department and calculating mean salary", "is_final_answer": true, "final_answer": "Average salary by department: Engineering: X, Sales: Y, HR: Z"}}

User: "What percentage of total sales does each Product Category represent?"
Response: {{"thinking": "Calculate total sales per category and their percentage of grand total", "needs_more_steps": false, "pandas_code": "df.groupby('Product Category')['Total Amount'].sum() / df['Total Amount'].sum() * 100", "explanation": "Group by Product Category, sum Total Amount per category, divide by total sales, multiply by 100 for percentage", "is_final_answer": true, "final_answer": "Percentage breakdown by Product Category: Beauty: X%, Clothing: Y%, Electronics: Z%"}}

User: "How many rows have Feature1 greater than 50?"
Response: {{"thinking": "Count rows where Feature1 > 50", "needs_more_steps": false, "pandas_code": "len(df[df['Feature1'] > 50])", "explanation": "Counting rows where Feature1 exceeds 50", "is_final_answer": true, "final_answer": "There are X rows where Feature1 is greater than 50"}}
"""
    
    def _validate_code_against_schema(self, df: pd.DataFrame, code: str) -> Optional[str]:
        """Validate pandas code against DataFrame schema before execution.
        Returns error message if validation fails, None if valid."""
        import re
        
        actual_columns = set(df.columns)
        
        # Extract column references from code
        # Patterns: df['column'], df["column"], df.column, df['column'].method
        patterns = [
            r"df\['([^']+)'\]",  # df['column']
            r'df\["([^"]+)"\]',  # df["column"]
            r"df\.([a-zA-Z_][a-zA-Z0-9_]*)",  # df.column (but not df.head, df.groupby, etc.)
        ]
        
        referenced_columns = set()
        for pattern in patterns:
            matches = re.findall(pattern, code)
            referenced_columns.update(matches)
        
        # Filter out pandas methods (not actual columns)
        pandas_methods = {'head', 'tail', 'describe', 'info', 'shape', 'columns', 'dtypes', 
                         'groupby', 'sort_values', 'query', 'drop', 'rename', 'insert',
                         'select_dtypes', 'sum', 'mean', 'count', 'max', 'min', 'fillna',
                         'dropna', 'isnull', 'notnull', 'merge', 'join', 'to_dict', 'copy',
                         'nunique', 'unique', 'value_counts', 'apply', 'map', 'applymap',
                         'agg', 'aggregate', 'transform', 'sample', 'duplicated', 'astype',
                         'isna', 'notna', 'reset_index', 'set_index', 'loc', 'iloc', 'at', 'iat'}
        referenced_columns = {col for col in referenced_columns if col not in pandas_methods}
        
        # Check for non-existent columns
        missing_columns = referenced_columns - actual_columns
        
        if missing_columns:
            # Find similar column names (fuzzy match)
            from difflib import get_close_matches
            suggestions = {}
            for missing_col in missing_columns:
                matches = get_close_matches(missing_col, actual_columns, n=3, cutoff=0.6)
                if matches:
                    suggestions[missing_col] = matches
            
            error_msg = f"Column(s) not found: {', '.join(missing_columns)}. "
            error_msg += f"Available columns: {', '.join(sorted(actual_columns))}. "
            if suggestions:
                error_msg += "Did you mean: "
                error_msg += ", ".join([f"'{missing}' -> {matches}" for missing, matches in suggestions.items()])
            
            return error_msg
        
        return None
    
    def _is_mostly_numeric(self, series: pd.Series) -> bool:
        """Check if a series has mostly numeric values (>= 60%)"""
        if len(series) == 0:
            return False
        
        numeric_count = 0
        total_count = 0
        
        for val in series.dropna():
            total_count += 1
            try:
                float(val)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        if total_count == 0:
            return False
        
        return (numeric_count / total_count) >= 0.6
    
    def _detect_dtype_drift(self, original_df: pd.DataFrame, current_df: pd.DataFrame) -> Optional[Tuple[AnomalyDetails, List[UserChoice]]]:
        """Detect dtype drift between original and current DataFrames.
        Also detects pre-existing object columns that should be numeric.
        
        Returns:
            Tuple of (AnomalyDetails, List[UserChoice]) if drift detected, None otherwise
        """
        # Compare dtypes
        original_dtypes = original_df.dtypes.to_dict()
        current_dtypes = current_df.dtypes.to_dict()
        
        # Find columns with dtype changes OR pre-existing object columns with mostly numeric values
        drifted_columns = []
        for col in original_dtypes:
            if col in current_dtypes:
                orig_dtype = str(original_dtypes[col])
                curr_dtype = str(current_dtypes[col])
                try:
                    logger.debug(f"[DriftCheck] Column='{col}' orig_dtype='{orig_dtype}' curr_dtype='{curr_dtype}'")
                except Exception:
                    pass
                
                # Case 1: Detect numeric -> object drift (dtype changed during execution)
                if orig_dtype in ['int64', 'float64'] and curr_dtype == 'object':
                    drifted_columns.append(col)
                
                # Case 2: Detect pre-existing object columns that should be numeric
                # (column was object from the start but has mostly numeric values)
                elif orig_dtype == 'object' and curr_dtype == 'object':
                    # Check if column has mostly numeric-like values
                    is_numeric_like = self._is_mostly_numeric(current_df[col])
                    try:
                        logger.debug(f"[DriftCheck] Column='{col}' numeric_like={is_numeric_like}")
                    except Exception:
                        pass
                    if is_numeric_like:
                        drifted_columns.append(col)
        
        if not drifted_columns:
            return None
        
        # Build anomaly details
        sample_values = {}
        for col in drifted_columns:
            # Get non-numeric values in supposedly numeric column
            non_numeric = []
            for val in current_df[col].dropna().unique()[:5]:
                try:
                    float(val)
                except (ValueError, TypeError):
                    non_numeric.append(val)
            sample_values[col] = non_numeric
        
        affected_cols_str = ', '.join([f"'{col}'" for col in drifted_columns])
        
        anomaly = AnomalyDetails(
            anomaly_type="dtype_drift",
            affected_columns=drifted_columns,
            message=f"I detected that column(s) {affected_cols_str} contain mostly numbers, but some values are strings (dtype drift). This can affect calculations.",
            current_dtypes={col: str(current_dtypes[col]) for col in drifted_columns},
            expected_dtypes={col: str(original_dtypes[col]) for col in drifted_columns},
            sample_values=sample_values,
            severity="warning"
        )
        
        # Build user choices
        choices = [
            UserChoice(
                id="convert_numeric",
                label="Convert to numeric (invalid → NaN)",
                description=f"Convert {affected_cols_str} to numeric, replacing invalid values with NaN. This allows calculations to proceed.",
                is_safe=True
            ),
            UserChoice(
                id="ignore_rows",
                label="Ignore invalid rows",
                description=f"Filter out rows where {affected_cols_str} have non-numeric values before calculating.",
                is_safe=True
            ),
            UserChoice(
                id="treat_as_text",
                label="Treat as text and continue",
                description=f"Keep {affected_cols_str} as text. Numeric operations may fail.",
                is_safe=True
            ),
            UserChoice(
                id="cancel",
                label="Cancel the analysis",
                description="Stop the current operation without making changes.",
                is_safe=True
            )
        ]
        
        return anomaly, choices
    
    def _safe_execute_pandas(self, df: pd.DataFrame, code: str) -> Tuple[Any, pd.DataFrame, Optional[str]]:
        """Safely execute pandas code and return result or error"""
        # Pre-execution validation
        validation_error = self._validate_code_against_schema(df, code)
        if validation_error:
            logger.warning(f"⚠️ Code validation failed: {validation_error}")
            return None, df, validation_error
        
        # Minimal safe globals injected for LLM-generated code
        safe_globals = {
            "__builtins__": {},  # block dangerous builtins
            "pd": pd,
            "np": np,
            "float": float,
            "int": int,
            "str": str,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
        }
        local_vars = {"df": df}

        try:
            # First try evaluating as an expression to capture the returned value (e.g., Series/array)
            try:
                result_value = eval(code, safe_globals, local_vars)
                updated_df = local_vars.get("df", df)
                return result_value, updated_df, None
            except SyntaxError:
                # Not an expression (likely assignment/mutation); fall back to exec
                pass

            # Execute statements (mutations/assignments)
            exec(code, safe_globals, local_vars)

            updated_df = local_vars.get("df", df)

            # If the specific variable 'result' exists, return it
            if "result" in local_vars:
                return local_vars["result"], updated_df, None

            # Otherwise, return the df itself as the result (e.g. after filtering)
            return updated_df, updated_df, None

        except KeyError as e:
            # Enhanced KeyError handling with column suggestions
            column_name = str(e).strip("'\"")
            from difflib import get_close_matches
            similar = get_close_matches(column_name, df.columns, n=3, cutoff=0.6)
            error_msg = f"Column '{column_name}' not found. Available columns: {', '.join(df.columns)}."
            if similar:
                error_msg += f" Did you mean: {', '.join(similar)}?"
            return None, df, error_msg
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
        
        # Initialize current_df with the input dataframe
        current_df = df.copy()
        
        # Enhance question with column suggestions
        enhanced_question = self._enhance_query_with_column_suggestions(question, current_df)
        
        df_context = self._get_dataframe_context(current_df, file_id, thread_id)
        system_prompt = self._build_system_prompt(df_context, session_context, all_files_context)
        
        steps_taken = []
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {enhanced_question}"}
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
                raw_response_text = response_text
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

                    def _extract_json_object(text: str) -> str:
                        start = text.find("{")
                        end = text.rfind("}")
                        return text[start:end + 1] if start != -1 and end != -1 and end > start else text

                    def _repair_common_json_issues(text: str) -> str:
                        # Repair common invalid JSON emitted by LLMs:
                        # - raw newlines/tabs inside quoted strings
                        # This does NOT attempt to fix truncation/unterminated strings.
                        out = []
                        in_string = False
                        escape = False
                        for ch in text:
                            if escape:
                                out.append(ch)
                                escape = False
                                continue
                            if ch == "\\":
                                out.append(ch)
                                escape = True
                                continue
                            if ch == '"':
                                out.append(ch)
                                in_string = not in_string
                                continue
                            if in_string and ch == "\n":
                                out.append("\\n")
                                continue
                            if in_string and ch == "\r":
                                out.append("\\r")
                                continue
                            if in_string and ch == "\t":
                                out.append("\\t")
                                continue
                            out.append(ch)
                        return "".join(out)

                    candidate = _extract_json_object(response_text)
                    candidate = _repair_common_json_issues(candidate)

                    response = json.loads(candidate)

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
                        "raw_response": response_text[:500],
                        "raw_response_text": raw_response_text
                    })
                    # Ask LLM to fix the response with stricter instruction
                    conversation.append({"role": "assistant", "content": response_text[:500]})
                    conversation.append({
                        "role": "user", 
                        "content": "Please respond with valid JSON only (no markdown). Keep thinking brief. Ensure pandas_code is a single line. Required keys: thinking, pandas_code, is_final_answer, final_answer."
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
                    "explanation": explanation,
                    "raw_response_text": raw_response_text
                }
                
                # Execute pandas code if provided
                if pandas_code:
                    # Capture before state for observation
                    before_shape = current_df.shape
                    before_columns = current_df.columns.tolist()
                    before_sample = current_df.head(3).to_dict(orient="records") if len(current_df) > 0 else []
                    
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
                    
                    # Capture after state and create observation
                    after_shape = updated_df.shape
                    after_columns = updated_df.columns.tolist()
                    after_sample = updated_df.head(3).to_dict(orient="records") if len(updated_df) > 0 else []
                    
                    # Calculate changes
                    cols_added = list(set(after_columns) - set(before_columns))
                    cols_removed = list(set(before_columns) - set(after_columns))
                    rows_change = after_shape[0] - before_shape[0]
                    
                    changes_parts = []
                    if cols_added:
                        changes_parts.append(f"Added columns: {', '.join(cols_added)}")
                    if cols_removed:
                        changes_parts.append(f"Removed columns: {', '.join(cols_removed)}")
                    if rows_change > 0:
                        changes_parts.append(f"Added {rows_change} rows")
                    elif rows_change < 0:
                        changes_parts.append(f"Removed {abs(rows_change)} rows")
                    if before_shape == after_shape and before_columns == after_columns:
                        changes_parts.append("Data values modified")
                    
                    step_info["observation"] = {
                        "before_shape": before_shape,
                        "after_shape": after_shape,
                        "changes_summary": "; ".join(changes_parts) if changes_parts else "No structural changes"
                    }
                        
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
                        # Handle Series with MultiIndex (from groupby with multiple columns)
                        # These produce tuple keys that JSON can't serialize
                        if isinstance(result.index, pd.MultiIndex):
                            # Convert MultiIndex Series to DataFrame then to records
                            result_df = result.reset_index()
                            records = result_df.head(10).to_dict(orient="records")
                            safe_records = json.loads(json.dumps(records, default=str))
                            step_info["result_preview"] = safe_records
                            step_info["result_shape"] = result_df.shape
                            current_result = result_df.to_dict(orient="records")
                        else:
                            # Regular Series with simple index
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
                    
                    # Log raw answer before enhancement
                    logger.info(f"🔍 Raw answer: {final_answer[:150]}")
                    
                    # Enhance answer with actual computed values
                    enhanced_answer = self._enhance_answer_with_data(final_answer, current_result)

                    # Percentage sanity guard
                    sanity_warning = self._check_percentage_invariants(enhanced_answer, current_result)
                    if sanity_warning:
                        enhanced_answer = f"{enhanced_answer} ({sanity_warning})"

                    logger.info(f"🔍 Enhanced answer: {enhanced_answer[:150]}")
                    
                    # DTYPE DRIFT DETECTION: Check for dtype changes before finalizing
                    drift_check = self._detect_dtype_drift(df, current_df)
                    if drift_check is not None:
                        anomaly, user_choices = drift_check
                        logger.warning(f"⚠️ Dtype drift detected: {anomaly.affected_columns}")
                        
                        # Return anomaly result instead of final answer
                        query_result = QueryResult(
                            question=question,
                            answer=anomaly.message,
                            steps_taken=steps_taken,
                            final_data=None,
                            success=False,
                            status="anomaly_detected",
                            needs_user_input=True,
                            anomaly=anomaly,
                            user_choices=user_choices,
                            pending_action="dtype_conversion",
                            final_dataframe=current_df
                        )
                        
                        # Add metrics
                        query_metrics["latency_ms"] = (time.time() - query_start) * 1000
                        end_memory = process.memory_info().rss / 1024 / 1024
                        query_metrics["memory_used_mb"] = end_memory - start_memory
                        query_result.execution_metrics = query_metrics
                        
                        self._log_execution_metrics(query_metrics, False)
                        return query_result
                    
                    # Create query result first
                    query_result = QueryResult(
                        question=question,
                        answer=enhanced_answer,
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
                    
                    # Cache the result (keyed by file and question)
                    spreadsheet_memory.cache_query_result(file_id, question, query_result, thread_id)
                    
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
    
    async def generate_actions(
        self,
        df: pd.DataFrame,
        instruction: str,
        df_context: Dict[str, Any] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Generate structured actions from natural language instruction.
        
        This method uses the LLM to convert natural language instructions
        into a list of structured action objects (filter, sort, add_column, etc.)
        instead of raw pandas code.
        
        Args:
            df: DataFrame to generate actions for
            instruction: Natural language instruction
            df_context: Pre-computed DataFrame context (optional)
        
        Returns:
            Tuple of (actions_list, reasoning)
        """
        if not df_context:
            df_context = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample": df.head(5).to_dict(orient='records')
            }
        
        # Build action vocabulary system prompt
        system_prompt = f"""You are an expert at converting natural language instructions into structured spreadsheet actions.

=== DATAFRAME CONTEXT ===
Shape: {df_context['shape'][0]} rows × {df_context['shape'][1]} columns
Columns: {', '.join(df_context['columns'])}
Data Types: {', '.join([f"{col}({dtype})" for col, dtype in df_context['dtypes'].items()])}

Sample Data (first 5 rows):
{json.dumps(df_context['sample'], indent=2, default=str)}

=== AVAILABLE ACTIONS ===

1. **filter** - Filter rows by condition
   {{
     "action_type": "filter",
     "column": "column_name",
     "operator": "=="|"!="|">"|"<"|">="|"<="|"contains"|"startswith"|"endswith",
     "value": "value_to_compare"
   }}

2. **sort** - Sort by column(s)
   {{
     "action_type": "sort",
     "columns": ["col1", "col2"],
     "ascending": [true, false]
   }}

3. **add_column** - Add calculated column
   {{
     "action_type": "add_column",
     "new_column": "TotalScore",
     "formula": "{{Feature1}} + {{Feature2}}"
   }}

4. **rename_column** - Rename columns
   {{
     "action_type": "rename_column",
     "old_names": ["OldName1", "OldName2"],
     "new_names": ["NewName1", "NewName2"]
   }}

5. **drop_column** - Remove columns
   {{
     "action_type": "drop_column",
     "columns": ["col1", "col2"]
   }}

6. **groupby** - Group and aggregate
   {{
     "action_type": "groupby",
     "group_by_columns": ["Department"],
     "aggregate": {{"Salary": "mean", "Age": "count"}}
   }}

7. **fillna** - Fill missing values
   {{
     "action_type": "fillna",
     "columns": ["col1"],
     "method": "value"|"forward"|"backward"|"mean"|"median",
     "fill_value": 0  // only if method is "value"
   }}

8. **drop_duplicates** - Remove duplicates
   {{
     "action_type": "drop_duplicates",
     "subset": ["col1", "col2"],
     "keep": "first"|"last"
   }}

9. **add_serial_number** - Add serial number column
   {{
     "action_type": "add_serial_number",
     "column_name": "Sl.No.",
     "start": 1,
     "position": "first"|"last"
   }}

=== RESPONSE FORMAT (JSON) ===
{{
  "reasoning": "Step-by-step explanation of what actions are needed",
  "actions": [
    // List of action objects
  ]
}}

=== INSTRUCTIONS ===
1. Analyze the user instruction carefully
2. Break it down into a sequence of actions
3. Use the simplest actions that accomplish the goal
4. Ensure column names match exactly what's in the DataFrame
5. For calculations, use {{column_name}} syntax in formulas
6. Return valid JSON only (no markdown)

Now convert this instruction to actions:
"{instruction}"
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        try:
            response_text, _ = self._get_completion(messages, temperature=0.1)
            
            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            response = json.loads(response_text.strip())
            
            actions = response.get("actions", [])
            reasoning = response.get("reasoning", "")
            
            return actions, reasoning
            
        except Exception as e:
            logger.error(f"Failed to generate actions: {e}", exc_info=True)
            # Return empty action list on failure
            return [], f"Error generating actions: {str(e)}"
    
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

