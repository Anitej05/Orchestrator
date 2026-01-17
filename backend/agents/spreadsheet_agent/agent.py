"""
Main Spreadsheet Agent Class

Handles orchestrator communication with proper request/response formatting.
Routes actions to appropriate components (parser, executor, anomaly detector).

Requirements: 9.1, 9.3, 14.1, 14.2, 14.3
"""

import logging
import time
from typing import Dict, Any, Optional
from difflib import get_close_matches

import pandas as pd

from agents.spreadsheet_agent.dialogue_manager import (
    dialogue_manager,
    ResponseStatus,
    ExecutionMetrics
)
from agents.spreadsheet_agent.dataframe_cache import DataFrameCache
from agents.spreadsheet_agent.query_executor import QueryExecutor, QueryPlan
from agents.spreadsheet_agent.anomaly_detector import AnomalyDetector

# Import parsing components
from agents.spreadsheet_agent.parsing import (
    DocumentSectionDetector,
    IntentionalGapDetector,
    TableDetector,
    ContextBuilder,
    MetadataExtractor,
    SchemaExtractor
)
from agents.spreadsheet_agent.parsing_models import ParsedSpreadsheet, DocumentType
from agents.spreadsheet_agent.file_loader import FileLoader

logger = logging.getLogger(__name__)


class SpreadsheetAgent:
    """
    Main agent class for spreadsheet operations.
    
    Responsibilities:
    - Handle /execute and /continue endpoints
    - Route actions to appropriate components
    - Format responses for orchestrator
    - Handle errors gracefully with user-friendly messages
    - Implement fuzzy column name matching
    """
    
    def __init__(self):
        """Initialize the spreadsheet agent"""
        self.logger = logging.getLogger(f"{__name__}.SpreadsheetAgent")
        self.dialogue_manager = dialogue_manager
        self.dataframe_cache = DataFrameCache()
        self.query_executor = QueryExecutor()
        self.anomaly_detector = AnomalyDetector()
        
        # Initialize parsing components
        self.document_section_detector = DocumentSectionDetector()
        self.intentional_gap_detector = IntentionalGapDetector()
        self.table_detector = TableDetector()
        self.context_builder = ContextBuilder()
        self.metadata_extractor = MetadataExtractor()
        self.schema_extractor = SchemaExtractor()
        self.file_loader = FileLoader()
    
    # ========================================================================
    # MAIN ENDPOINTS (Task 10.1)
    # ========================================================================
    
    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle new task execution from orchestrator.
        
        Args:
            request: Dictionary with:
                - thread_id: str
                - file_id: Optional[str]
                - action: Optional[str] (e.g., 'analyze', 'query', 'aggregate')
                - prompt: Optional[str] (natural language query)
                - parameters: Dict[str, Any]
        
        Returns:
            AgentResponse dictionary with status, result, explanation, metrics
            
        Requirements: 9.1, 9.3
        """
        start_time = time.time()
        thread_id = request.get('thread_id', 'default')
        file_id = request.get('file_id')
        action = request.get('action')
        prompt = request.get('prompt')
        parameters = request.get('parameters', {})
        
        self.logger.info(f"[EXECUTE] thread_id={thread_id}, file_id={file_id}, action={action}")
        
        try:
            # Validate request
            if not file_id:
                return self._error_response(
                    "file_id is required",
                    start_time,
                    error_details={"request": request}
                )
            
            # Load dataframe
            df, error = self._load_dataframe(file_id, thread_id)
            if error:
                return error
            
            # Route to action handler
            if action:
                response = self._route_action(action, df, file_id, thread_id, parameters, start_time)
            elif prompt:
                response = self._handle_prompt(prompt, df, file_id, thread_id, parameters, start_time)
            else:
                return self._error_response(
                    "Either 'action' or 'prompt' is required",
                    start_time
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"[EXECUTE] Unexpected error: {e}", exc_info=True)
            return self._error_response(
                f"Unexpected error: {str(e)}",
                start_time,
                error_details={"exception_type": type(e).__name__}
            )
    
    def continue_execution(self, thread_id: str, user_input: str) -> Dict[str, Any]:
        """
        Resume paused execution with user input.
        
        Args:
            thread_id: Thread identifier
            user_input: User's answer to the pending question
        
        Returns:
            AgentResponse dictionary
            
        Requirements: 9.3
        """
        start_time = time.time()
        
        self.logger.info(f"[CONTINUE] thread_id={thread_id}, user_input={user_input[:50]}...")
        
        try:
            # Load dialogue state
            state = self.dialogue_manager.load_state(thread_id)
            
            if not state:
                return self._error_response(
                    f"No pending dialogue found for thread {thread_id}",
                    start_time
                )
            
            # Get pending operation context
            pending_operation = state.get('pending_operation')
            file_id = state.get('file_id')
            anomaly = state.get('anomaly')
            
            if not pending_operation:
                return self._error_response(
                    "No pending operation to continue",
                    start_time
                )
            
            # Load dataframe
            df, error = self._load_dataframe(file_id, thread_id)
            if error:
                return error
            
            # Handle anomaly resolution
            if anomaly and pending_operation == 'anomaly_resolution':
                response = self._handle_anomaly_resolution(
                    df, file_id, thread_id, anomaly, user_input, start_time
                )
            else:
                response = self._error_response(
                    f"Unknown pending operation: {pending_operation}",
                    start_time
                )
            
            # Clear pending question
            self.dialogue_manager.clear_pending_question(thread_id)
            
            return response
            
        except Exception as e:
            self.logger.error(f"[CONTINUE] Unexpected error: {e}", exc_info=True)
            return self._error_response(
                f"Unexpected error: {str(e)}",
                start_time
            )
    
    # ========================================================================
    # ACTION ROUTING (Task 10.1)
    # ========================================================================
    
    def _route_action(
        self,
        action: str,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Route action to appropriate handler.
        
        Args:
            action: Action name (e.g., 'analyze', 'query', 'aggregate')
            df: DataFrame to operate on
            file_id: File identifier
            thread_id: Thread identifier
            parameters: Action parameters
            start_time: Operation start time
        
        Returns:
            AgentResponse dictionary
            
        Requirements: 9.1
        """
        self.logger.info(f"[ROUTE] action={action}, parameters={list(parameters.keys())}")
        
        # Map actions to handlers
        action_handlers = {
            'analyze': self._handle_analyze,
            'parse': self._handle_parse,
            'query': self._handle_query,
            'aggregate': self._handle_aggregate,
            'filter': self._handle_filter,
            'sort': self._handle_sort,
            'transform': self._handle_transform,
            'detect_anomalies': self._handle_detect_anomalies,
            'build_context': self._handle_build_context
        }
        
        handler = action_handlers.get(action)
        
        if not handler:
            return self._error_response(
                f"Unknown action: {action}. Available actions: {list(action_handlers.keys())}",
                start_time
            )
        
        try:
            return handler(df, file_id, thread_id, parameters, start_time)
        except Exception as e:
            self.logger.error(f"[ROUTE] Handler error for action '{action}': {e}", exc_info=True)
            return self._error_response(
                f"Action '{action}' failed: {str(e)}",
                start_time,
                error_details={"action": action, "exception_type": type(e).__name__}
            )
    
    # ========================================================================
    # ACTION HANDLERS
    # ========================================================================
    
    def _handle_analyze(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle analyze action - return basic DataFrame information"""
        self.logger.info(f"[ANALYZE] file_id={file_id}")
        
        # Create metrics
        metrics = self.dialogue_manager.create_metrics(
            start_time=start_time,
            rows_processed=len(df),
            columns_affected=len(df.columns)
        )
        
        # Format result with basic DataFrame info
        result = {
            "file_id": file_id,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(5).to_dict(orient='records'),
            "summary_stats": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {}
        }
        
        response = self.dialogue_manager.create_success_response(
            result=result,
            explanation=f"Analyzed spreadsheet: {df.shape[0]} rows × {df.shape[1]} columns",
            metrics=metrics
        )
        
        return response.to_dict()
    
    def _handle_query(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle query action - execute pandas query"""
        query_str = parameters.get('query')
        
        if not query_str:
            return self._error_response("'query' parameter is required", start_time)
        
        self.logger.info(f"[QUERY] query={query_str[:100]}")
        
        try:
            # Create query plan for pandas query
            query_plan = QueryPlan(
                operation='filter',
                conditions={'query_string': query_str}
            )
            
            # Execute query using query executor
            result = self.query_executor.execute_pandas_query(df, query_str)
            
            if not result.success:
                return self._error_response(result.error or "Query execution failed", start_time)
            
            result_df = result.data
            
            # Check for anomalies in the result
            anomaly = self.anomaly_detector.detect_anomalies(result_df)
            
            if anomaly:
                # Save state and return NEEDS_INPUT
                self.dialogue_manager.save_state(thread_id, {
                    'pending_operation': 'anomaly_resolution',
                    'file_id': file_id,
                    'anomaly': anomaly,
                    'query': query_str
                })
                
                metrics = self.dialogue_manager.create_metrics(
                    start_time=start_time,
                    rows_processed=len(df)
                )
                
                response = self.dialogue_manager.create_anomaly_response(
                    anomaly=anomaly,
                    metrics=metrics
                )
                
                return response.to_dict()
            
            # Create metrics
            metrics = self.dialogue_manager.create_metrics(
                start_time=start_time,
                rows_processed=len(result_df),
                columns_affected=len(result_df.columns)
            )
            
            # Format result
            result_data = {
                "query": query_str,
                "rows_returned": len(result_df),
                "data": result_df.head(100).to_dict(orient='records')
            }
            
            response = self.dialogue_manager.create_success_response(
                result=result_data,
                explanation=f"Query returned {len(result_df)} rows",
                metrics=metrics
            )
            
            return response.to_dict()
            
        except Exception as e:
            return self._handle_pandas_error(e, start_time, df, context="query execution")
    
    def _handle_aggregate(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle aggregate action - compute aggregations"""
        column = parameters.get('column')
        operation = parameters.get('operation', 'sum')
        
        if not column:
            return self._error_response("'column' parameter is required", start_time)
        
        # Validate column exists
        if column not in df.columns:
            return self._handle_column_not_found(column, df, start_time)
        
        self.logger.info(f"[AGGREGATE] column={column}, operation={operation}")
        
        try:
            # Create query plan for aggregation
            query_plan = QueryPlan(
                operation='aggregate',
                conditions={
                    'function': operation,
                    'column': column
                }
            )
            
            # Execute aggregation using query executor
            result = self.query_executor.execute_query(df, query_plan)
            
            if not result.success:
                return self._error_response(result.error or "Aggregation failed", start_time)
            
            result_value = result.data
            
            # Create metrics
            metrics = self.dialogue_manager.create_metrics(
                start_time=start_time,
                rows_processed=len(df),
                columns_affected=1
            )
            
            # Format result
            result_data = {
                "column": column,
                "operation": operation,
                "value": float(result_value) if pd.notna(result_value) else None
            }
            
            response = self.dialogue_manager.create_success_response(
                result=result_data,
                explanation=f"{operation.title()} of '{column}': {result_value}",
                metrics=metrics
            )
            
            return response.to_dict()
            
        except Exception as e:
            return self._handle_pandas_error(e, start_time, df, context=f"aggregation on column '{column}'")
    
    def _handle_filter(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle filter action"""
        column = parameters.get('column')
        operator = parameters.get('operator', '==')
        value = parameters.get('value')
        
        if not column or value is None:
            return self._error_response("'column' and 'value' parameters are required", start_time)
        
        # Validate column exists
        if column not in df.columns:
            return self._handle_column_not_found(column, df, start_time)
        
        self.logger.info(f"[FILTER] column={column}, operator={operator}, value={value}")
        
        try:
            # Create query plan for filter
            query_plan = QueryPlan(
                operation='filter',
                conditions={
                    'column': column,
                    'operator': operator,
                    'value': value
                }
            )
            
            # Execute filter using query executor
            result = self.query_executor.execute_query(df, query_plan)
            
            if not result.success:
                return self._error_response(result.error or "Filter failed", start_time)
            
            result_df = result.data
            
            # Create metrics
            metrics = self.dialogue_manager.create_metrics(
                start_time=start_time,
                rows_processed=len(df),
                columns_affected=len(df.columns)
            )
            
            # Format result
            result_data = {
                "filter": f"{column} {operator} {value}",
                "rows_matched": len(result_df),
                "data": result_df.head(100).to_dict(orient='records')
            }
            
            response = self.dialogue_manager.create_success_response(
                result=result_data,
                explanation=f"Filter matched {len(result_df)} rows",
                metrics=metrics
            )
            
            return response.to_dict()
            
        except Exception as e:
            return self._handle_pandas_error(e, start_time, df, context=f"filter on column '{column}'")
    
    def _handle_sort(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle sort action"""
        columns = parameters.get('columns', [])
        ascending = parameters.get('ascending', True)
        
        if not columns:
            return self._error_response("'columns' parameter is required", start_time)
        
        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return self._handle_column_not_found(missing_cols[0], df, start_time)
        
        self.logger.info(f"[SORT] columns={columns}, ascending={ascending}")
        
        try:
            # Create query plan for sort
            query_plan = QueryPlan(
                operation='sort',
                conditions={
                    'columns': columns,
                    'ascending': ascending
                }
            )
            
            # Execute sort using query executor
            result = self.query_executor.execute_query(df, query_plan)
            
            if not result.success:
                return self._error_response(result.error or "Sort failed", start_time)
            
            result_df = result.data
            
            # Create metrics
            metrics = self.dialogue_manager.create_metrics(
                start_time=start_time,
                rows_processed=len(df),
                columns_affected=len(df.columns)
            )
            
            # Format result
            result_data = {
                "sorted_by": columns,
                "ascending": ascending,
                "rows": len(result_df),
                "data": result_df.head(100).to_dict(orient='records')
            }
            
            response = self.dialogue_manager.create_success_response(
                result=result_data,
                explanation=f"Sorted by {', '.join(columns)} ({'ascending' if ascending else 'descending'})",
                metrics=metrics
            )
            
            return response.to_dict()
            
        except Exception as e:
            return self._handle_pandas_error(e, start_time, df, context=f"sort by columns {columns}")
    
    def _handle_transform(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle transform action - apply transformation code"""
        code = parameters.get('code')
        
        if not code:
            return self._error_response("'code' parameter is required", start_time)
        
        self.logger.info(f"[TRANSFORM] code={code[:100]}")
        
        try:
            # Execute transformation
            local_vars = {"df": df.copy(), "pd": pd}
            exec(code, {"__builtins__": {}}, local_vars)
            result_df = local_vars.get("df", df)
            
            # Store modified dataframe
            self.dataframe_cache.store(thread_id, file_id, result_df, {})
            
            # Create metrics
            metrics = self.dialogue_manager.create_metrics(
                start_time=start_time,
                rows_processed=len(result_df),
                columns_affected=len(result_df.columns)
            )
            
            # Format result
            result = {
                "before_shape": df.shape,
                "after_shape": result_df.shape,
                "data": result_df.head(100).to_dict(orient='records')
            }
            
            response = self.dialogue_manager.create_success_response(
                result=result,
                explanation=f"Transformation applied: {df.shape} → {result_df.shape}",
                metrics=metrics
            )
            
            return response.to_dict()
            
        except Exception as e:
            return self._handle_pandas_error(e, start_time, df, context="transformation")
    
    def _handle_detect_anomalies(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle detect_anomalies action"""
        self.logger.info(f"[DETECT_ANOMALIES] file_id={file_id}")
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_all_anomalies(df)
        
        # Create metrics
        metrics = self.dialogue_manager.create_metrics(
            start_time=start_time,
            rows_processed=len(df),
            columns_affected=len(df.columns)
        )
        
        # Format result
        result = {
            "anomalies_found": len(anomalies),
            "anomalies": [
                {
                    "type": anomaly.type,
                    "columns": anomaly.columns,
                    "message": anomaly.message,
                    "severity": anomaly.severity,
                    "sample_values": anomaly.sample_values
                }
                for anomaly in anomalies
            ]
        }
        
        response = self.dialogue_manager.create_success_response(
            result=result,
            explanation=f"Found {len(anomalies)} anomalies",
            metrics=metrics
        )
        
        return response.to_dict()
    
    def _handle_parse(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Handle parse action - perform intelligent spreadsheet parsing.
        
        This integrates all parsing components to:
        1. Detect document sections and intentional gaps
        2. Identify and extract primary data tables
        3. Extract schema and metadata
        4. Build structured context for LLM consumption
        """
        self.logger.info(f"[PARSE] file_id={file_id}")
        
        try:
            # Step 1: Detect document sections
            sections = self.document_section_detector.detect_sections(df)
            
            # Step 2: Detect intentional gaps
            intentional_gaps = self.intentional_gap_detector.identify_intentional_gaps(df)
            
            # Step 3: Detect primary data table
            table_regions = self.table_detector.detect_all_tables(df)
            primary_table = self.table_detector.detect_primary_table(df) if table_regions else None
            
            # Step 4: Extract schema for primary table
            schema = None
            if primary_table:
                # Detect header row (returns row index, not TableRegion)
                header_row_idx = self.table_detector.detect_header_row(df, primary_table)
                
                # Extract schema using the table region and header row index
                schema = self.schema_extractor.extract_schema(df, primary_table, header_row_idx)
            
            # Step 5: Extract metadata
            metadata = self.metadata_extractor.extract_metadata(df, sections)
            
            # Step 6: Create ParsedSpreadsheet object
            tables = []
            if primary_table and schema:
                # Extract table dataframe
                table_df = df.iloc[
                    primary_table.start_row:primary_table.end_row + 1,
                    primary_table.start_col:primary_table.end_col + 1
                ].copy()
                tables.append((primary_table, table_df, schema))
            
            parsed_spreadsheet = ParsedSpreadsheet(
                file_id=file_id,
                sheet_name="Sheet1",  # Default sheet name
                document_type=DocumentType.DATA_TABLE,  # Default document type
                metadata=metadata,
                sections=sections,
                tables=tables,
                raw_df=df,
                intentional_gaps=intentional_gaps
            )
            
            # Step 7: Build structured context
            max_tokens = parameters.get('max_tokens', 4000)
            context = self.context_builder.build_structured_context(
                parsed=parsed_spreadsheet,
                max_tokens=max_tokens
            )
            
            # Create metrics
            metrics = self.dialogue_manager.create_metrics(
                start_time=start_time,
                rows_processed=len(df),
                columns_affected=len(df.columns)
            )
            
            # Format result
            result = {
                "file_id": file_id,
                "document_type": metadata.get('document_type', 'unknown'),
                "sections": [
                    {
                        "type": section.section_type,
                        "start_row": section.start_row,
                        "end_row": section.end_row,
                        "content_type": section.content_type
                    }
                    for section in sections
                ],
                "primary_table": {
                    "start_row": primary_table.start_row,
                    "end_row": primary_table.end_row,
                    "start_col": primary_table.start_col,
                    "end_col": primary_table.end_col,
                    "confidence": primary_table.confidence
                } if primary_table else None,
                "schema": {
                    "headers": schema.headers,
                    "dtypes": schema.dtypes,
                    "row_count": schema.row_count,
                    "col_count": schema.col_count
                } if schema else None,
                "metadata": metadata,
                "intentional_gaps": intentional_gaps,
                "structured_context": context
            }
            
            response = self.dialogue_manager.create_success_response(
                result=result,
                explanation=f"Parsed spreadsheet: found {len(sections)} sections, {len(table_regions)} tables",
                metrics=metrics
            )
            
            return response.to_dict()
            
        except Exception as e:
            return self._handle_pandas_error(e, start_time, df, context="parsing")
    
    def _handle_build_context(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Handle build_context action - build LLM context from parsed spreadsheet.
        
        This creates token-efficient context representations for LLM consumption
        while preserving document structure and preventing hallucination.
        """
        self.logger.info(f"[BUILD_CONTEXT] file_id={file_id}")
        
        try:
            # Get context format preference
            format_type = parameters.get('format', 'structured')  # 'structured', 'compact', 'full'
            max_tokens = parameters.get('max_tokens', 4000)
            
            # First parse the document if not already done
            sections = self.document_section_detector.detect_sections(df)
            intentional_gaps = self.intentional_gap_detector.identify_intentional_gaps(df)
            table_regions = self.table_detector.detect_all_tables(df)
            primary_table = self.table_detector.detect_primary_table(df) if table_regions else None
            
            # Extract schema and metadata
            schema = None
            if primary_table:
                header_row_idx = self.table_detector.detect_header_row(df, primary_table)
                schema = self.schema_extractor.extract_schema(df, primary_table, header_row_idx)
            
            metadata = self.metadata_extractor.extract_metadata(df, sections)
            
            # Create ParsedSpreadsheet object
            tables = []
            if primary_table and schema:
                table_df = df.iloc[
                    primary_table.start_row:primary_table.end_row + 1,
                    primary_table.start_col:primary_table.end_col + 1
                ].copy()
                tables.append((primary_table, table_df, schema))
            
            parsed_spreadsheet = ParsedSpreadsheet(
                file_id=file_id,
                sheet_name="Sheet1",
                document_type=DocumentType.DATA_TABLE,
                metadata=metadata,
                sections=sections,
                tables=tables,
                raw_df=df,
                intentional_gaps=intentional_gaps
            )
            
            # Build context based on format type
            if format_type == 'structured':
                context = self.context_builder.build_structured_context(
                    parsed=parsed_spreadsheet,
                    max_tokens=max_tokens
                )
            elif format_type == 'compact':
                context = self.context_builder.build_compact_context(
                    parsed=parsed_spreadsheet,
                    max_tokens=max_tokens
                )
            else:  # full
                context = self.context_builder.build_full_context(
                    parsed=parsed_spreadsheet
                )
            
            # Create metrics
            metrics = self.dialogue_manager.create_metrics(
                start_time=start_time,
                rows_processed=len(df),
                columns_affected=len(df.columns)
            )
            
            # Format result
            result = {
                "file_id": file_id,
                "format_type": format_type,
                "context": context,
                "token_estimate": len(str(context).split()) * 1.3,  # Rough token estimate
                "sections_count": len(sections),
                "tables_count": len(table_regions),
                "has_primary_table": primary_table is not None
            }
            
            response = self.dialogue_manager.create_success_response(
                result=result,
                explanation=f"Built {format_type} context with ~{result['token_estimate']:.0f} tokens",
                metrics=metrics
            )
            
            return response.to_dict()
            
        except Exception as e:
            return self._handle_pandas_error(e, start_time, df, context="context building")
    
    def _handle_prompt(
        self,
        prompt: str,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        parameters: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle natural language prompt"""
        self.logger.info(f"[PROMPT] prompt={prompt[:100]}")
        
        # For now, return error - this would integrate with LLM for complex queries
        return self._error_response(
            "Natural language prompts not yet implemented. Use specific actions instead.",
            start_time,
            error_details={"prompt": prompt[:200]}
        )
    
    def _handle_anomaly_resolution(
        self,
        df: pd.DataFrame,
        file_id: str,
        thread_id: str,
        anomaly: Any,
        user_choice: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Handle user's choice for anomaly resolution"""
        self.logger.info(f"[ANOMALY_RESOLUTION] choice={user_choice}")
        
        try:
            # Apply fix based on user choice
            fixed_df = self.anomaly_detector.apply_fix(df, anomaly, user_choice)
            
            # Store fixed dataframe
            self.dataframe_cache.store(thread_id, file_id, fixed_df, {})
            
            # Create metrics
            metrics = self.dialogue_manager.create_metrics(
                start_time=start_time,
                rows_processed=len(fixed_df),
                columns_affected=len(anomaly.columns)
            )
            
            # Format result
            result = {
                "anomaly_type": anomaly.type,
                "fix_applied": user_choice,
                "affected_columns": anomaly.columns,
                "before_shape": df.shape,
                "after_shape": fixed_df.shape
            }
            
            response = self.dialogue_manager.create_success_response(
                result=result,
                explanation=f"Applied fix '{user_choice}' to {len(anomaly.columns)} column(s)",
                metrics=metrics
            )
            
            return response.to_dict()
            
        except Exception as e:
            return self._handle_pandas_error(e, start_time, df, context="anomaly resolution")
    
    # ========================================================================
    # ERROR HANDLING (Task 10.2)
    # ========================================================================
    
    def _handle_pandas_error(
        self,
        error: Exception,
        start_time: float,
        df: pd.DataFrame,
        context: str = "operation"
    ) -> Dict[str, Any]:
        """
        Handle pandas exceptions with user-friendly messages.
        
        Args:
            error: The exception that occurred
            start_time: Operation start time
            df: DataFrame being operated on
            context: Description of the operation
        
        Returns:
            AgentResponse dictionary with ERROR status
            
        Requirements: 14.1
        """
        error_str = str(error)
        error_type = type(error).__name__
        
        self.logger.error(f"[PANDAS_ERROR] {error_type} during {context}: {error_str}")
        
        # Provide user-friendly error messages
        if "KeyError" in error_type:
            # Extract column name from error
            import re
            match = re.search(r"'([^']+)'", error_str)
            if match:
                column = match.group(1)
                return self._handle_column_not_found(column, df, start_time)
        
        if "TypeError" in error_type and "numeric" in error_str.lower():
            user_message = (
                f"Cannot perform {context} because the column contains non-numeric values. "
                f"Try converting the column to numeric first or filtering out non-numeric rows."
            )
        elif "ValueError" in error_type:
            user_message = f"Invalid value provided for {context}: {error_str}"
        elif "AttributeError" in error_type:
            user_message = f"Invalid operation for {context}: {error_str}"
        else:
            user_message = f"Error during {context}: {error_str}"
        
        return self._error_response(
            user_message,
            start_time,
            error_details={
                "error_type": error_type,
                "original_error": error_str,
                "context": context
            }
        )
    
    def _handle_column_not_found(
        self,
        column: str,
        df: pd.DataFrame,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Handle column not found error with fuzzy matching suggestions.
        
        Args:
            column: The column name that wasn't found
            df: DataFrame to search for similar columns
            start_time: Operation start time
        
        Returns:
            AgentResponse dictionary with ERROR status and suggestions
            
        Requirements: 14.2
        """
        # Use fuzzy matching to find similar column names
        similar_columns = get_close_matches(column, df.columns.tolist(), n=3, cutoff=0.6)
        
        error_message = f"Column '{column}' not found."
        
        if similar_columns:
            error_message += f" Did you mean: {', '.join(similar_columns)}?"
        else:
            error_message += f" Available columns: {', '.join(df.columns.tolist())}"
        
        self.logger.warning(f"[COLUMN_NOT_FOUND] {error_message}")
        
        return self._error_response(
            error_message,
            start_time,
            error_details={
                "requested_column": column,
                "available_columns": df.columns.tolist(),
                "suggestions": similar_columns
            }
        )
    
    def _load_dataframe(
        self,
        file_id: str,
        thread_id: str
    ) -> tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Load dataframe from cache or session.
        
        Args:
            file_id: File identifier
            thread_id: Thread identifier
        
        Returns:
            Tuple of (dataframe, error_response)
            If successful, returns (df, None)
            If failed, returns (None, error_response_dict)
            
        Requirements: 14.3
        """
        try:
            # Try to retrieve from cache
            df, metadata = self.dataframe_cache.retrieve(thread_id, file_id)
            
            if df is not None:
                self.logger.debug(f"[LOAD] Retrieved from cache: file_id={file_id}")
                return df, None
            
            # Try to load from session
            from agents.spreadsheet_agent.session import get_dataframe
            self.logger.info(f"[LOAD] Loading from session: file_id={file_id}")
            df = get_dataframe(file_id, thread_id)
            
            if df is None:
                error_response = self._error_response(
                    f"File {file_id} not found",
                    time.time(),
                    error_details={"file_id": file_id, "thread_id": thread_id}
                )
                return None, error_response
            
            # Store in cache
            self.dataframe_cache.store(thread_id, file_id, df, {"loaded_from": "session"})
            
            return df, None
            
        except Exception as e:
            self.logger.error(f"[LOAD] Error loading file {file_id}: {e}", exc_info=True)
            error_response = self._error_response(
                f"Failed to load file {file_id}: {str(e)}",
                time.time(),
                error_details={
                    "file_id": file_id,
                    "thread_id": thread_id,
                    "error_type": type(e).__name__
                }
            )
            return None, error_response
    
    def _error_response(
        self,
        error_message: str,
        start_time: float,
        error_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error_message: User-friendly error message
            start_time: Operation start time
            error_details: Optional detailed error information
        
        Returns:
            AgentResponse dictionary with ERROR status
        """
        metrics = self.dialogue_manager.create_metrics(start_time=start_time)
        
        response = self.dialogue_manager.create_error_response(
            error_message=error_message,
            error_details=error_details,
            metrics=metrics
        )
        
        return response.to_dict()


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create a global agent instance for use in endpoints
spreadsheet_agent = SpreadsheetAgent()
