"""
Unified metrics display utilities for all agents.
Provides consistent, debug-friendly formatting for execution and session-level metrics.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def display_execution_metrics(metrics: Dict[str, Any], agent_name: str = "Agent", 
                            operation_name: str = "Operation", success: bool = True):
    """
    Display execution-level metrics for a single operation.
    Logs to terminal with structured formatting for easy debugging.
    
    Args:
        metrics: Dict with latency_ms, tokens_input, tokens_output, llm_calls, cache_hit_rate, etc.
        agent_name: Name of the agent (e.g., "Spreadsheet", "Document", "Browser")
        operation_name: Name of the operation (e.g., "nl_query", "analyze", "extract")
        success: Whether the operation succeeded
    """
    status_emoji = "✅" if success else "❌"
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    logger.info("")
    logger.info("━" * 80)
    logger.info(f"{status_emoji} [{agent_name}] {operation_name.upper()} - EXECUTION METRICS [{timestamp}]")
    logger.info("━" * 80)
    
    # Performance section
    if metrics.get('latency_ms') is not None:
        logger.info(f"")
        logger.info(f"⏱️  PERFORMANCE:")
        latency = metrics.get('latency_ms', 0)
        logger.info(f"    Total Latency:        {latency:.1f} ms")
        
        if metrics.get('rag_retrieval_ms'):
            logger.info(f"    RAG Retrieval:        {metrics['rag_retrieval_ms']:.1f} ms")
        
        if metrics.get('llm_call_ms'):
            logger.info(f"    LLM Processing:       {metrics['llm_call_ms']:.1f} ms")
    
    # Token usage section
    if metrics.get('tokens_input') is not None or metrics.get('tokens_output') is not None:
        logger.info(f"")
        logger.info(f"💬 TOKEN USAGE:")
        input_tokens = metrics.get('tokens_input', 0)
        output_tokens = metrics.get('tokens_output', 0)
        logger.info(f"    Input Tokens:         {input_tokens}")
        logger.info(f"    Output Tokens:        {output_tokens}")
        logger.info(f"    Total Tokens:         {input_tokens + output_tokens}")
    
    # LLM calls section
    if metrics.get('llm_calls') is not None:
        logger.info(f"")
        logger.info(f"🤖 LLM CALLS:")
        logger.info(f"    Total Calls:          {metrics['llm_calls']}")
    
    # RAG section (document agent specific)
    if metrics.get('chunks_retrieved') is not None and metrics['chunks_retrieved'] > 0:
        logger.info(f"")
        logger.info(f"📚 RAG RETRIEVAL:")
        logger.info(f"    Chunks Retrieved:     {metrics['chunks_retrieved']}")
        if metrics.get('avg_chunks_per_query'):
            logger.info(f"    Avg per Query:        {metrics['avg_chunks_per_query']:.1f}")
    
    # Cache section
    if metrics.get('cache_hit_rate') is not None:
        logger.info(f"")
        logger.info(f"💾 CACHE:")
        cache_rate = metrics.get('cache_hit_rate', 0)
        cache_status = "✅ HIT" if metrics.get('cache_hit') else "⚠️  MISS"
        logger.info(f"    Status:               {cache_status}")
        logger.info(f"    Hit Rate:             {cache_rate:.1f}%")
    
    # Retry section (spreadsheet agent specific)
    if metrics.get('retry_success_rate') is not None:
        logger.info(f"")
        logger.info(f"🔄 RETRY METRICS:")
        logger.info(f"    Success Rate:         {metrics['retry_success_rate']:.1f}%")
    
    # Resource section
    if metrics.get('memory_used_mb') is not None or metrics.get('peak_memory_mb') is not None:
        logger.info(f"")
        logger.info(f"💻 RESOURCES:")
        if metrics.get('memory_used_mb') is not None:
            logger.info(f"    Memory Used:          {metrics['memory_used_mb']:.1f} MB")
        if metrics.get('peak_memory_mb') is not None:
            logger.info(f"    Peak Memory:          {metrics['peak_memory_mb']:.1f} MB")
    
    # Error section (only if there are errors)
    if metrics.get('error') or not success:
        logger.info(f"")
        logger.info(f"⚠️  ERROR:")
        error_msg = metrics.get('error', 'Operation failed without specific error message')
        logger.info(f"    {error_msg}")
    
    logger.info("━" * 80)
    logger.info("")


def display_session_metrics(session_metrics: Dict[str, Any], agent_name: str = "Agent"):
    """
    Display session-level metrics aggregated across multiple operations.
    Shows cumulative stats, success rates, provider breakdown, etc.
    
    Args:
        session_metrics: Dict with queries, llm_calls, cache, performance, resource, etc.
        agent_name: Name of the agent
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    logger.info("")
    logger.info("═" * 80)
    logger.info(f"📊 [{agent_name}] SESSION-LEVEL METRICS [{timestamp}]")
    logger.info("═" * 80)
    
    # Queries section (spreadsheet agent)
    if 'queries' in session_metrics:
        queries = session_metrics['queries']
        total = queries.get('total', 0)
        successful = queries.get('successful', 0)
        failed = queries.get('failed', 0)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        logger.info(f"")
        logger.info(f"📝 QUERIES:")
        logger.info(f"    Total:                {total}")
        logger.info(f"    Successful:           {successful}")
        logger.info(f"    Failed:               {failed}")
        logger.info(f"    Success Rate:         {success_rate:.1f}%")
    
    # API calls section (document agent)
    if 'api_calls' in session_metrics:
        api_calls = session_metrics['api_calls']
        if any(api_calls.values()):
            logger.info(f"")
            logger.info(f"📡 API CALLS:")
            for api_type, count in api_calls.items():
                if count > 0:
                    logger.info(f"    {api_type.capitalize():20s} {count}")
    
    # Performance section
    if 'performance' in session_metrics:
        perf = session_metrics['performance']
        logger.info(f"")
        logger.info(f"⏱️  PERFORMANCE:")
        if perf.get('total_latency_ms'):
            logger.info(f"    Total Latency:        {perf['total_latency_ms']:.1f} ms")
        if perf.get('avg_latency_ms'):
            logger.info(f"    Avg per Request:      {perf['avg_latency_ms']:.1f} ms")
        if perf.get('rag_retrieval_ms'):
            logger.info(f"    RAG Retrieval:        {perf['rag_retrieval_ms']:.1f} ms")
        if perf.get('llm_call_ms'):
            logger.info(f"    LLM Processing:       {perf['llm_call_ms']:.1f} ms")
        if perf.get('requests_completed'):
            logger.info(f"    Requests Completed:   {perf['requests_completed']}")
    
    # LLM calls section
    if 'llm_calls' in session_metrics:
        llm_calls = session_metrics['llm_calls']
        total = llm_calls.get('total', 0)
        if total > 0:
            logger.info(f"")
            logger.info(f"🤖 LLM CALLS BREAKDOWN:")
            logger.info(f"    Total:                {total}")
            
            # Show provider breakdown if available
            if 'by_provider' in llm_calls:
                provider_stats = llm_calls['by_provider']
                for provider, count in provider_stats.items():
                    if count > 0:
                        logger.info(f"    {provider.capitalize():18s} {count}")
            else:
                # Show by operation type if available
                for op_type, count in llm_calls.items():
                    if op_type != 'total' and count > 0:
                        logger.info(f"    {op_type.capitalize():18s} {count}")
    
    # Tokens section (spreadsheet agent)
    if 'tokens' in session_metrics:
        tokens = session_metrics['tokens']
        input_total = tokens.get('input_total', 0)
        output_total = tokens.get('output_total', 0)
        if input_total > 0 or output_total > 0:
            logger.info(f"")
            logger.info(f"💬 TOKEN USAGE (SESSION TOTAL):")
            logger.info(f"    Input Tokens:         {input_total}")
            logger.info(f"    Output Tokens:        {output_total}")
            logger.info(f"    Total Tokens:         {input_total + output_total}")
    
    # Cache section
    if 'cache' in session_metrics:
        cache = session_metrics['cache']
        hits = cache.get('hits', 0)
        misses = cache.get('misses', 0)
        total_cache = hits + misses
        cache_rate = (hits / total_cache * 100) if total_cache > 0 else 0
        
        if total_cache > 0:
            logger.info(f"")
            logger.info(f"💾 CACHE:")
            logger.info(f"    Hits:                 {hits}")
            logger.info(f"    Misses:               {misses}")
            logger.info(f"    Hit Rate:             {cache_rate:.1f}%")
    
    # Retry section (spreadsheet agent)
    if 'retry' in session_metrics:
        retry = session_metrics['retry']
        total_retries = retry.get('total_retries', 0)
        successful_retries = retry.get('successful_retries', 0)
        success_rate = (successful_retries / total_retries * 100) if total_retries > 0 else 0
        
        if total_retries > 0:
            logger.info(f"")
            logger.info(f"🔄 RETRY METRICS:")
            logger.info(f"    Total Retries:        {total_retries}")
            logger.info(f"    Successful:           {successful_retries}")
            logger.info(f"    Success Rate:         {success_rate:.1f}%")
    
    # RAG section (document agent)
    if 'rag' in session_metrics:
        rag = session_metrics['rag']
        chunks_total = rag.get('chunks_retrieved_total', 0)
        if chunks_total > 0:
            logger.info(f"")
            logger.info(f"📚 RAG RETRIEVAL (SESSION):")
            logger.info(f"    Chunks Retrieved:     {chunks_total}")
            if rag.get('avg_chunks_per_query'):
                logger.info(f"    Avg per Query:        {rag['avg_chunks_per_query']:.1f}")
            if rag.get('vector_stores_loaded'):
                logger.info(f"    Stores Loaded:        {rag['vector_stores_loaded']}")
            if rag.get('retrieval_failures'):
                logger.info(f"    Retrieval Failures:   {rag['retrieval_failures']}")
    
    # Processing section (document agent)
    if 'processing' in session_metrics:
        processing = session_metrics['processing']
        if processing.get('files_processed', 0) > 0 or processing.get('batch_operations', 0) > 0:
            logger.info(f"")
            logger.info(f"⚙️  PROCESSING:")
            if processing.get('files_processed'):
                logger.info(f"    Files Processed:      {processing['files_processed']}")
            if processing.get('batch_operations'):
                logger.info(f"    Batch Operations:     {processing['batch_operations']}")
    
    # Errors section
    errors_exist = False
    error_info = {}
    
    if 'errors' in session_metrics:
        errors = session_metrics['errors']
        total_errors = errors.get('total', 0)
        if total_errors > 0:
            errors_exist = True
            error_info = errors
    
    if errors_exist:
        logger.info(f"")
        logger.info(f"⚠️  ERRORS:")
        logger.info(f"    Total Errors:         {error_info.get('total', 0)}")
        for error_type, count in error_info.items():
            if error_type != 'total' and count > 0:
                logger.info(f"    {error_type.replace('_', ' ').title():17s} {count}")
    
    # Resource section
    if 'resource' in session_metrics:
        resource = session_metrics['resource']
        if resource.get('peak_memory_mb') or resource.get('current_memory_mb'):
            logger.info(f"")
            logger.info(f"💻 RESOURCES:")
            if resource.get('current_memory_mb'):
                logger.info(f"    Current Memory:       {resource['current_memory_mb']:.1f} MB")
            if resource.get('peak_memory_mb'):
                logger.info(f"    Peak Memory:          {resource['peak_memory_mb']:.1f} MB")
            if resource.get('avg_cpu_percent'):
                logger.info(f"    Avg CPU:              {resource['avg_cpu_percent']:.1f}%")
    
    logger.info("═" * 80)
    logger.info("")


def format_metric_value(value: Any, metric_type: str = "numeric") -> str:
    """
    Format a metric value for display with appropriate precision.
    
    Args:
        value: The metric value
        metric_type: Type of metric (numeric, percentage, time, memory, tokens)
    
    Returns:
        Formatted string representation
    """
    if value is None:
        return "N/A"
    
    if metric_type == "percentage":
        return f"{value:.1f}%"
    elif metric_type == "time":
        if value < 1:
            return f"{value*1000:.0f}μs"
        elif value < 1000:
            return f"{value:.1f}ms"
        else:
            return f"{value/1000:.2f}s"
    elif metric_type == "memory":
        if value < 1:
            return f"{value*1024:.1f}KB"
        else:
            return f"{value:.1f}MB"
    elif metric_type == "tokens":
        if value < 1000:
            return f"{value}"
        else:
            return f"{value/1000:.1f}K"
    else:  # numeric
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
