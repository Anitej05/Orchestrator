import logging
import time
import psutil
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict

# Configure logger
logger = logging.getLogger("TelemetryService")

@dataclass
class RequestMetrics:
    total: int = 0
    successful: int = 0
    failed: int = 0
    
@dataclass
class PerformanceMetrics:
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    requests_completed: int = 0
    
@dataclass
class ErrorMetrics:
    total: int = 0
    planning_errors: int = 0
    execution_errors: int = 0
    agent_errors: int = 0
    by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

@dataclass
class AgentMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    by_agent: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

@dataclass
class ToolMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    by_tool: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

class TelemetryService:
    """
    Centralized service for collecting, aggregating, and reporting system metrics.
    Replaces global metrics dictionaries in the orchestrator.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.requests = RequestMetrics()
        self.performance = PerformanceMetrics()
        self.errors = ErrorMetrics()
        self.agents = AgentMetrics()
        self.tools = ToolMetrics()
        
    def log_request(self, success: bool, latency_ms: float = 0):
        """Log a completed orchestrator request."""
        self.requests.total += 1
        if success:
            self.requests.successful += 1
        else:
            self.requests.failed += 1
            
        if latency_ms > 0:
            self.performance.total_latency_ms += latency_ms
            self.performance.requests_completed += 1
            self.performance.avg_latency_ms = (
                self.performance.total_latency_ms / self.performance.requests_completed
            )

    def log_agent_call(self, agent_name: str, success: bool, duration_ms: float = 0):
        """Log an individual agent execution."""
        self.agents.total_calls += 1
        self.agents.by_agent[agent_name] += 1
        
        if success:
            self.agents.successful_calls += 1
        else:
            self.agents.failed_calls += 1

    def log_tool_call(self, tool_name: str, success: bool, duration_ms: float = 0):
        """Log a tool execution call."""
        self.tools.total_calls += 1
        self.tools.by_tool[tool_name] += 1
        
        if success:
            self.tools.successful_calls += 1
        else:
            self.tools.failed_calls += 1

    def log_error(self, category: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """
        Log an error occurrence.
        category: 'planning', 'execution', 'agent', or 'other'
        """
        self.errors.total += 1
        
        if category == 'planning':
            self.errors.planning_errors += 1
        elif category == 'execution':
            self.errors.execution_errors += 1
        elif category == 'agent':
            self.errors.agent_errors += 1
            
        # Track specific error types (simple string clustering)
        error_type = error_message.split(':')[0] if ':' in error_message else error_message[:50]
        self.errors.by_type[error_type] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive snapshot of current metrics."""
        uptime_seconds = time.time() - self.start_time
        
        # Calculate success rate
        total_requests = self.requests.total
        success_rate = (
            (self.requests.successful / total_requests * 100) 
            if total_requests > 0 else 0.0
        )
        
        # Resource usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            "uptime_seconds": uptime_seconds,
            "success_rate": success_rate,
            "requests": asdict(self.requests),
            "agents": {
                "total_calls": self.agents.total_calls, 
                "successful_calls": self.agents.successful_calls,
                "failed_calls": self.agents.failed_calls,
                "by_agent": dict(self.agents.by_agent)
            },
            "tools": {
                "total_calls": self.tools.total_calls,
                "successful_calls": self.tools.successful_calls,
                "failed_calls": self.tools.failed_calls,
                "by_tool": dict(self.tools.by_tool)
            },
            "performance": asdict(self.performance),
            "errors": {
                "total": self.errors.total,
                "planning_errors": self.errors.planning_errors,
                "execution_errors": self.errors.execution_errors,
                "agent_errors": self.errors.agent_errors,
                "by_type": dict(self.errors.by_type)
            },
            "resource": {
                "current_memory_mb": memory_mb
            }
        }

    def print_metrics_report(self, operation: str, success: bool):
        """Print a formatted report to the logs (replaces log_orchestrator_metrics)."""
        status_emoji = "✅" if success else "❌"
        
        logger.info("")
        logger.info(f"{status_emoji} TELEMETRY REPORT - {operation}")
        logger.info("")
        
        # Request Stats
        metrics = self.get_metrics()
        reqs = metrics["requests"]
        logger.info("Requests:")
        logger.info(f"  Total: {reqs['total']}")
        logger.info(f"  Successful: {reqs['successful']}")
        logger.info(f"  Failed: {reqs['failed']}")
        logger.info(f"  Success Rate: {metrics['success_rate']:.1f}%")
        
        # Agent Stats
        agents = metrics["agents"]
        logger.info("")
        logger.info("Agent Calls:")
        logger.info(f"  Total: {agents['total_calls']}")
        logger.info(f"  Successful: {agents['successful_calls']}")
        logger.info(f"  Failed: {agents['failed_calls']}")
        
        if agents['by_agent']:
            logger.info("")
            logger.info("Top Agents Used:")
            sorted_agents = sorted(
                agents['by_agent'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for agent_name, count in sorted_agents:
                logger.info(f"  {agent_name}: {count} calls")
                
        # Errors
        errs = metrics["errors"]
        if errs['total'] > 0:
            logger.info("")
            logger.info("Errors:")
            logger.info(f"  Total: {errs['total']}")
            logger.info(f"  Planning: {errs['planning_errors']}")
            logger.info(f"  Execution: {errs['execution_errors']}")
            logger.info(f"  Agent: {errs['agent_errors']}")
            
        # Resources
        logger.info("")
        logger.info("Resources:")
        logger.info(f"  Memory: {metrics['resource']['current_memory_mb']:.1f} MB")
        logger.info("")

# Global singleton
telemetry_service = TelemetryService()
