"""
Tool Router - Deterministic tool routing with validation.

This module provides explicit tool-first routing logic:
1. Check if a direct tool can handle the intent
2. Validate required parameters are available
3. Return routing decision with reasoning

Industry standard: Route 70%+ of simple queries through direct tools.
"""

import logging
from typing import Dict, Any, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger("AgentOrchestrator")


class ToolRoutingDecision(BaseModel):
    """Result of tool routing evaluation"""
    use_tool: bool = Field(..., description="Whether to use a direct tool")
    tool_name: Optional[str] = Field(None, description="Selected tool name")
    tool_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for tool")
    reasoning: str = Field(..., description="Why this decision was made")
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence")
    missing_params: Set[str] = Field(default_factory=set, description="Missing required parameters")


class ToolRouter:
    """Deterministic tool routing with validation"""
    
    def __init__(self, tool_registry):
        """
        Initialize router with tool registry.
        
        Args:
            tool_registry: Tool registry module for tool lookup
        """
        self.tool_registry = tool_registry
    
    def route(self, intent, context: Dict = None) -> ToolRoutingDecision:
        """
        Decide whether to route to a direct tool.
        
        Args:
            intent: Intent object from classifier
            context: Additional context (state, uploaded files, etc.)
            
        Returns:
            ToolRoutingDecision with routing choice and reasoning
        """
        context = context or {}
        
        # PRIORITY 1: Check if intent has explicit tool hint
        if intent.tool_hint:
            return self._validate_tool_hint(intent)
        
        # PRIORITY 2: Category-based routing
        if intent.category == "data_query":
            # Data queries should prefer tools
            return self._route_data_query(intent)
        
        if intent.category in ["document_task", "spreadsheet_task"]:
            # File tasks require agents (no direct tools)
            return ToolRoutingDecision(
                use_tool=False,
                reasoning=f"File tasks require specialized agents ({intent.category})",
                confidence=0.95
            )
        
        if intent.category == "web_navigation":
            # Web navigation requires browser agent
            return ToolRoutingDecision(
                use_tool=False,
                reasoning="Web navigation requires browser automation agent",
                confidence=0.90
            )
        
        # PRIORITY 3: Fallback - no clear tool match
        return ToolRoutingDecision(
            use_tool=False,
            reasoning="No direct tool available for this intent category",
            confidence=0.60
        )
    
    def _validate_tool_hint(self, intent) -> ToolRoutingDecision:
        """Validate that hinted tool exists and has required params"""
        tool_name = intent.tool_hint
        
        # Check if tool exists
        tool = self.tool_registry.get_tool_by_name(tool_name)
        if not tool:
            logger.warning(f"Tool hint '{tool_name}' not found in registry")
            return ToolRoutingDecision(
                use_tool=False,
                reasoning=f"Tool '{tool_name}' not found in registry",
                confidence=0.0
            )
        
        # Get required parameters
        required_params = self.tool_registry.get_required_params(tool_name)
        
        # Check if all required params are available in intent.entities
        # This now includes both re-extracted entities AND pre-extracted task.parameters
        available_params = set(intent.entities.keys())
        missing = set(required_params) - available_params
        
        if missing:
            # Try to infer missing params
            inferred = self._infer_missing_params(missing, intent)
            if inferred:
                intent.entities.update(inferred)
                missing = missing - set(inferred.keys())
        
        if missing:
            logger.warning(f"Tool '{tool_name}' missing required params: {missing}. Available: {available_params}")
            return ToolRoutingDecision(
                use_tool=False,
                tool_name=tool_name,
                reasoning=f"Missing required parameters: {', '.join(missing)}",
                confidence=0.0,
                missing_params=missing
            )
        
        # All params available - route to tool
        logger.info(f"✅ Tool '{tool_name}' validated with params: {list(available_params)}")
        return ToolRoutingDecision(
            use_tool=True,
            tool_name=tool_name,
            tool_params=intent.entities,
            reasoning=f"Tool '{tool_name}' matched with all required params",
            confidence=intent.confidence
        )
    
    def _route_data_query(self, intent) -> ToolRoutingDecision:
        """Route data queries to appropriate tools"""
        # Try keyword matching in tool registry
        query_keywords = self._extract_keywords(intent.entities.get("query", ""))
        
        if not query_keywords:
            return ToolRoutingDecision(
                use_tool=False,
                reasoning="No keywords extracted for tool matching",
                confidence=0.0
            )
        
        # Match against tool capabilities
        matches = self.tool_registry.match_by_keywords(query_keywords)
        
        if not matches:
            return ToolRoutingDecision(
                use_tool=False,
                reasoning="No tools matched the query keywords",
                confidence=0.0
            )
        
        # Get best match
        best_match = matches[0]  # Sorted by score
        
        if best_match["score"] < 0.6:
            return ToolRoutingDecision(
                use_tool=False,
                reasoning=f"Best tool match score too low: {best_match['score']:.2f}",
                confidence=best_match["score"]
            )
        
        # Validate params
        tool_name = best_match["tool_name"]
        required_params = self.tool_registry.get_required_params(tool_name)
        available_params = set(intent.entities.keys())
        missing = set(required_params) - available_params
        
        if missing:
            return ToolRoutingDecision(
                use_tool=False,
                tool_name=tool_name,
                reasoning=f"Tool matched but missing params: {', '.join(missing)}",
                confidence=best_match["score"],
                missing_params=missing
            )
        
        return ToolRoutingDecision(
            use_tool=True,
            tool_name=tool_name,
            tool_params=intent.entities,
            reasoning=f"Keyword-matched tool with score {best_match['score']:.2f}",
            confidence=best_match["score"]
        )
    
    def _infer_missing_params(self, missing: Set[str], intent) -> Dict[str, Any]:
        """Try to infer missing parameters from context"""
        inferred = {}
        
        # If query is missing, use the full prompt
        if "query" in missing:
            inferred["query"] = intent.entities.get("query") or ""
        
        # Add other inference rules here
        
        return inferred
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return set()
        
        # Simple keyword extraction (remove stopwords)
        stopwords = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                    "get", "fetch", "find", "show", "tell", "me", "about", "for"}
        
        words = text.lower().split()
        keywords = {w for w in words if w not in stopwords and len(w) > 2}
        
        return keywords


def create_tool_router():
    """Create tool router with tool registry"""
    from orchestrator.tool_registry import get_tool_registry
    
    tool_registry_instance = get_tool_registry()
    return ToolRouter(tool_registry_instance)


# Convenience function
def route_to_tool(intent, context: Dict = None) -> ToolRoutingDecision:
    """Route intent to tool or agent"""
    router = create_tool_router()
    return router.route(intent, context)
