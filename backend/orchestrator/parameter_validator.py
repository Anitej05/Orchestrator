"""
Parameter Validator - Unified parameter extraction, validation, and merging.

This module provides a centralized way to handle parameters flowing through the orchestration:
1. Extract parameters from Task objects (pre-extracted by LLM)
2. Extract parameters from Intent classification (pattern-matched)
3. Merge parameters with priority: Task.parameters > Intent.entities
4. Validate parameters match tool signatures
5. Provide detailed error feedback for missing/invalid parameters
"""

import logging
from typing import Dict, Any, Optional, Set, List
from pydantic import BaseModel, Field
from schemas import Task

logger = logging.getLogger("AgentOrchestrator")


class ParameterSchema(BaseModel):
    """Schema for tool parameters (defines what a tool accepts)"""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (str, int, float, bool, list, dict)")
    required: bool = Field(default=True, description="Whether parameter is required")
    description: Optional[str] = Field(None, description="Parameter description")
    default: Optional[Any] = Field(None, description="Default value if not provided")
    choices: Optional[List[Any]] = Field(None, description="Allowed values for enum-like params")


class ParameterContext(BaseModel):
    """Container for all parameters flowing through the orchestration"""
    task_params: Dict[str, Any] = Field(default_factory=dict, description="Pre-extracted from Task")
    intent_params: Dict[str, Any] = Field(default_factory=dict, description="Extracted from Intent")
    merged_params: Dict[str, Any] = Field(default_factory=dict, description="Merged with priority")
    missing_params: Set[str] = Field(default_factory=set, description="Required params not provided")
    extra_params: Set[str] = Field(default_factory=set, description="Params not in schema")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    is_valid: bool = Field(default=False, description="Whether parameters are valid for tool")


class ParameterValidator:
    """Centralized parameter validation and merging"""
    
    def __init__(self, tool_registry):
        """
        Initialize validator with tool registry.
        
        Args:
            tool_registry: Tool registry for schema lookup
        """
        self.tool_registry = tool_registry
    
    def validate_and_merge(
        self,
        task: Task,
        tool_name: str,
        intent_params: Dict[str, Any]
    ) -> ParameterContext:
        """
        Extract, merge, and validate parameters for a task/tool combination.
        
        Priority order:
        1. Task.parameters (from LLM parse - most reliable)
        2. Intent.entities (from pattern matching - re-extracted)
        
        Args:
            task: Task object with pre-extracted parameters
            tool_name: Name of tool to validate against
            intent_params: Parameters from intent classification
            
        Returns:
            ParameterContext with merged params, validation status, and errors
        """
        context = ParameterContext(
            task_params=dict(task.parameters) if task.parameters else {},
            intent_params=dict(intent_params) if intent_params else {}
        )
        
        # Merge: Task params take priority over intent params
        context.merged_params = {**context.intent_params, **context.task_params}
        
        logger.info(f"🔀 [MERGE] Task params ({len(context.task_params)}): {context.task_params}")
        logger.info(f"🔀 [MERGE] Intent params ({len(context.intent_params)}): {context.intent_params}")
        logger.info(f"🔀 [MERGE] Merged params ({len(context.merged_params)}): {context.merged_params}")
        
        # Get schema for tool - Build from tool registry methods
        required_params = self.tool_registry.get_required_params(tool_name)
        tool = self.tool_registry.get_tool_by_name(tool_name)
        
        # Build schema from tool
        schema = []
        if tool and hasattr(tool, "args_schema"):
            args_schema = getattr(tool, "args_schema", None)
            if args_schema and hasattr(args_schema, "model_json_schema"):
                try:
                    tool_schema_json = args_schema.model_json_schema()
                    properties = tool_schema_json.get("properties", {})
                    for param_name, param_info in properties.items():
                        schema.append(ParameterSchema(
                            name=param_name,
                            type=param_info.get("type", "string"),
                            required=param_name in required_params,
                            default=param_info.get("default"),
                            choices=param_info.get("enum")
                        ))
                except Exception as e:
                    logger.warning(f"Failed to extract schema from tool '{tool_name}': {e}")
        
        # Fallback: build minimal schema from required_params
        if not schema and required_params:
            for param_name in required_params:
                schema.append(ParameterSchema(
                    name=param_name,
                    type="string",
                    required=True
                ))
        
        if not schema:
            logger.warning(f"No schema found for tool '{tool_name}' - skipping validation")
            context.is_valid = True  # Allow execution without schema
            return context
        
        # Validate
        self._validate_parameters(context, schema, tool_name)
        
        return context
    
    def _validate_parameters(
        self,
        context: ParameterContext,
        schema: List[ParameterSchema],
        tool_name: str
    ) -> None:
        """
        Validate merged parameters against tool schema.
        
        Args:
            context: ParameterContext to validate and update
            schema: Tool parameter schema
            tool_name: Tool name for logging
        """
        param_names = {p.name for p in schema}
        required_params = {p.name for p in schema if p.required}
        
        # Check for missing required parameters
        missing = required_params - set(context.merged_params.keys())
        if missing:
            context.missing_params = missing
            error_msg = f"Missing required parameters for {tool_name}: {', '.join(missing)}"
            context.validation_errors.append(error_msg)
            logger.warning(f"❌ {error_msg}")
        
        # Check for extra/unknown parameters
        extra = set(context.merged_params.keys()) - param_names
        if extra:
            context.extra_params = extra
            logger.info(f"⚠️ Tool '{tool_name}' received extra parameters (will be filtered): {extra}")
        
        # Type validation and coercion
        for param_schema in schema:
            param_name = param_schema.name
            
            if param_name not in context.merged_params:
                # Use default if available
                if param_schema.default is not None:
                    context.merged_params[param_name] = param_schema.default
                    logger.info(f"📝 Using default for '{param_name}': {param_schema.default}")
                continue
            
            value = context.merged_params[param_name]
            
            # Validate type
            if not self._validate_type(value, param_schema.type):
                error_msg = f"Invalid type for '{param_name}': expected {param_schema.type}, got {type(value).__name__}"
                context.validation_errors.append(error_msg)
                logger.warning(f"❌ {error_msg}")
            
            # Validate choices
            if param_schema.choices and value not in param_schema.choices:
                error_msg = f"Invalid value for '{param_name}': {value} not in {param_schema.choices}"
                context.validation_errors.append(error_msg)
                logger.warning(f"❌ {error_msg}")
        
        # Set validation status
        context.is_valid = len(context.validation_errors) == 0
        
        if context.is_valid:
            logger.info(f"✅ Parameters validated successfully for tool '{tool_name}'")
        else:
            logger.error(f"❌ Parameter validation failed for '{tool_name}': {context.validation_errors}")
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate that value matches expected type.
        
        Args:
            value: Value to validate
            expected_type: Expected type name (str, int, float, bool, list, dict)
            
        Returns:
            True if valid, False otherwise
        """
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),  # int is valid for float
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        
        expected = type_map.get(expected_type)
        if expected is None:
            logger.warning(f"Unknown type '{expected_type}'")
            return True  # Unknown type = assume valid
        
        return isinstance(value, expected)
    
    def get_valid_params(self, context: ParameterContext, schema: List[ParameterSchema]) -> Dict[str, Any]:
        """
        Extract only valid parameters for tool execution (filter out extras).
        
        Args:
            context: Validated ParameterContext
            schema: Tool parameter schema
            
        Returns:
            Dictionary with only valid parameters
        """
        param_names = {p.name for p in schema}
        valid_params = {
            k: v for k, v in context.merged_params.items()
            if k in param_names
        }
        
        logger.info(f"🔧 Filtered params for tool execution: {list(valid_params.keys())}")
        return valid_params
