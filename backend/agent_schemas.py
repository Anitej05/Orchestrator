"""
Agent Schema Validation

Pydantic models for validating agent definitions against the standardized schema.
All agent JSON files must conform to these models before being synced to the database.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import re


# ============================================================================
# ENUMS
# ============================================================================

class AgentTypeEnum(str, Enum):
    """Supported agent types"""
    HTTP_REST = "http_rest"
    MCP_HTTP = "mcp_http"
    TOOL = "tool"


class AgentStatus(str, Enum):
    """Agent status values"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"


class CapabilityPriority(str, Enum):
    """Capability priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CredentialFieldType(str, Enum):
    """Credential field input types"""
    TEXT = "text"
    PASSWORD = "password"
    URL = "url"
    SELECT = "select"


class HttpMethod(str, Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class TransportType(str, Enum):
    """MCP transport types"""
    SSE = "sse"
    HTTP = "http"


# ============================================================================
# CAPABILITY MODELS
# ============================================================================

class Capability(BaseModel):
    """Individual capability definition"""
    id: str = Field(..., pattern=r"^[a-z0-9-]+$", description="Capability ID (kebab-case)")
    name: str = Field(..., min_length=1, description="Human-readable capability name")
    description: str = Field(..., min_length=1, description="What this capability does")
    keywords: List[str] = Field(..., min_length=2, description="Search keywords (min 2)")
    requires_permission: bool = Field(..., description="Does this need special permissions?")
    examples: List[str] = Field(..., min_length=1, description="Example queries (min 1)")
    related_endpoints: Optional[List[str]] = Field(None, description="Related endpoint paths")
    related_functions: Optional[List[str]] = Field(None, description="Related function names")

    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v):
        """Ensure keywords are non-empty and unique"""
        if len(v) < 2:
            raise ValueError("At least 2 keywords required")
        if len(set(v)) != len(v):
            raise ValueError("Keywords must be unique")
        if any(not k.strip() for k in v):
            raise ValueError("Keywords cannot be empty strings")
        return v

    @field_validator('examples')
    @classmethod
    def validate_examples(cls, v):
        """Ensure examples are non-empty"""
        if any(not e.strip() for e in v):
            raise ValueError("Examples cannot be empty strings")
        return v


class CapabilityCategory(BaseModel):
    """Category grouping related capabilities"""
    name: str = Field(..., min_length=1, description="Category name")
    description: str = Field(..., min_length=1, description="What this category covers")
    priority: CapabilityPriority = Field(..., description="Priority level")
    capabilities: List[Capability] = Field(..., min_length=1, description="Capabilities in this category")

    @field_validator('capabilities')
    @classmethod
    def validate_unique_capability_ids(cls, v):
        """Ensure capability IDs are unique within category"""
        ids = [cap.id for cap in v]
        if len(set(ids)) != len(ids):
            raise ValueError("Capability IDs must be unique within category")
        return v


class CapabilityStructure(BaseModel):
    """Complete capability structure for an agent"""
    categories: List[CapabilityCategory] = Field(..., min_length=1, description="Capability categories")
    all_keywords: List[str] = Field(..., min_length=1, description="Flat list of all keywords")

    @field_validator('categories')
    @classmethod
    def validate_unique_category_names(cls, v):
        """Ensure category names are unique"""
        names = [cat.name for cat in v]
        if len(set(names)) != len(names):
            raise ValueError("Category names must be unique")
        return v

    @model_validator(mode='after')
    def validate_all_keywords_match(self):
        """Ensure all_keywords contains all keywords from capabilities"""
        capability_keywords = set()
        for category in self.categories:
            for capability in category.capabilities:
                capability_keywords.update(capability.keywords)
        
        all_keywords_set = set(self.all_keywords)
        
        # Check if all_keywords has at least some keywords from capabilities
        # Don't enforce strict matching to allow backward compatibility
        if not capability_keywords.issubset(all_keywords_set | capability_keywords):
            # Just ensure all_keywords isn't empty
            if len(all_keywords_set) == 0:
                raise ValueError("all_keywords cannot be empty")
        
        return self


# ============================================================================
# CONNECTION CONFIG MODELS
# ============================================================================

class RetryConfig(BaseModel):
    """HTTP retry configuration"""
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    backoff_factor: float = Field(2.0, ge=1.0, description="Backoff multiplier")


class HttpRestConfig(BaseModel):
    """Configuration for HTTP REST agents"""
    base_url: str = Field(..., pattern=r"^https?://", description="Base URL for API")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Request timeout (seconds)")
    headers: Optional[Dict[str, str]] = Field(None, description="Default headers")
    retry_config: Optional[RetryConfig] = Field(None, description="Retry configuration")


class McpHttpConfig(BaseModel):
    """Configuration for MCP HTTP agents"""
    base_url: str = Field(..., pattern=r"^https?://", description="Base URL for MCP server")
    transport: TransportType = Field(..., description="Transport method")
    timeout: Optional[int] = Field(60, ge=1, le=300, description="Request timeout (seconds)")
    headers: Optional[Dict[str, str]] = Field(None, description="Default headers")


# ============================================================================
# ENDPOINT MODELS
# ============================================================================

class ParameterValidation(BaseModel):
    """Validation rules for parameters"""
    min: Optional[int] = Field(None, description="Minimum value/length")
    max: Optional[int] = Field(None, description="Maximum value/length")
    pattern: Optional[str] = Field(None, description="Regex pattern")
    enum: Optional[List[str]] = Field(None, description="Allowed values")


class Parameter(BaseModel):
    """Endpoint parameter definition"""
    name: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$", description="Parameter name (snake_case)")
    param_type: str = Field(..., description="Parameter type")
    required: bool = Field(..., description="Is this parameter required?")
    description: str = Field(..., min_length=1, description="Parameter description")
    default_value: Optional[Any] = Field(None, description="Default value")
    validation: Optional[ParameterValidation] = Field(None, description="Validation rules")

    @field_validator('param_type')
    @classmethod
    def validate_param_type(cls, v):
        """Validate parameter type"""
        valid_types = ['string', 'integer', 'number', 'boolean', 'array', 'object', 'file']
        if v not in valid_types:
            raise ValueError(f"param_type must be one of {valid_types}")
        return v


class Endpoint(BaseModel):
    """Agent endpoint definition"""
    endpoint: str = Field(..., description="Endpoint path (relative path starting with / or full URL)")
    http_method: HttpMethod = Field(..., description="HTTP method")
    description: str = Field(..., min_length=1, description="Endpoint description")
    request_format: Optional[str] = Field(None, description="Request format (json, form-data)")
    parameters: List[Parameter] = Field(default_factory=list, description="Endpoint parameters")
    response_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for response")

    @field_validator('request_format')
    @classmethod
    def validate_request_format(cls, v):
        """Validate request format if provided"""
        if v is not None and v not in ['json', 'form-data', 'form', 'multipart/form-data']:
            raise ValueError("request_format must be 'json', 'form-data', 'form', or 'multipart/form-data'")
        return v


# ============================================================================
# CREDENTIAL MODELS
# ============================================================================

class CredentialValidation(BaseModel):
    """Validation rules for credential fields"""
    pattern: Optional[str] = Field(None, description="Regex pattern")
    enum: Optional[List[str]] = Field(None, description="Allowed values (for select)")


class CredentialField(BaseModel):
    """Credential field definition"""
    name: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$", description="Field name (snake_case)")
    label: str = Field(..., min_length=1, description="Display label")
    type: CredentialFieldType = Field(..., description="Input field type")
    required: bool = Field(..., description="Is this field required?")
    description: str = Field(..., min_length=1, description="Help text for users")
    placeholder: Optional[str] = Field(None, description="Example value")
    validation: Optional[CredentialValidation] = Field(None, description="Validation rules")

    @model_validator(mode='after')
    def validate_select_has_enum(self):
        """Ensure select fields have enum validation"""
        if self.type == CredentialFieldType.SELECT:
            if not self.validation or not self.validation.enum:
                raise ValueError("Select fields must have validation.enum")
        return self


# ============================================================================
# TOOL MODELS
# ============================================================================

class ToolFunction(BaseModel):
    """Tool function definition"""
    name: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$", description="Function name (snake_case)")
    description: str = Field(..., min_length=1, description="What this function does")
    async_: bool = Field(False, alias="async", description="Is this an async function?")
    parameters_schema: Dict[str, Any] = Field(..., description="JSON schema for parameters")

    class Config:
        populate_by_name = True


class ToolRegistry(BaseModel):
    """Registry of tool functions for TOOL agents"""
    module: str = Field(..., pattern=r"^[a-z_][a-z0-9_.]*$", description="Python module path")
    functions: List[ToolFunction] = Field(..., min_length=1, description="Tool functions")

    @field_validator('functions')
    @classmethod
    def validate_unique_function_names(cls, v):
        """Ensure function names are unique"""
        names = [func.name for func in v]
        if len(set(names)) != len(names):
            raise ValueError("Function names must be unique")
        return v


# ============================================================================
# MAIN AGENT SCHEMA
# ============================================================================

class AgentSchema(BaseModel):
    """Complete agent schema definition"""
    # Core identity
    id: str = Field(..., pattern=r"^[a-z0-9_-]+$", description="Agent ID (kebab-case or snake_case)")
    owner_id: str = Field(..., min_length=1, description="Owner identifier")
    name: str = Field(..., min_length=1, description="Human-readable name")
    description: str = Field(..., min_length=1, description="Detailed description")
    agent_type: AgentTypeEnum = Field(..., description="Agent type")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version (X.Y.Z)")
    
    # Capabilities
    capabilities: CapabilityStructure = Field(..., description="Structured capabilities")
    
    # Status & pricing
    status: AgentStatus = Field(..., description="Agent status")
    price_per_call_usd: float = Field(..., ge=0, description="Cost per call (USD)")
    
    # Authentication
    requires_credentials: bool = Field(False, description="Does agent need credentials?")
    credential_fields: List[CredentialField] = Field(default_factory=list, description="Credential definitions")
    
    # Connection configuration (type-specific)
    connection_config: Optional[Any] = Field(None, description="Connection configuration")
    
    # Endpoints (for HTTP/MCP agents)
    endpoints: Optional[List[Endpoint]] = Field(None, description="Agent endpoints")
    
    # Tool functions (for TOOL agents)
    tool_functions: Optional[List[str]] = Field(None, description="Tool function names")
    tool_registry: Optional[ToolRegistry] = Field(None, description="Tool function registry")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    category: Optional[str] = Field(None, description="Primary category")
    icon: Optional[str] = Field(None, description="Icon identifier")
    documentation_url: Optional[str] = Field(None, pattern=r"^https?://", description="Documentation URL")
    
    # Signatures
    public_key_pem: str = Field(..., description="Public key for verification")

    @model_validator(mode='after')
    def validate_type_specific_fields(self):
        """Validate type-specific requirements"""
        agent_type = self.agent_type
        
        if agent_type == AgentTypeEnum.HTTP_REST:
            # HTTP_REST requires connection_config and endpoints
            if not self.connection_config:
                raise ValueError("HTTP_REST agents require connection_config")
            if not self.endpoints or len(self.endpoints) == 0:
                raise ValueError("HTTP_REST agents require at least one endpoint")
            if self.tool_functions:
                raise ValueError("HTTP_REST agents cannot have tool_functions")
            
            # Validate connection_config structure
            HttpRestConfig.model_validate(self.connection_config)
            
        elif agent_type == AgentTypeEnum.MCP_HTTP:
            # MCP_HTTP requires connection_config and endpoints
            if not self.connection_config:
                raise ValueError("MCP_HTTP agents require connection_config")
            if not self.endpoints or len(self.endpoints) == 0:
                raise ValueError("MCP_HTTP agents require at least one endpoint")
            if self.tool_functions:
                raise ValueError("MCP_HTTP agents cannot have tool_functions")
            
            # Validate connection_config structure
            McpHttpConfig.model_validate(self.connection_config)
            
        elif agent_type == AgentTypeEnum.TOOL:
            # TOOL requires tool_functions, no connection_config or endpoints
            if not self.tool_functions or len(self.tool_functions) == 0:
                raise ValueError("TOOL agents require at least one tool_function")
            if self.connection_config:
                raise ValueError("TOOL agents cannot have connection_config")
            if self.endpoints:
                raise ValueError("TOOL agents cannot have endpoints")
        
        return self

    @model_validator(mode='after')
    def validate_credential_fields(self):
        """Validate credential fields if required"""
        if self.requires_credentials and len(self.credential_fields) == 0:
            raise ValueError("If requires_credentials=True, must provide credential_fields")
        if not self.requires_credentials and len(self.credential_fields) > 0:
            raise ValueError("If requires_credentials=False, cannot have credential_fields")
        return self

    @model_validator(mode='after')
    def validate_related_references(self):
        """Validate that related_endpoints and related_functions reference existing items"""
        # Build set of available endpoints
        endpoint_paths = set()
        if self.endpoints:
            endpoint_paths = {ep.endpoint for ep in self.endpoints}
        
        # Build set of available functions
        function_names = set()
        if self.tool_functions:
            function_names = set(self.tool_functions)
        
        # Check all capabilities
        for category in self.capabilities.categories:
            for capability in category.capabilities:
                # Check related_endpoints
                if capability.related_endpoints:
                    for endpoint in capability.related_endpoints:
                        if endpoint not in endpoint_paths:
                            raise ValueError(
                                f"Capability '{capability.id}' references non-existent endpoint '{endpoint}'"
                            )
                
                # Check related_functions
                if capability.related_functions:
                    for func in capability.related_functions:
                        if func not in function_names:
                            raise ValueError(
                                f"Capability '{capability.id}' references non-existent function '{func}'"
                            )
        
        return self


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_agent_schema(agent_data: dict) -> tuple[bool, Optional[str], Optional[AgentSchema]]:
    """
    Validate agent data against schema.
    
    Args:
        agent_data: Dictionary containing agent definition
        
    Returns:
        Tuple of (is_valid, error_message, validated_schema)
    """
    try:
        validated = AgentSchema.model_validate(agent_data)
        return True, None, validated
    except Exception as e:
        return False, str(e), None


def validate_agent_file(file_path: str) -> tuple[bool, Optional[str], Optional[AgentSchema]]:
    """
    Validate agent JSON file against schema.
    
    Args:
        file_path: Path to agent JSON file
        
    Returns:
        Tuple of (is_valid, error_message, validated_schema)
    """
    import json
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            agent_data = json.load(f)
        return validate_agent_schema(agent_data)
    except FileNotFoundError:
        return False, f"File not found: {file_path}", None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None
    except Exception as e:
        return False, f"Validation error: {e}", None


# ============================================================================
# MIGRATION HELPERS
# ============================================================================

def migrate_legacy_capabilities(legacy_capabilities: List[str]) -> CapabilityStructure:
    """
    Migrate legacy flat capability list to structured format.
    
    Args:
        legacy_capabilities: Old flat list of capability strings
        
    Returns:
        Structured CapabilityStructure
    """
    # Group capabilities into a single "General" category
    capabilities = []
    for cap in legacy_capabilities:
        capability_id = cap.lower().replace(' ', '-')
        capabilities.append(Capability(
            id=capability_id,
            name=cap.title(),
            description=f"Capability: {cap}",
            keywords=[cap, capability_id],
            requires_permission=False,
            examples=[f"Use {cap}"],
        ))
    
    category = CapabilityCategory(
        name="General",
        description="General agent capabilities",
        priority=CapabilityPriority.MEDIUM,
        capabilities=capabilities
    )
    
    return CapabilityStructure(
        categories=[category],
        all_keywords=legacy_capabilities
    )
