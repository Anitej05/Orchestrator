"""
Agent Schema Migration Script

This script helps migrate agent JSON files from the old flat structure
to the new standardized schema structure.

Usage:
    python migrate_agent_schema.py <agent_json_file>
    python migrate_agent_schema.py --all  # Migrate all agents
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_schemas import (
    AgentSchema,
    CapabilityStructure,
    CapabilityCategory,
    Capability,
    CapabilityPriority,
    validate_agent_schema,
    validate_agent_file
)


def detect_capability_categories(capabilities: List[str]) -> Dict[str, List[str]]:
    """
    Automatically detect and group capabilities into categories based on keywords.
    
    Args:
        capabilities: List of capability strings
        
    Returns:
        Dictionary mapping category names to lists of capabilities
    """
    categories = {
        "Email Management": [],
        "Document Operations": [],
        "Data Analysis": [],
        "Web Automation": [],
        "File Operations": [],
        "Communication": [],
        "Search & Discovery": [],
        "Content Creation": [],
        "Integration": [],
        "General": []
    }
    
    # Keywords that indicate category
    category_keywords = {
        "Email Management": ["email", "mail", "inbox", "send", "compose", "message"],
        "Document Operations": ["document", "doc", "docx", "pdf", "word", "edit", "format"],
        "Data Analysis": ["analyze", "analysis", "summarize", "extract", "rag", "question"],
        "Web Automation": ["browser", "web", "automation", "browse", "navigate", "click"],
        "File Operations": ["file", "upload", "download", "save", "delete", "attachment"],
        "Communication": ["slack", "discord", "chat", "notify", "webhook"],
        "Search & Discovery": ["search", "find", "lookup", "query", "discover"],
        "Content Creation": ["create", "generate", "write", "compose", "build"],
        "Integration": ["api", "integration", "connect", "sync", "zoho", "composio"]
    }
    
    for cap in capabilities:
        cap_lower = cap.lower()
        categorized = False
        
        # Check each category's keywords
        for category, keywords in category_keywords.items():
            if any(keyword in cap_lower for keyword in keywords):
                categories[category].append(cap)
                categorized = True
                break
        
        # If not categorized, put in General
        if not categorized:
            categories["General"].append(cap)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def create_capability_from_string(cap_string: str, related_endpoints: List[str] = None) -> Capability:
    """
    Create a Capability object from a capability string.
    
    Args:
        cap_string: Capability description string
        related_endpoints: Optional list of related endpoint paths
        
    Returns:
        Capability object
    """
    # Generate ID
    cap_id = cap_string.lower().replace(' ', '-').replace('_', '-')
    
    # Generate name (title case)
    cap_name = cap_string.replace('_', ' ').title()
    
    # Generate keywords (split by spaces/underscores and add original)
    keywords = [cap_string]
    keywords.extend(cap_string.replace('_', ' ').split())
    keywords = list(set(k.lower() for k in keywords if k))[:5]  # Limit to 5 unique
    
    # Add more keywords if needed
    while len(keywords) < 2:
        keywords.append(cap_id)
    
    # Generate examples
    examples = [
        f"Use {cap_string}",
        f"Help me {cap_string}",
    ]
    
    return Capability(
        id=cap_id,
        name=cap_name,
        description=f"Capability to {cap_string}",
        keywords=keywords,
        requires_permission=any(word in cap_string.lower() for word in ['send', 'create', 'delete', 'modify', 'write']),
        examples=examples,
        related_endpoints=related_endpoints,
        related_functions=None
    )


def map_capabilities_to_endpoints(capabilities: List[str], endpoints: List[Dict]) -> Dict[str, List[str]]:
    """
    Map capabilities to their related endpoints based on keywords.
    
    Args:
        capabilities: List of capability strings
        endpoints: List of endpoint definitions
        
    Returns:
        Dictionary mapping capability strings to endpoint paths
    """
    mapping = {}
    
    for cap in capabilities:
        cap_lower = cap.lower()
        related = []
        
        for endpoint in endpoints:
            endpoint_desc = endpoint.get('description', '').lower()
            endpoint_path = endpoint.get('endpoint', '').lower()
            
            # Check if capability keywords appear in endpoint description or path
            cap_words = cap_lower.replace('_', ' ').split()
            if any(word in endpoint_desc or word in endpoint_path for word in cap_words):
                related.append(endpoint['endpoint'])
        
        mapping[cap] = related
    
    return mapping


def migrate_agent_json(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate an agent JSON from old format to new standardized format.
    
    Args:
        old_data: Original agent data
        
    Returns:
        Migrated agent data conforming to new schema
    """
    print(f"Migrating agent: {old_data.get('id', 'unknown')}")
    
    # Start with base structure
    new_data = {
        "id": old_data["id"],
        "owner_id": old_data.get("owner_id", "orbimesh-vendor"),
        "name": old_data["name"],
        "description": old_data["description"],
        "agent_type": old_data.get("agent_type", "http_rest"),
        "version": old_data.get("version", "1.0.0"),
        "status": old_data.get("status", "active"),
        "price_per_call_usd": old_data.get("price_per_call_usd", 0.01),
        "requires_credentials": old_data.get("requires_credentials", False),
        "credential_fields": old_data.get("credential_fields", []),
        "connection_config": old_data.get("connection_config"),
        "endpoints": old_data.get("endpoints"),
        "tool_functions": old_data.get("tool_functions"),
        "tool_registry": old_data.get("tool_registry"),
        "tags": old_data.get("tags", []),
        "category": old_data.get("category"),
        "icon": old_data.get("icon"),
        "documentation_url": old_data.get("documentation_url"),
        "public_key_pem": old_data.get("public_key_pem", "-----BEGIN PUBLIC KEY-----\nYOUR_PUBLIC_KEY_HERE\n-----END PUBLIC KEY-----")
    }
    
    # Handle capabilities
    old_capabilities = old_data.get("capabilities", [])
    
    # Check if already in new format
    if isinstance(old_capabilities, dict) and "categories" in old_capabilities:
        print("  ✓ Capabilities already in new format")
        new_data["capabilities"] = old_capabilities
    else:
        print(f"  → Migrating {len(old_capabilities)} capabilities to structured format")
        
        # Detect categories
        categorized_caps = detect_capability_categories(old_capabilities)
        print(f"  → Detected {len(categorized_caps)} categories")
        
        # Map capabilities to endpoints
        cap_endpoint_mapping = {}
        if new_data.get("endpoints"):
            cap_endpoint_mapping = map_capabilities_to_endpoints(old_capabilities, new_data["endpoints"])
        
        # Build structured capabilities
        categories = []
        all_keywords = list(old_capabilities)  # Keep original for backward compatibility
        
        for cat_name, caps in categorized_caps.items():
            print(f"    - {cat_name}: {len(caps)} capabilities")
            
            # Determine priority
            priority = CapabilityPriority.MEDIUM
            if cat_name in ["Email Management", "Document Operations", "Web Automation"]:
                priority = CapabilityPriority.HIGH
            elif cat_name == "General":
                priority = CapabilityPriority.LOW
            
            # Create capability objects
            capability_objects = []
            for cap in caps:
                related_endpoints = cap_endpoint_mapping.get(cap)
                capability_obj = create_capability_from_string(cap, related_endpoints)
                capability_objects.append(capability_obj.model_dump())
            
            categories.append({
                "name": cat_name,
                "description": f"{cat_name} capabilities",
                "priority": priority.value,
                "capabilities": capability_objects
            })
        
        new_data["capabilities"] = {
            "categories": categories,
            "all_keywords": all_keywords
        }
    
    return new_data


def migrate_file(input_path: Path, output_path: Path = None, validate: bool = True):
    """
    Migrate a single agent JSON file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (if None, overwrites input)
        validate: Whether to validate against schema after migration
    """
    if output_path is None:
        output_path = input_path
    
    print(f"\n{'='*60}")
    print(f"Migrating: {input_path.name}")
    print(f"{'='*60}")
    
    try:
        # Load old data
        with open(input_path, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        
        # Migrate
        new_data = migrate_agent_json(old_data)
        
        # Validate if requested
        if validate:
            print("\n  Validating against schema...")
            is_valid, error, validated = validate_agent_schema(new_data)
            
            if not is_valid:
                print(f"  ✗ Validation failed: {error}")
                print("\n  Saving anyway (with warnings)...")
            else:
                print("  ✓ Validation successful!")
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✓ Saved to: {output_path}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def migrate_all_agents(agent_dir: Path):
    """
    Migrate all agent JSON files in a directory.
    
    Args:
        agent_dir: Directory containing agent JSON files
    """
    json_files = list(agent_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {agent_dir}")
        return
    
    print(f"\nFound {len(json_files)} agent files to migrate")
    
    for json_file in json_files:
        migrate_file(json_file, validate=True)
    
    print("\n" + "="*60)
    print("Migration complete!")
    print("="*60)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python migrate_agent_schema.py <agent_json_file>")
        print("  python migrate_agent_schema.py --all")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == "--all":
        # Migrate all agents
        agent_dir = Path(__file__).parent / "Agent_entries"
        migrate_all_agents(agent_dir)
    else:
        # Migrate single file
        input_path = Path(arg)
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            sys.exit(1)
        
        migrate_file(input_path, validate=True)


if __name__ == "__main__":
    main()
