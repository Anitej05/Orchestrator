"""
Artifact Management System - Usage Examples

This file demonstrates how to use the artifact management system
in various scenarios within the orchestrator.
"""

import asyncio
import json
from typing import Dict, Any

# Import artifact services
from services.artifact_service import (
    ArtifactStore,
    ArtifactType,
    ArtifactPriority,
    ArtifactReference,
    store_artifact,
    retrieve_artifact,
    compress_for_context,
    get_thread_context
)

from services.context_optimizer import (
    optimize_context,
    OptimizedContext
)

from services.conversation_artifact_integration import (
    save_conversation,
    load_conversation,
    get_optimized_llm_context
)


# ============================================================================
# EXAMPLE 1: Storing Large Agent Results
# ============================================================================

def example_store_agent_result():
    """
    Example: Store a large browser automation result as an artifact.
    
    Instead of keeping the full result in context, we store it as an artifact
    and use a lightweight reference.
    """
    print("\n=== Example 1: Storing Large Agent Results ===\n")
    
    # Simulated large browser automation result
    browser_result = {
        "task": "Navigate to example.com and extract content",
        "status": "success",
        "steps": [
            {"action": "navigate", "url": "https://example.com", "duration": 1.2},
            {"action": "wait", "selector": "body", "duration": 0.5},
            {"action": "extract", "content": "Example Domain..." * 100},  # Large content
        ],
        "screenshots": [
            {"step": 1, "base64": "iVBORw0KGgo..." * 1000},  # Large base64
            {"step": 2, "base64": "iVBORw0KGgo..." * 1000},
        ],
        "extracted_data": {
            "title": "Example Domain",
            "links": ["https://www.iana.org/domains/example"],
            "text_content": "This domain is for use in illustrative examples..." * 50
        }
    }
    
    # Store as artifact
    metadata = store_artifact(
        content=browser_result,
        name="browser_result_example_com",
        artifact_type=ArtifactType.RESULT,
        thread_id="example-thread-123",
        description="Browser automation result for example.com",
        tags=["browser", "automation", "example.com"],
        priority=ArtifactPriority.MEDIUM
    )
    
    print(f"Stored artifact: {metadata.id}")
    print(f"Size: {metadata.size_bytes} bytes")
    print(f"Compressed: {metadata.compressed}")
    
    # Instead of including full result in context, use reference
    context_reference = f"[ARTIFACT:{metadata.id}] Browser result for example.com - {metadata.description}"
    print(f"\nContext reference (use this in LLM context):")
    print(context_reference)
    
    # Later, retrieve full content when needed
    artifact = retrieve_artifact(metadata.id)
    if artifact:
        print(f"\nRetrieved artifact with {len(artifact.content['steps'])} steps")
    
    return metadata.id


# ============================================================================
# EXAMPLE 2: Compressing Canvas Content
# ============================================================================

def example_compress_canvas():
    """
    Example: Compress large canvas HTML content.
    
    Canvas content (HTML dashboards, visualizations) can be very large.
    We store it as an artifact and keep only a reference in context.
    """
    print("\n=== Example 2: Compressing Canvas Content ===\n")
    
    # Simulated large canvas HTML
    canvas_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard</title>
        <style>
            /* Lots of CSS... */
            .dashboard { display: grid; }
            .chart { width: 100%; height: 300px; }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="chart" id="chart1">
                <!-- Chart data... -->
            </div>
            <!-- More charts and content... -->
        </div>
        <script>
            // Lots of JavaScript...
            const data = [/* large dataset */];
        </script>
    </body>
    </html>
    """ * 20  # Make it large
    
    thread_id = "canvas-example-456"
    
    # Compress for context
    result = compress_for_context(
        content=canvas_html,
        name="dashboard_canvas",
        thread_id=thread_id,
        artifact_type=ArtifactType.CANVAS
    )
    
    if isinstance(result, ArtifactReference):
        print(f"Canvas compressed to artifact: {result.id}")
        print(f"Original size: {len(canvas_html)} chars")
        print(f"Reference size: {result.size_bytes} bytes")
        print(f"\nContext string:")
        print(result.to_context_string())
    else:
        print("Canvas was small enough to keep inline")
    
    return result


# ============================================================================
# EXAMPLE 3: Optimizing Orchestrator State
# ============================================================================

def example_optimize_state():
    """
    Example: Optimize a full orchestrator state for LLM context.
    
    The optimizer prioritizes important information and compresses
    or excludes less important content to fit within token limits.
    """
    print("\n=== Example 3: Optimizing Orchestrator State ===\n")
    
    # Simulated orchestrator state with various fields
    state = {
        "original_prompt": "Search for the latest AI news and create a summary dashboard",
        "thread_id": "optimize-example-789",
        "status": "executing",
        "pending_user_input": False,
        "question_for_user": None,
        "planning_mode": False,
        
        # Messages (can be long)
        "messages": [
            {"type": "user", "content": "Search for the latest AI news"},
            {"type": "assistant", "content": "I'll search for AI news..." * 10},
            {"type": "user", "content": "Now create a dashboard"},
            {"type": "assistant", "content": "Creating dashboard..." * 10},
            {"type": "user", "content": "Add more charts"},
            {"type": "assistant", "content": "Adding charts..." * 10},
        ],
        
        # Parsed tasks
        "parsed_tasks": [
            {"task_name": "search_news", "task_description": "Search for AI news articles"},
            {"task_name": "create_dashboard", "task_description": "Create visualization dashboard"},
        ],
        
        # Completed tasks (can be large with results)
        "completed_tasks": [
            {
                "task_name": "search_news",
                "result": {
                    "articles": [
                        {"title": f"AI Article {i}", "content": "Content..." * 50}
                        for i in range(10)
                    ]
                }
            }
        ],
        
        # Task-agent pairs
        "task_agent_pairs": [
            {
                "task_name": "search_news",
                "primary": {
                    "id": "groq_search_agent",
                    "name": "Groq Search Agent",
                    "description": "Searches the web...",
                    "capabilities": ["web search", "summarization"] * 10,
                }
            }
        ],
        
        # Canvas content (large HTML)
        "canvas_content": "<html>Dashboard content...</html>" * 100,
        "canvas_type": "html",
        "has_canvas": True,
    }
    
    # Optimize for 4000 tokens
    result = optimize_context(
        state=state,
        thread_id="optimize-example-789",
        max_tokens=4000
    )
    
    print(f"Optimization Results:")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Tokens saved: {result.tokens_saved}")
    print(f"  Compression ratio: {result.compression_ratio:.2%}")
    print(f"  Included blocks: {len(result.included_blocks)}")
    print(f"  Excluded blocks: {len(result.excluded_blocks)}")
    print(f"  Artifact references: {len(result.artifact_references)}")
    
    print(f"\nOptimized context preview (first 500 chars):")
    print(result.user_context[:500] + "...")
    
    return result


# ============================================================================
# EXAMPLE 4: Saving/Loading Conversations with Artifacts
# ============================================================================

def example_conversation_with_artifacts():
    """
    Example: Save and load a conversation with automatic artifact compression.
    
    Large fields are automatically compressed to artifacts when saving,
    and expanded when loading.
    """
    print("\n=== Example 4: Conversation with Artifacts ===\n")
    
    thread_id = "conversation-example-101"
    
    # Simulated conversation data
    conversation = {
        "thread_id": thread_id,
        "status": "completed",
        "messages": [
            {"type": "user", "content": "Analyze this data", "timestamp": 1234567890},
            {"type": "assistant", "content": "Analysis complete..." * 100, "timestamp": 1234567900},
        ],
        "final_response": "Here's the analysis summary...",
        "canvas_content": "<html>Large visualization...</html>" * 50,
        "canvas_type": "html",
        "has_canvas": True,
        "completed_tasks": [
            {"task_name": "analyze", "result": {"data": list(range(1000))}}
        ],
        "metadata": {
            "original_prompt": "Analyze this data",
            "currentStage": "completed"
        }
    }
    
    # Save with compression
    print("Saving conversation with artifact compression...")
    saved = save_conversation(
        conversation_data=conversation,
        thread_id=thread_id,
        compress=True
    )
    
    # Check what was compressed
    if "_artifact_refs" in saved:
        print(f"\nCompressed fields:")
        for field, ref in saved["_artifact_refs"].items():
            print(f"  - {field}: artifact {ref['artifact_id']} ({ref['size_bytes']} bytes)")
    
    # Load with expansion
    print("\nLoading conversation with artifact expansion...")
    loaded = load_conversation(thread_id=thread_id, expand=True)
    
    if loaded:
        print(f"Loaded conversation with {len(loaded.get('messages', []))} messages")
        print(f"Canvas content restored: {len(loaded.get('canvas_content', ''))} chars")
    
    return loaded


# ============================================================================
# EXAMPLE 5: Getting Thread Context
# ============================================================================

def example_thread_context():
    """
    Example: Get optimized context for all artifacts in a thread.
    
    This is useful when you need to provide context about what
    artifacts are available without loading their full content.
    """
    print("\n=== Example 5: Thread Context ===\n")
    
    thread_id = "context-example-202"
    
    # First, create some artifacts for this thread
    store_artifact(
        content={"data": list(range(100))},
        name="dataset_1",
        artifact_type=ArtifactType.DATA,
        thread_id=thread_id,
        description="First dataset with 100 items"
    )
    
    store_artifact(
        content="def process_data(): pass" * 50,
        name="processing_code",
        artifact_type=ArtifactType.CODE,
        thread_id=thread_id,
        description="Data processing code"
    )
    
    store_artifact(
        content={"status": "success", "output": "Processed data..."},
        name="processing_result",
        artifact_type=ArtifactType.RESULT,
        thread_id=thread_id,
        description="Result of data processing"
    )
    
    # Get thread context
    context = get_thread_context(thread_id)
    
    print(f"Thread {thread_id} context:")
    print(f"  Artifact count: {context['artifact_count']}")
    print(f"  Tokens saved: {context['total_tokens_saved']}")
    print(f"\nContext string for LLM:")
    print(context['context_string'])
    
    return context


# ============================================================================
# MAIN
# ============================================================================

def run_all_examples():
    """Run all examples"""
    print("=" * 60)
    print("ARTIFACT MANAGEMENT SYSTEM - USAGE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_store_agent_result()
    example_compress_canvas()
    example_optimize_state()
    example_conversation_with_artifacts()
    example_thread_context()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
