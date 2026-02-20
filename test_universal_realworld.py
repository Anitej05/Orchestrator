"""
Universal Agent - Complex Real-World Test Suite
Tests the agent with diverse, challenging tasks across all capabilities.
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8070"

def test_health():
    """Check agent health."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health: {r.json()}")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_capabilities():
    """List agent capabilities."""
    try:
        r = requests.get(f"{BASE_URL}/capabilities", timeout=5)
        caps = r.json()
        print(f"\nCapabilities for {caps.get('agent_name', 'Unknown')}:")
        for cap in caps.get("capabilities", []):
            print(f"  - {cap.get('name', 'Unknown')}: {cap.get('description', '')}")
        return True
    except Exception as e:
        print(f"Capabilities check failed: {e}")
        return False

def execute_task(task_name: str, payload: dict, timeout: int = 120):
    """Send a task to the universal agent and get the response."""
    print(f"\n{'='*80}")
    print(f"TASK: {task_name}")
    print(f"{'='*80}")
    
    start = time.time()
    try:
        r = requests.post(
            f"{BASE_URL}/execute",
            json=payload,
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            print(f"\n[SUCCESS] Status: {result.get('status', 'unknown')} ({elapsed:.1f}s)")
            print(f"Summary: {result.get('summary', 'N/A')}")
            
            # Print result content
            res = result.get("result")
            if isinstance(res, dict):
                answer = res.get("result") or res.get("answer") or json.dumps(res, indent=2, default=str)
            elif isinstance(res, str):
                answer = res
            else:
                answer = str(res)
            
            # Truncate long responses for readability
            if len(answer) > 2000:
                print(f"\nResponse (first 2000 chars):\n{answer[:2000]}...")
            else:
                print(f"\nResponse:\n{answer}")
            
            return result
        else:
            print(f"\n[ERROR] HTTP {r.status_code}: {r.text[:500]}")
            return None
    except requests.Timeout:
        elapsed = time.time() - start
        print(f"\n[TIMEOUT] Request timed out after {elapsed:.1f}s")
        return None
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[ERROR] {e} (after {elapsed:.1f}s)")
        return None


def main():
    print("=" * 80)
    print("UNIVERSAL AGENT - REAL-WORLD COMPLEX TASK TESTING")
    print("=" * 80)
    
    # Pre-flight checks
    if not test_health():
        print("Agent is not healthy. Exiting.")
        sys.exit(1)
    
    test_capabilities()
    
    # ──────────────────────────────────────────────────────────────
    # TASK 1: Complex Multi-Step Analysis
    # ──────────────────────────────────────────────────────────────
    execute_task(
        "Task 1: System Design Analysis",
        {
            "prompt": (
                "Design a scalable microservices architecture for a real-time "
                "collaborative document editing platform similar to Google Docs. "
                "Include: (1) The core services needed (auth, document, presence, "
                "conflict resolution), (2) How you'd handle Operational Transform "
                "or CRDTs for concurrent edits, (3) WebSocket vs SSE for real-time "
                "updates, (4) Database choices for each service, and (5) A caching "
                "strategy for frequently accessed documents."
            )
        }
    )
    
    # ──────────────────────────────────────────────────────────────
    # TASK 2: Code Generation - Complex Algorithm
    # ──────────────────────────────────────────────────────────────
    execute_task(
        "Task 2: Code Generation - LRU Cache with TTL",
        {
            "prompt": (
                "Write a production-ready Python implementation of an LRU Cache with "
                "Time-To-Live (TTL) support. Requirements: "
                "1) O(1) get/put operations using OrderedDict, "
                "2) Configurable max capacity, "
                "3) Per-key TTL with automatic expiration, "
                "4) Thread-safe with proper locking, "
                "5) Metrics tracking (hit rate, eviction count), "
                "6) Include comprehensive unit tests."
            ),
            "action": "generate_code"
        }
    )
    
    # ──────────────────────────────────────────────────────────────
    # TASK 3: Research Task
    # ──────────────────────────────────────────────────────────────
    execute_task(
        "Task 3: Research - AI Agent Architectures",
        {
            "prompt": (
                "Research and compare the latest AI agent architectures: "
                "ReAct, Chain-of-Thought, Tree-of-Thought, and Graph-of-Thought. "
                "For each, explain: (1) How it works conceptually, "
                "(2) Strengths and weaknesses, (3) Best use cases, "
                "(4) Real-world implementations/papers. "
                "Then provide a recommendation for building a general-purpose "
                "AI coding assistant."
            ),
            "action": "research"
        }
    )
    
    # ──────────────────────────────────────────────────────────────
    # TASK 4: Problem Solving - Complex Optimization
    # ──────────────────────────────────────────────────────────────
    execute_task(
        "Task 4: Problem Solving - Task Scheduling",
        {
            "prompt": (
                "A company has 10 servers with different capacities (CPU, RAM, bandwidth). "
                "They need to schedule 50 containerized microservices, each with specific "
                "resource requirements and inter-service communication dependencies. "
                "Some services need to be co-located for low latency, while others must "
                "be separated for fault tolerance. Design an algorithm to optimally "
                "place these services. Consider: bin-packing constraints, affinity/anti-affinity "
                "rules, and the ability to handle server failures with minimal rebalancing. "
                "Provide the algorithm's time complexity and a practical Python implementation."
            ),
            "action": "solve_problem"
        }
    )
    
    # ──────────────────────────────────────────────────────────────
    # TASK 5: Creative + Technical Writing
    # ──────────────────────────────────────────────────────────────
    execute_task(
        "Task 5: Technical Blog Post",
        {
            "prompt": (
                "Write a compelling technical blog post titled: 'Building Fault-Tolerant "
                "Distributed Systems: Lessons from Running 100k Containers in Production'. "
                "Cover: real-world war stories of cascading failures, circuit breaker patterns, "
                "chaos engineering practices, and practical tips for building resilient systems. "
                "The tone should be engaging and accessible while still being technically deep. "
                "Include code examples where appropriate."
            ),
            "action": "creative_write"
        }
    )
    
    # ──────────────────────────────────────────────────────────────
    # TASK 6: Multi-step General Task (LLM planning)
    # ──────────────────────────────────────────────────────────────
    execute_task(
        "Task 6: Multi-Step Data Pipeline Design",
        {
            "prompt": (
                "I need to build a real-time data pipeline that: "
                "1) Ingests click-stream events from a website at 100k events/sec, "
                "2) Enriches events with user profile data from a PostgreSQL database, "
                "3) Performs real-time sessionization (group clicks into 30-min sessions), "
                "4) Computes running aggregations (page views per page, avg session duration), "
                "5) Writes results to both a real-time dashboard (via WebSocket) and a data lake "
                "(Parquet files in S3). "
                "Design this pipeline with specific technology choices, explain why you chose them, "
                "and provide a working proof-of-concept in Python using Kafka + Faust/Flink."
            )
        }
    )
    
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
