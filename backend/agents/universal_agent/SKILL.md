---
id: universal_agent
name: universal_agent
port: 8100
version: 1.0.0
description: A general-purpose agent for arbitrary tasks using LLM reasoning, code execution, and tool use.
---

# Universal Agent

A flexible, general-purpose agent that can handle any arbitrary task through:
1. **LLM Planning**: Break down complex tasks into manageable steps
2. **Code Execution**: Write and run Python code when needed
3. **Tool Usage**: Utilize available tools when appropriate
4. **Reasoning**: Apply logical reasoning and analysis
5. **Synthesis**: Compile information into coherent responses

## Architecture

The Universal Agent follows the BaseAgent pattern with:
- Capability-based task handling
- LLM-driven planning and execution
- Error recovery and retry mechanisms
- Built-in telemetry and logging

## Spawning Behavior

This agent spawns **on-demand**:
- Starts only when a general task is requested
- Automatically terminates after 5 minutes of idle time
- Can handle multiple sequential tasks in one session

## Example Workflows

### Example 1: Data Analysis
```
User: "Analyze this sales data and tell me the top 3 products"
↓
Universal Agent spawned
↓
1. Load and examine data
2. Perform analysis
3. Generate insights
4. Return results
↓
Agent terminated (after idle timeout)
```

### Example 2: Research Task
```
User: "Research renewable energy trends in 2024"
↓
Universal Agent spawned
↓
1. Plan research approach
2. Use web search tool
3. Synthesize information
4. Compile comprehensive report
↓
Agent terminated
```

### Example 3: Creative Writing
```
User: "Write a short story about AI becoming sentient"
↓
Universal Agent spawned
↓
1. Plan story structure
2. Generate content iteratively
3. Review and refine
4. Return final story
↓
Agent terminated
```
