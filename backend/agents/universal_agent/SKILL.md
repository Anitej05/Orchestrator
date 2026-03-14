---
id: universal_agent
name: Universal Agent
port: 8100
version: 1.0.0
description: >
  General-purpose agent for arbitrary tasks using LLM reasoning,
  code execution, and tool use. Fallback when no specialized agent matches.
model: ollama/minimax-m2.5:cloud
context_strategy: standard
requires_auth: false
triggers:
  - general task
  - research
  - analyze
  - creative writing
  - summarize
  - plan
  - brainstorm
capabilities:
  - llm_reasoning
  - code_execution
  - tool_usage
  - synthesis
  - planning
not_for: []
---

# Universal Agent

A flexible, general-purpose agent that can handle any arbitrary task through:
1. **LLM Planning**: Break down complex tasks into manageable steps
2. **Code Execution**: Write and run Python code when needed
3. **Tool Usage**: Utilize available tools when appropriate
4. **Reasoning**: Apply logical reasoning and analysis
5. **Synthesis**: Compile information into coherent responses

## Spawning Behavior

This agent spawns **on-demand**:
- Starts only when a general task is requested
- Automatically terminates after 5 minutes of idle time
- Can handle multiple sequential tasks in one session
