---
id: universal_agent
name: Universal Agent
description: A general-purpose agent capable of handling any arbitrary task through LLM reasoning, code execution, and tool use. Spawns on-demand for tasks not covered by specialized agents.
version: 1.0.0
port: 8070
type: agent
category: general

capabilities:
  - name: general_task_execution
    description: Execute any arbitrary task using LLM reasoning, planning, and step-by-step execution
    
  - name: code_generation
    description: Write and execute Python code to solve problems
    
  - name: analysis
    description: Analyze data, text, or situations and provide insights
    
  - name: research
    description: Research topics and compile comprehensive information
    
  - name: creative_writing
    description: Write creative content like stories, poems, scripts
    
  - name: problem_solving
    description: Break down complex problems and solve them systematically

parameters:
  - name: prompt
    type: string
    required: true
    description: The task to be executed by the universal agent
    
  - name: context
    type: object
    required: false
    description: Additional context or files for the task

examples:
  - prompt: "Write a Python script to analyze this CSV file and find trends"
  - prompt: "Research the history of quantum computing and summarize key milestones"
  - prompt: "Create a marketing strategy for a new eco-friendly water bottle"
  - prompt: "Debug this code and explain what's wrong with it"
  - prompt: "Draft an email to my team about the project deadline extension"

use_when: |
  - Task doesn't fit specialized agents (browser, spreadsheet, mail, document, zoho_books)
  - Creative or analytical tasks requiring reasoning
  - Tasks requiring code generation or problem-solving
  - Research and information synthesis
  - Writing and content creation
  - Debugging or explaining code
  - General questions or tasks that need step-by-step execution

not_for: |
  - Web browsing or scraping (use Browser Agent)
  - Excel/CSV data processing (use Spreadsheet Agent)
  - Email sending (use Mail Agent)
  - Document processing (use Document Agent)
  - Accounting/invoicing (use Zoho Books Agent)
  - Simple tool calls (use direct tools instead)

lifecycle: on_demand
auto_terminate: 300

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
