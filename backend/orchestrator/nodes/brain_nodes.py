
import logging
import json
import traceback
from typing import Dict, Any, List, Optional, Union
from langchain_core.messages import SystemMessage, HumanMessage

from pydantic import BaseModel, Field

from orchestrator.state import State
from schemas import TaskItem, TaskStatus, TaskPriority
from services.inference_service import inference_service, InferencePriority


from services.terminal_service import terminal_service
from services.agent_registry_service import agent_registry
from services.tool_registry_service import tool_registry
from orchestrator.content_orchestrator import get_optimized_llm_context, hooks
from services.code_sandbox_service import code_sandbox
from services.telemetry_service import telemetry_service
import httpx
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Update Schema to include file path awareness
class TaskUpdate(BaseModel):
    """Structure for the Brain's decision to update the To-Do list."""
    thought: str = Field(..., description="Reasoning for the changes")
    new_tasks: List[TaskItem] = Field(default_factory=list, description="New tasks to add")
    completed_task_id: Optional[str] = Field(None, description="ID of the task to mark as completed")
    failed_task_id: Optional[str] = Field(None, description="ID of the task to mark as failed")
    error_note: Optional[str] = Field(None, description="Note on why a task failed")
    reorder_tasks: Optional[List[str]] = Field(None, description="List of task IDs in new execution order")
    next_task_id: Optional[str] = Field(None, description="Explicitly select the next task ID to run")
    is_finished: bool = Field(False, description="True if the entire workflow is done")
    # New: Persistent memory update
    memory_update: Optional[Dict[str, Any]] = Field(None, description="Key-value pairs to store in persistent memory")
    # New: Suggest a command, agent, or tool
    suggested_command: Optional[str] = Field(None, description="Shell command, or 'AGENT: <name> <task>', or 'TOOL: <name> <args>'")

async def manage_todo_list(state: State, config: Optional[Dict] = None) -> Dict:
    """
    The Brain Node.
    """
    try:
        todo_list = state.get("todo_list", [])
        memory = state.get("memory", {})
        iteration = state.get("iteration_count", 0)
        
        if not todo_list and state.get("original_prompt"):
            return _initialize_todo_list(state)
        
        pending_tasks = [t for t in todo_list if t['status'] == TaskStatus.PENDING]
        current_task = next((t for t in todo_list if t['status'] == TaskStatus.IN_PROGRESS), None)
        completed_tasks = [t for t in todo_list if t['status'] == TaskStatus.COMPLETED]
        failed_tasks = [t for t in todo_list if t['status'] == TaskStatus.FAILED]
        
        # Get Available Agents & Tools
        active_agents = agent_registry.list_active_agents()
        agent_list_str = "\n".join([f"- {a['name']}: {a['description']} (Capabilities: {a.get('capabilities')})" for a in active_agents])
        
        tool_list = tool_registry.list_tools()
        tool_list_str = "\n".join([f"- {t['name']}: {t['description']}" for t in tool_list])

        # Build context
        # Build optimized context using CMS
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        optimized_res = get_optimized_llm_context(state, thread_id)
        # BUG FIX: Key is "context", not "context_string"
        history_str = optimized_res.get("context", "No historical context available.")
        
        # Build list preview (just status and description, no results)
        list_preview = ""
        for t in todo_list:
            list_preview += f"- [{t['status'].upper()}] {t['description']} (ID: {t['id']})\n"

        prompt = f"""
        You are the Brain of an autonomous agent. 
        Your goal is to achieve the Current Objective by managing a To-Do list.
        
        CURRENT OBJECTIVE: {state.get("original_prompt")}
        
        PERSISTENT MEMORY:
        {json.dumps(memory, indent=2, default=str)}
        
        RECENT ACTION HISTORY & RESULTS (Managed by CMS):
        {history_str}
        
        FULL TO-DO LIST STATUS:
        {list_preview}
        
        CRITICAL INSTRUCTIONS:
        1. STRATEGY: If a task fails, try a different approach. Store key facts in 'memory_update'.
        2. CONTEXT: Check the 'HISTORY' above for results of previous steps.
        3. FINISHING: If objective is met, set 'is_finished' to True.
        
        RESOURCES:
        - TERMINAL: Use for file ops and scripts.
        - AGENTS: {agent_list_str}
        - TOOLS: {tool_list_str}

        Decide the next step.
        - If a task fits a Specialized Agent (e.g. "Analyze spreadsheet"), delegate it!
        - If a task needs a Tool (e.g. "Scan image"), use it.
        - If a task is file ops, use Terminal.
        - Otherwise, generic Python logic.
        
        CRITICAL: 
        1. If the Last Task Result is insufficient, you MUST create a NEW TASK associated with the next step (e.g. "Read file", "Search detailed query").
        2. If all tasks are completed and the Current Objective is met, you MUST set 'is_finished' to True.
        
        Return JSON matching TaskUpdate schema.
        Input for 'suggested_command' MUST follow these formats:
        - Shell: "TERM: ls -la"
        - Agent: "AGENT: SpreadsheetAgent Analyze data.csv"
        - Tool: 'TOOL: get_wikipedia_summary {{"title": "Super Bowl LVIII"}}'
        (ALWAYS use JSON for TOOL arguments if they are complex keys!)
        """
        
        # Use generate_structured for robust JSON parsing
        try:
            print(f"DEBUG: Calling Brain Inference... Prompt len: {len(prompt)}")
            update = await inference_service.generate_structured(
                messages=[HumanMessage(content=prompt)],
                schema=TaskUpdate,
                priority=InferencePriority.SPEED,
                temperature=0.2
            )
            print(f"DEBUG: Brain Inference Complete. Update: {update}")
        except Exception as e:
            print(f"DEBUG: Brain Inference Failed: {e}")
            logger.error(f"Brain LLM failed: {e}")
            return {"error": f"Brain failed: {str(e)}"}
        
        # Apply updates
        new_todo_list = list(todo_list)
        
        # Handle completion (Brain signaling a task is done)
        if update.completed_task_id:
            for t in new_todo_list:
                if t['id'] == update.completed_task_id:
                    t['status'] = TaskStatus.COMPLETED
                    t['completed_at'] = time.time()
                    
        # Update memory
        if update.memory_update:
            memory.update(update.memory_update)

        # Add tasks (with simple duplicate prevention)
        for task in update.new_tasks:
            t_dict = task.dict() if hasattr(task, 'dict') else task
            
            # Check if this task already exists (description match)
            is_dup = any(
                t['description'].strip().lower() == t_dict['description'].strip().lower() 
                for t in new_todo_list
                if t['status'] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.IN_PROGRESS]
            )
            
            if not is_dup:
                new_todo_list.append(t_dict)
            else:
                logger.info(f"Brain tried to add duplicate task: {t_dict['description']}. Ignoring.")
            
        # Select next
        if update.next_task_id:
            next_task_id = update.next_task_id
        else:
            # Default to first pending task
            next_pending = next((t for t in new_todo_list if t['status'] == TaskStatus.PENDING), None)
            next_task_id = next_pending['id'] if next_pending else None
        
        # If Brain suggested a command/agent/tool, attach it
        if next_task_id and update.suggested_command:
            for t in new_todo_list:
                if t['id'] == next_task_id:
                    # Determine prefix if not present (Brain might forget 'TERM:')
                    cmd = update.suggested_command
                    # Simple heuristic normalization
                    if not (cmd.startswith("TERM:") or cmd.startswith("AGENT:") or cmd.startswith("TOOL:")):
                         # Default to TERM if it looks like shell, else just pass it
                         cmd = f"TERM:{cmd}" 
                    t['code_snippet'] = cmd
        
        # Activate task
        if next_task_id:
             for t in new_todo_list:
                if t['id'] == next_task_id:
                    t['status'] = TaskStatus.IN_PROGRESS
                    
        return {
            "todo_list": new_todo_list,
            "memory": memory,
            "current_task_id": next_task_id,
            "iteration_count": iteration + 1,
            "final_response": "Done" if update.is_finished else None,
            "error": None # Clear previous error if any
        }

    except Exception as e:
        logger.error(f"Brain error: {e}")
        # Return state to avoid amnesia, plus the error
        return {
            "error": str(e),
            "todo_list": state.get("todo_list", []),
            "memory": state.get("memory", {})
        }


async def execute_next_action(state: State, config: Optional[Dict] = None) -> Dict:
    """
    The Hands Node.
    Executes Python Code, Terminal, Agents, or Tools.
    """
    try:
        todo_list = state.get("todo_list", [])
        memory = state.get("memory", {})
        current_task_id = state.get("current_task_id")
        current_task = next((t for t in todo_list if t['id'] == current_task_id), None)
        
        if not current_task:
            return {"error": "No current task found to execute."}
            
        logger.info(f"Executing: {current_task['description']}")
        
        snippet = current_task.get("code_snippet", "")
        result_content = ""
        is_success = False
        
        if snippet.startswith("TERM:"):
            cmd = snippet[5:].strip()
            # Terminal service is sync (subprocess matches sync)
            start_time = time.time()
            res = terminal_service.execute_command(cmd)
            duration = (time.time() - start_time) * 1000
            
            result_content = res['stdout'] or res['stderr']
            is_success = res['returncode'] == 0
            
            telemetry_service.log_tool_call("Terminal", is_success, duration)
            
        elif snippet.startswith("AGENT:"):
            # Greedy Match for multi-word agent names
            agents = agent_registry.list_active_agents()
            agent_names = [a['name'] for a in agents]
            
            agent = None
            agent_name = ""
            instruction = ""
            
            # Try matching from longest to shortest name
            sorted_agent_names = sorted(agent_names, key=len, reverse=True)
            content = snippet[6:].strip()
            
            for name in sorted_agent_names:
                if content.lower().startswith(name.lower()):
                    agent_name = name
                    agent = next(a for a in agents if a['name'] == name)
                    instruction = content[len(name):].strip()
                    break
            
            if not agent:
                 # Fallback to old splitting if no match found
                 parts = content.split(' ', 1)
                 agent_name = parts[0]
                 instruction = parts[1] if len(parts) > 1 else current_task['description']
                 agent = next((a for a in agents if a['name'].lower() == agent_name.lower()), None)
            
            start_time = time.time()
            if agent:
                 # 1. Try to get URL from connection_config (Dynamic/Registry-based)
                 base_url = (agent.get('connection_config') or {}).get('base_url')
                 if not base_url:
                      # Hardcoded fallback
                      port_map = {"SpreadsheetAgent": 9000, "DocumentAgent": 8070, "BrowserAgent": 8090, "MailAgent": 8040}
                      port = port_map.get(agent['name'], 8000)
                      base_url = f"http://localhost:{port}"

                 url = f"{base_url}/process" 
                 
                 async with httpx.AsyncClient(timeout=60.0) as client:
                      try:
                           resp = await client.post(url, json={"request": instruction})
                           agent_res = resp.json()
                           result_content = json.dumps(agent_res)
                           is_success = resp.status_code == 200
                      except Exception as e:
                           result_content = f"Agent Connection Error: {e}"
                           is_success = False
            else:
                result_content = f"Agent {agent_name} not found."
                is_success = False
            
            duration = (time.time() - start_time) * 1000
            telemetry_service.log_agent_call(agent_name, is_success, duration)
                
        elif snippet.startswith("TOOL:"):
            parts = snippet[5:].strip().split(' ', 1)
            tool_name = parts[0]
            args_str = parts[1] if len(parts) > 1 else "{}"
            
            try:
                args = json.loads(args_str)
            except:
                args = {"query": args_str} if args_str else {}
            
            logger.info(f"Invoking Tool: {tool_name} with {args}")
            exec_result = await tool_registry.execute_tool(tool_name, args)
            
            if exec_result["success"]:
                result_content = str(exec_result["result"])
                is_success = True
            else:
                result_content = exec_result["error"]
                is_success = False
                
        else:
            # Default to Code Sandbox
            res = code_sandbox.execute_code(
                f"# Execution for: {current_task['description']}\nprint('Executed {current_task['description']}')",
                session_id="orchestrator_main"
            )
            result_content = res['stdout']
            is_success = res['success']

        # Process result through CMS hooks
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        processed_result = await hooks.on_task_complete(
            current_task['description'],
            {"result": result_content, "status": "completed" if is_success else "failed"},
            thread_id
        )

        # Update Task
        for t in todo_list:
            if t['id'] == current_task_id:
                t['result'] = processed_result
                t['status'] = TaskStatus.COMPLETED if is_success else TaskStatus.FAILED

        return {
            "todo_list": todo_list,
            "memory": memory,
            "error": None
        }

    except Exception as e:
        logger.error(f"Execution error: {e}")
        return {
            "error": str(e),
            "todo_list": state.get("todo_list", []),
            "memory": state.get("memory", {})
        }


def _initialize_todo_list(state: State) -> Dict:
    initial_task = TaskItem(
        description="Analyze user request and create a detailed plan",
        status=TaskStatus.PENDING
    )
    return {
        "todo_list": [initial_task.dict()],
        "iteration_count": 0,
        "memory": {}
    }

def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text
