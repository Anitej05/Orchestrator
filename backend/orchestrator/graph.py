# In Project_Agent_Directory/orchestrator/graph.py

from orchestrator.state import State, CompletedTask
from schemas import (
    ParsedRequest,
    PriorityMappingResponse,
    TaskAgentPair,
    ExecutionPlan,
    SelectedEndpoint,
    AgentCard,
    PlannedTask
)
from sentence_transformers import SentenceTransformer
from models import AgentCapability
import httpx
import asyncio
import json
import time
import os
import re
import io
import numpy as np
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.networks import HttpUrl
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage, ChatMessage
from langchain_cerebras import ChatCerebras
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Literal
from database import SessionLocal
from models import AgentCapability
from sqlalchemy import select
import logging
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableConfig

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, HttpUrl):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

# --- Utility Function ---
def extract_json_from_response(text: str) -> str | None:
    """
    A robust function to extract a JSON object from a string that may contain
    <think> blocks, markdown, and other conversational text.

    Args:
        text: The raw string output from the language model.

    Returns:
        A clean string of the JSON object if found, otherwise None.
    """
    if not isinstance(text, str):
        return None

    # 1. First, try to find a JSON object embedded in a markdown code block.
    # This is the most reliable method. The regex is non-greedy.
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 2. If no markdown block is found, strip any <think> blocks and then
    # try to find the first valid JSON object in the remaining text.
    text_no_thinking = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 3. Find the first '{' and the last '}' in the cleaned text. This is a
    # common pattern for LLM responses that are just the JSON object.
    start = text_no_thinking.find('{')
    end = text_no_thinking.rfind('}')
    if start != -1 and end != -1 and end > start:
        potential_json = text_no_thinking[start:end+1]
        try:
            # Validate if the extracted string is actually valid JSON
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            # The substring was not valid JSON, so we'll pass and let the next method try
            pass

    # 4. As a last resort, if the above methods fail, return None.
    return None

def serialize_complex_object(obj):
    """Helper function to serialize complex objects consistently"""
    try:
        # First try direct JSON serialization
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Handle different object types
        if isinstance(obj, HttpUrl):
            return str(obj)  # Convert HttpUrl to string
        elif hasattr(obj, 'dict'):
            # Pydantic models
            try:
                return obj.dict()
            except:
                pass
        elif hasattr(obj, '__dict__'):
            # Regular Python objects
            try:
                return obj.__dict__
            except:
                pass
        elif isinstance(obj, (list, tuple)):
            # Handle lists/tuples of complex objects
            try:
                return [serialize_complex_object(item) for item in obj]
            except:
                pass
        elif hasattr(obj, 'model_dump'):
            # Newer Pydantic v2 models
            try:
                return obj.model_dump()
            except:
                pass
        
        # Fallback to string representation
        return str(obj)

# --- New Pydantic Schemas for New Nodes ---
class PlanValidation(BaseModel):
    """Schema for the pre-flight plan validation node."""
    status: str = Field(..., description="Either 'ready' or 'user_input_required'.")
    question: Optional[str] = Field(None, description="The question to ask the user if parameters are missing.")

class AgentResponseEvaluation(BaseModel):
    """Schema for evaluating an agent's response post-flight."""
    status: str = Field(..., description="Either 'complete' or 'user_input_required'.")
    question: Optional[str] = Field(None, description="The clarifying question to ask the user if the result is vague.")

class PlanValidationResult(BaseModel):
    """Schema for the advanced validation node's output."""
    status: Literal["ready", "replan_needed", "user_input_required"] = Field(..., description="The status of the plan validation.")
    reasoning: Optional[str] = Field(None, description="Required explanation if status is 'replan_needed' or 'user_input_required'.")
    question: Optional[str] = Field(None, description="The direct question for the user if input is absolutely required.")


def strip_think_tags(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

original_generate = ChatCerebras._generate

def patched_generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
    chat_result = original_generate(self, messages, **kwargs)
    for generation in chat_result.generations:
        if isinstance(generation, ChatGeneration) and hasattr(generation.message, 'content'):
            original_content = generation.message.content
            generation.message.content = strip_think_tags(original_content)
    return chat_result

def invoke_json(self, prompt: str, pydantic_schema: Any, max_retries: int = 3):
    """
    A more robust version of invoke_json that uses the enhanced parser.
    """
    original_prompt = prompt 
    
    for attempt in range(max_retries):
        failed_object_str = ""
        try:
            # The initial prompt to the LLM remains the same
            json_prompt = f"""
            {prompt}

            Please provide your response in a valid JSON format that adheres to the following Pydantic schema:
            
            ```json
            {json.dumps(pydantic_schema.model_json_schema(), indent=2)}
            ```

            IMPORTANT: Only output the JSON object itself, without any extra text, explanations, or markdown formatting.
            """
            response_content = self.invoke(json_prompt).content
            logger.info(f"LLM RAW RESPONSE (Attempt {attempt + 1}):\\n---START---\\n{response_content}\\n---END---")

            # --- USE THE NEW PARSER HERE ---
            json_str = extract_json_from_response(response_content)
            
            if json_str:
                parsed_json = json.loads(json_str)
                validated_obj = pydantic_schema.model_validate(parsed_json)
                
                # Use model_dump_json for safe serialization
                failed_object_str = validated_obj.model_dump_json(indent=2)
                
                return validated_obj
            else:
                # If the new parser returns None, no valid JSON was found
                raise ValueError("No valid JSON object could be extracted from the response.")

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                # The retry logic remains the same
                retry_context = f"<your_previous_invalid_output>\\n{failed_object_str}\\n</your_previous_invalid_output>" if failed_object_str else ""
                prompt = f"""
                Your previous attempt failed because the output was not valid JSON or could not be extracted. Please re-evaluate the original request and provide a valid, clean JSON response.
                <error>{e}</error>
                {retry_context}
                Original prompt was:\\n{original_prompt}
                Please correct your response and try again.
                """
            else:
                logging.error(f"Failed to get a valid JSON response after {max_retries} attempts.")
                raise

ChatCerebras._generate = patched_generate
ChatCerebras.invoke_json = invoke_json

logging.info("ChatCerebras has been monkey-patched to strip <think> tags and handle JSON manually.")

load_dotenv()

logger = logging.getLogger("AgentOrchestrator")
PLAN_DIR = "agent_plans"
os.makedirs(PLAN_DIR, exist_ok=True)

embedding_model = SentenceTransformer('all-mpnet-base-v2')

cached_capabilities = {
    "texts": [],
    "embeddings": None,
    "timestamp": 0
}
CACHE_DURATION_SECONDS = 300

# --- New File-Based Memory Functions ---
def save_plan_to_file(state: State):
    """Saves the current plan and completed tasks to a Markdown file."""
    thread_id = state.get("thread_id")
    if not thread_id:
        return {}

    plan_path = os.path.join(PLAN_DIR, f"{thread_id}-plan.md")

    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(f"# Execution Plan for Thread: {thread_id}\n\n")
        f.write(f"**Original Prompt:** {state.get('original_prompt', 'N/A')}\n\n")

        f.write("## Pending Tasks\n")
        if state.get("task_plan"):
            for i, batch in enumerate(state["task_plan"]):
                f.write(f"### Batch {i+1}\n")
                for task in batch:
                    task_name = getattr(task, 'task_name', 'N/A')
                    task_description = getattr(task, 'task_description', 'N/A')
                    primary_id = getattr(getattr(task, 'primary', None), 'id', 'N/A')

                    # Write each line separately to ensure correct newlines
                    f.write(f"- **Task**: `{task_name}`\n")
                    f.write(f"  - **Description**: {task_description}\n")
                    f.write(f"  - **Agent**: {primary_id}\n")
        else:
            f.write("- No pending tasks.\n")

        f.write("\n## Completed Tasks\n")
        if state.get("completed_tasks"):
            for task in state["completed_tasks"]:
                task_name = task.get('task_name', 'N/A')
                result_str = json.dumps(task.get('result', {}), indent=2, cls=CustomJSONEncoder)
                
                # Indent every line of the JSON string for proper markdown rendering
                indented_result_str = "\n".join("      " + line for line in result_str.splitlines())

                # Write each part of the completed task entry separately
                f.write(f"- **Task**: `{task_name}`\n")
                f.write("  - **Result**:\n")
                f.write("    ```json\n")
                f.write(f"{indented_result_str}\n")
                f.write("    ```\n")
        else:
            f.write("- No completed tasks.\n")

    logger.info(f"Plan for thread {thread_id} saved to {plan_path}")
    return {}

# --- Existing and Modified Graph Nodes ---

def get_all_capabilities():
    global cached_capabilities
    now = time.time()

    if now - cached_capabilities["timestamp"] < CACHE_DURATION_SECONDS and cached_capabilities["texts"]:
        logger.info("Using cached capabilities and embeddings.")
        return cached_capabilities["texts"], cached_capabilities["embeddings"]

    logger.info("Fetching and embedding capabilities from database...")
    db = SessionLocal()
    try:
        results = db.query(AgentCapability.capability_text).distinct().all()
        capability_texts = [res[0] for res in results]
        
        if capability_texts:
            cached_capabilities["texts"] = capability_texts
            # Embeddings are no longer used in the graph but kept for potential future use
            cached_capabilities["timestamp"] = now
        else:
            cached_capabilities["texts"] = []
            
        return cached_capabilities["texts"], None # Return None for embeddings
    finally:
        db.close()

async def fetch_agents_for_task(client: httpx.AsyncClient, task_name: str, url: str):
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        validated_agents = [AgentCard.model_validate(agent) for agent in response.json()]
        return {"task_name": task_name, "agents": validated_agents}
    except (httpx.RequestError, httpx.HTTPStatusError, ValidationError) as e:
        logger.error(f"API call or validation failed for task '{task_name}': {e}")
        return {"task_name": task_name, "agents": []}

def parse_prompt(state: State):
    logger.info(f"Parsing prompt: '{state['original_prompt']}'")
    llm = ChatCerebras(model="gpt-oss-120b")

    # --- Create a formatted history of the conversation ---
    history = ""
    if messages := state.get('messages'):
        # Limit to the last few messages to keep the prompt concise
        for msg in messages[-5:]: # Using the last 5 messages as context
            if hasattr(msg, 'type') and msg.type == "human":
                history += f"Human: {msg.content}\n"
            elif hasattr(msg, 'type') and msg.type == "ai":
                history += f"AI: {msg.content}\n"
    # ---

    capability_texts, _ = get_all_capabilities()
    capabilities_list_str = ", ".join(f"'{c}'" for c in capability_texts)

    error_feedback = state.get("parsing_error_feedback")
    retry_prompt_injection = ""
    if error_feedback:
        retry_prompt_injection = f"""
        **IMPORTANT - PREVIOUS ATTEMPT FAILED:**
        You are being asked to try again because your previous attempt failed to produce a useful result.
        **Failure Feedback:** {error_feedback}
        Please analyze this feedback and the original user prompt carefully and generate a new set of tasks with much more detailed and specific `task_description` fields.
        """

    prompt = f"""
        You are an expert at breaking down any user request—no matter how short, vague, or poorly written—into a clear list of distinct tasks that can each be handled by a single agent.
        {retry_prompt_injection}

        Here is the recent conversation history for context:
        ---
        {history}
        ---

        Here is a list of agent capabilities that already exist in the system:
        ---
        AVAILABLE CAPABILITIES: [{capabilities_list_str}]
        ---

        Follow these rules:
        1.  **Group Related Information:** If the user asks for multiple pieces of information that are likely to be returned by a single tool or API call (e.g., "get news headlines, publishers, and links" or "get a stock's open, high, low, and close price"), you **MUST** treat this as a single, unified task. Do not split these into separate tasks. For example, a request for "news headlines, publishers, and links" should become a single task like "get company news with details".
        2.  **One Task, One Agent:** A "task" must represent ONE coherent, self-contained action that can be given to a single agent.
        3.  **No Unnecessary Splitting:** Do NOT split a task into smaller parts unless they are truly independent and could be completed by different agents without losing context.
        4.  **Simple Language:** Keep language simple and avoid technical jargon unless the user explicitly uses it.
        5.  **Infer Intent:** If the prompt is unclear, infer the most reasonable interpretation based on common intent.
        6.  **Strict Schema:** Always output tasks in the required schema.

        For each task you identify, provide:
        1. `task_name`: A short, descriptive name (e.g., "get_company_news", "summarize_document").
            - **Check Existing Capabilities First:** When choosing a `task_name`, you **MUST** check the AVAILABLE CAPABILITIES list. If a capability in the list is a good match for the grouped task, use that exact capability as the `task_name`.
            - **Create New if Needed:** If no single existing capability is a good fit for the grouped task, create a new, concise, 2-4 word `task_name` that accurately describes the entire action (e.g., "get_news_details", "get_ohlc_prices").
            - **Prefer Existing:** Always prefer using an existing capability if it covers the user's request to ensure a higher chance of finding an agent.
        2. `task_description`: A detailed explanation of what the task is and what needs to be done, including all the details from the user's prompt. For example, for "get AAPL news headlines with publishers and links", the description should be "Get the latest news headlines for AAPL, including the publisher and a link to the article for each headline."

        Also extract any general user expectations (tone, urgency, budget, quality rating, etc.) from the prompt, if present. If not present, set them to null.

        The user's prompt will be provided like this:
        ---
        {state['original_prompt']}
        ---

        **EXAMPLE:**
        If the user prompt is "Get the latest 10 news headlines for AAPL with publishers and article links.", your output should be:
        ```json
        {{
            "tasks": [
                {{
                    "task_name": "get company news headlines",
                    "task_description": "Get the latest 10 news headlines for AAPL, including the publisher and a link to the article for each headline."
                }}
            ],
            "user_expectations": {{}}
        }}
        ```

        Your output must follow the schema exactly, and all number fields must be numeric or null (never strings).
        When extracting `user_expectations`, follow this strictly:
        - Only include fields that the user explicitly mentioned (e.g., price, budget, tone, urgency, quality rating).
        - Do NOT include any field with a null value.
        - If, after removing nulls, no fields remain, set `user_expectations` to an empty object `{{}}`.
    """

    try:
        response = llm.invoke_json(prompt, ParsedRequest)
        logger.info(f"LLM parsed prompt into: {response}")

        if not response or not response.tasks:
            raise ValueError("LLM returned a valid JSON but with an empty list of tasks.")

        parsed_tasks = getattr(response, 'tasks', [])
        user_expectations = getattr(response, 'user_expectations', {})
    except Exception as e:
        logger.error(f"Failed to parse prompt after all retries: {e}")
        parsed_tasks = []
        user_expectations = {}

    current_retry_count = state.get('parse_retry_count', 0)

    return {
        "parsed_tasks": parsed_tasks,
        "user_expectations": user_expectations or {},
        "parsing_error_feedback": None,
        "parse_retry_count": current_retry_count + 1
    }

async def agent_directory_search(state: State):
    parsed_tasks = state.get('parsed_tasks', [])
    logger.info(f"Searching for agents for tasks: {[t.task_name for t in parsed_tasks]}")
    
    if not parsed_tasks:
        logger.warning("No valid tasks to process in agent_directory_search")
        return {"candidate_agents": {}}
    
    urls_to_fetch = []
    base_url = "http://127.0.0.1:8000/api/agents/search"
    user_expectations = state.get('user_expectations') or {}

    for task in parsed_tasks:
        params: Dict[str, Any] = {'capabilities': task.task_name}
        if 'price' in user_expectations:
            params['max_price'] = user_expectations['price']
        if 'rating' in user_expectations:
            params['min_rating'] = user_expectations['rating']
        
        request = httpx.Request("GET", base_url, params=params)
        urls_to_fetch.append((task.task_name, str(request.url), task.task_description))
    
    logger.info(f"Dispatching {len(urls_to_fetch)} agent search requests.")
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch_agents_for_task(client, name, url) for name, url, desc in urls_to_fetch))
    
    candidate_agents_map = {res['task_name']: res['agents'] for res in results}
    logger.info("Agent search complete.")

    for task in parsed_tasks:
        if not candidate_agents_map.get(task.task_name):
            error_feedback = (
                f"The previous attempt to parse the prompt resulted in the task description "
                f"'{task.task_description}', which was matched to the capability '{task.task_name}'. "
                f"However, no agents were found that could perform this task. Please generate a new, "
                f"more detailed and specific task description that better captures the user's intent."
            )
            logger.warning(f"Semantic failure for task '{task.task_name}'. Looping back to re-parse.")
            return {"candidate_agents": {}, "parsing_error_feedback": error_feedback}

    return {"candidate_agents": candidate_agents_map, "parsing_error_feedback": None}

class RankedAgents(BaseModel):
    ranked_agent_ids: List[str]

def rank_agents(state: State):
    # The rest of the function remains the same
    parsed_tasks = state.get('parsed_tasks', [])
    logger.info(f"Ranking agents for tasks: {[t.task_name for t in parsed_tasks]}")
    
    if not parsed_tasks:
        logger.warning("No tasks to rank in rank_agents")
        return {"task_agent_pairs": []}
    
    llm = ChatCerebras(model="gpt-oss-120b")
    
    final_selections = []
    for task in parsed_tasks:
        task_name = task.task_name
        candidate_agents = state.get('candidate_agents', {}).get(task_name, [])
        
        if not candidate_agents:
            continue

        if len(candidate_agents) == 1:
            primary_agent = candidate_agents[0]
            fallback_agents = []
        else:
            serializable_agents = [agent.model_dump() for agent in candidate_agents]
            
            prompt = f"""
            You are an expert at selecting the best agent for a given task.
            The user's task is: "{task.task_description}"

            Here are the available agents that claim to have the capability '{task.task_name}':
            ---
            {json.dumps(serializable_agents, indent=2, cls=CustomJSONEncoder)}
            ---

            Please rank these agents in order of suitability for the task, from best to worst. The best agent should be the one whose description and capabilities most closely match the user's task.

            Your output should be a JSON object with a single key, "ranked_agent_ids", which is a list of agent IDs in the correct order.
            """
            try:
                response = llm.invoke_json(prompt, RankedAgents)
                ranked_agent_ids = response.ranked_agent_ids
                
                sorted_agents = sorted(candidate_agents, key=lambda agent: ranked_agent_ids.index(agent.id) if agent.id in ranked_agent_ids else float('inf'))
                
                primary_agent = sorted_agents[0]
                fallback_agents = sorted_agents[1:4]
                
            except Exception as e:
                logger.error(f"LLM agent ranking failed: {e}. Falling back to default ranking.")
                scored_agents = []
                prices = [agent.price_per_call_usd for agent in candidate_agents if agent.price_per_call_usd is not None]
                min_price, max_price = (min(prices), max(prices)) if prices else (0, 0)
                price_range = (max_price - min_price) if (max_price > min_price) else 1.0
                
                for agent in candidate_agents:
                    norm_rating = (agent.rating - 1) / 4.0 if agent.rating is not None else 0
                    norm_price = 1 - ((agent.price_per_call_usd - min_price) / price_range) if price_range > 0 and agent.price_per_call_usd is not None else 1.0
                    score = (0.6 * norm_rating) + (0.4 * norm_price)
                    scored_agents.append({"agent": agent, "score": score})
                
                sorted_agents_by_score = sorted(scored_agents, key=lambda x: x['score'], reverse=True)
                primary_agent = sorted_agents_by_score[0]['agent']
                fallback_agents = [item['agent'] for item in sorted_agents_by_score[1:4]]

        pair = TaskAgentPair(
            task_name=task_name,
            task_description=task.task_description,
            primary=primary_agent,
            fallbacks=fallback_agents
        )
        final_selections.append(pair)
    
    logger.info("Agent ranking complete.")
    logger.debug(f"Final agent selections: {[p.model_dump_json(indent=2) for p in final_selections]}")
    return {"task_agent_pairs": final_selections}

def plan_execution(state: State, config: RunnableConfig):
    """
    Creates an initial execution plan or modifies an existing one if a replan is needed,
    and saves the result to a file.
    """
    replan_reason = state.get("replan_reason")
    llm = ChatCerebras(model="gpt-oss-120b")
    output_state = {}

    if replan_reason:
        # --- REPLANNING MODE ---
        logger.info(f"Replanning initiated. Reason: {replan_reason}")
        
        all_capabilities, _ = get_all_capabilities()
        capabilities_str = ", ".join(all_capabilities)

        prompt = f"""
        You are an expert autonomous planner. The current execution plan is stalled. Your task is to surgically insert a new task into the plan to resolve the issue.

        **Reason for Replan:** "{replan_reason}"
        **Current Stalled Plan:** {json.dumps([task.model_dump() for batch in state.get('task_plan', []) for task in batch], indent=2)}
        **Original User Prompt:** "{state['original_prompt']}"
        **Full List of Available System Capabilities:** [{capabilities_str}]
        
        **Instructions:**
        1.  Analyze the `Reason for Replan` to understand what's missing (e.g., "missing coordinates for Hyderabad").
        2.  Identify the best capability from the `Available System Capabilities` to find this missing information. The **"perform web search and summarize"** capability is perfect for this.
        3.  Create a new `PlannedTask`. The `task_description` should be a clear, self-contained instruction for another agent (e.g., "Find the latitude and longitude for Hyderabad, India using a web search"). You must select an agent and endpoint that provides the chosen capability.
        4.  **Insert this new task into the `Current Stalled Plan` *immediately before* the task that needs the information.**
        5.  Return the entire modified plan. The output MUST be a valid JSON object conforming to the `ExecutionPlan` schema.
        """
        try:
            response = llm.invoke_json(prompt, ExecutionPlan)
            output_state = {"task_plan": response.plan, "replan_reason": None} # Clear the reason after replanning
        except Exception as e:
            logger.error(f"Replanning failed: {e}. Falling back to asking user.")
            output_state = {
                "pending_user_input": True,
                "question_for_user": f"I tried to solve the issue of '{replan_reason}' but failed. Could you please provide the missing information directly?"
            }

    else:
        # --- INITIAL PLANNING MODE ---
        logger.info("Creating initial execution plan.")
        task_agent_pairs = state.get('task_agent_pairs', [])
        if not task_agent_pairs:
            return {"task_plan": []}

        prompt = f"""
        You are an expert project planner. Convert a list of tasks and their assigned agents into a final, executable plan.
        
        **Instructions:**
        1.  For each task, select the most appropriate endpoint from the `primary` agent's list.
        2.  Create an `ExecutionStep` with the agent `id`, `http_method`, and `endpoint`.
        3.  **Do not generate a payload.**
        4.  Your final `plan` must be a list of batches (list of lists). Group tasks that can run in parallel into the same batch. A task that depends on another must be in a subsequent batch.

        **Tasks to Plan:** {[p.model_dump_json() for p in task_agent_pairs]}
        You MUST only output a valid JSON object that conforms to the ExecutionPlan schema.
        """
        try:
            response = llm.invoke_json(prompt, ExecutionPlan)
            output_state = {"task_plan": response.plan or [], "user_response": None}
        except Exception as e:
            logger.error(f"Initial planning failed: {e}")
            output_state = {"task_plan": []}

    # *** THIS IS THE CRITICAL ADDITION ***
    # Save the new or modified plan to the file system immediately.
    save_plan_to_file({**state, **output_state, "thread_id": config.get("configurable", {}).get("thread_id")})
    
    return output_state


def validate_plan_for_execution(state: State):
    """
    Performs an advanced pre-flight check. It sets state flags to determine routing:
    - Sets 'replan_reason' if a solvable dependency is missing.
    - Sets 'pending_user_input' if user intervention is required.
    """
    logger.info("Performing dynamic validation of the execution plan...")
    task_plan = state.get("task_plan", [])
    if not task_plan or not task_plan[0]:
        return {"replan_reason": None, "pending_user_input": False}

    all_capabilities, _ = get_all_capabilities()
    capabilities_str = ", ".join(all_capabilities)
    
    llm = ChatCerebras(model="gpt-oss-120b")
    task_to_validate = task_plan[0][0]

    task_agent_pair = next((p for p in state.get('task_agent_pairs', []) if p.task_name == task_to_validate.task_name), None)
    if not task_agent_pair: return {"replan_reason": None, "pending_user_input": False}

    agent_card = task_agent_pair.primary
    selected_endpoint = next((ep for ep in agent_card.endpoints if str(ep.endpoint) == str(task_to_validate.primary.endpoint)), None)
    required_params = [p.name for p in selected_endpoint.parameters if p.required] if selected_endpoint else []

    if not required_params:
        return {"replan_reason": None, "pending_user_input": False}

    prompt = f"""
    You are an intelligent execution validator. Your job is to determine if a task can run, and if not, figure out how to unblock it.

    **Context:**
    - Original User Prompt: "{state['original_prompt']}"
    - Previously Completed Tasks: {state.get('completed_tasks', [])}
    - Task to Validate: "{task_to_validate.task_description}"
    - Required Parameters for this Task: {required_params}
    - All Available System Capabilities: [{capabilities_str}]

    **Your Decision Process:**
    1.  **Check Context:** Can all `Required Parameters` (e.g., 'latitude', 'longitude') be filled using the `Original User Prompt` or `Previously Completed Tasks`?
    2.  **If YES:** The task is ready. Respond with `status: "ready"`.
    3.  **If NO:** Determine the root cause.
        a. **Is the information implicitly available?** For example, if 'latitude' and 'longitude' are required but a city name (e.g., "Hyderabad") is available in the context, can the missing information be found?
        b. **Can another agent solve it?** Look at the `All Available System Capabilities`. Is there a capability like **"perform web search and summarize"** that could find the missing information?
           - If yes, the plan needs a new step. Respond with `status: "replan_needed"` and a clear `reasoning` that states exactly what is missing and how to find it (e.g., "Missing latitude and longitude for Hyderabad, which can be found using the 'perform web search and summarize' capability.").
        c. **Is the information truly missing?** Is it something only the user would know (like a private document or a personal preference)?
           - If yes, respond with `status: "user_input_required"` and formulate a clear, direct `question` for the user.

    Respond in a valid JSON format conforming to the PlanValidationResult schema.
    """
    
    validation = llm.invoke_json(prompt, PlanValidationResult)
    logger.info(f"Validation result: {validation.status}")

    if validation.status == "replan_needed":
        return {"replan_reason": validation.reasoning, "pending_user_input": False, "question_for_user": None}
    elif validation.status == "user_input_required":
        return {"pending_user_input": True, "question_for_user": validation.question, "replan_reason": None}
    
    return {"replan_reason": None, "pending_user_input": False, "question_for_user": None}


async def run_agent(planned_task: PlannedTask, agent_details: AgentCard, completed_tasks: List[CompletedTask], last_error: Optional[str] = None):
    """
    Runs a single agent for a task with intelligent retries for empty or unsatisfactory results.
    """
    logger.info(f"Running agent '{agent_details.name}' for task: '{planned_task.task_name}'")
    
    endpoint_url = str(planned_task.primary.endpoint)
    http_method = planned_task.primary.http_method.upper()
    
    selected_endpoint = next((ep for ep in agent_details.endpoints if str(ep.endpoint) == endpoint_url), None)

    if not selected_endpoint:
        error_msg = f"Error: Could not find endpoint details for '{endpoint_url}' on agent '{agent_details.name}'."
        logger.error(error_msg)
        return {"task_name": planned_task.task_name, "result": error_msg}

    payload_builder_llm = ChatCerebras(model="gpt-oss-120b")
    failed_attempts = []
    
    # This loop handles semantic retries (e.g., valid but empty responses)
    for attempt in range(3):
        failed_attempts_context = ""
        if failed_attempts:
            # This provides context about previous failed payloads and their empty results.
            failed_attempts_str = "\\n".join([f"- Payload: {att['payload']}\\n  - Result: {att['result']}" for att in failed_attempts])
            failed_attempts_context = f"""
            IMPORTANT: Your previous attempt(s) failed because the agent returned empty or unsatisfactory results. Do NOT repeat the same mistakes. Analyze the following failed attempts and generate a NEW, MODIFIED payload. Consider using broader search terms, different parameters, or a more general approach to maximize the chance of getting a non-empty result.

            <failed_attempts>
            {failed_attempts_str}
            </failed_attempts>
            """

        # This context is for HTTP 4xx errors from the outer loop in try_task_with_fallbacks
        http_error_context = f"\\nIMPORTANT: The last API call failed with a client error. Please correct the payload based on this feedback:\\n<error>\\n{last_error}\\n</error>\\n" if last_error else ""

        payload_prompt = f"""
        You are an expert at creating API requests. Your task is to generate a JSON payload for the following endpoint, based on the provided context.

        Endpoint Description: "{selected_endpoint.description}"
        Endpoint Parameters: {[p.model_dump_json() for p in selected_endpoint.parameters]}
        High-Level Task: "{planned_task.task_description}"
        Historical Context (previous task results): {completed_tasks}
        {http_error_context}
        {failed_attempts_context}
        Generate only the JSON payload required by the endpoint.
        """
        logger.debug(f"Payload builder prompt for task '{planned_task.task_name}' (Attempt {attempt + 1}):\\n{payload_prompt}")
        
        try:
            # Logic to generate and clean the payload
            payload_str = payload_builder_llm.invoke(payload_prompt).content
            cleaned_payload_str = strip_think_tags(payload_str)
            json_match = re.search(r"```json\\s*(\\{.*?\\})\\s*```", cleaned_payload_str, re.DOTALL)
            json_str = json_match.group(1) if json_match else cleaned_payload_str
            payload = json.loads(json_str)
            logger.info(f"LLM generated payload for task '{planned_task.task_name}': {payload}")
        except (json.JSONDecodeError, AttributeError) as e:
            error_msg = f"Error building payload for task '{planned_task.task_name}': {e}"
            logger.error(error_msg)
            return {"task_name": planned_task.task_name, "result": error_msg}

        headers = {}
        agent_id = agent_details.id
        if agent_id:
            env_var_name = f"{agent_id.upper().replace('-', '_')}_API_KEY"
            api_key = os.getenv(env_var_name)
            if api_key:
                logger.info(f"Found API key for agent '{agent_id}'. Adding to headers.")
                headers["Authorization"] = f"Bearer {api_key}"
                headers["x-scholarai-api-key"] = api_key

        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"Calling agent '{agent_details.name}' at '{endpoint_url}' with method '{http_method}'.")
                if http_method == 'GET':
                    response = await client.get(endpoint_url, params=payload, headers=headers, timeout=30.0)
                elif http_method == 'POST':
                    response = await client.post(endpoint_url, json=payload, headers=headers, timeout=30.0)
                else:
                    raise ValueError(f"Unsupported HTTP method '{http_method}'.")

                response.raise_for_status()
                
                result = response.json()

                # **INTELLIGENT VALIDATION**
                is_result_empty = not result or (isinstance(result, list) and not result) or (isinstance(result, dict) and "articles" in result and not result["articles"]) or (isinstance(result, dict) and not any(result.values()))

                if is_result_empty:
                    logger.warning(f"Agent returned a successful but empty response. Payload: {payload}. Result: {result}. Retrying...")
                    failed_attempts.append({"payload": payload, "result": str(result)})
                    continue  # This continues to the next attempt in the loop
                
                logger.info(f"Agent call successful for task '{planned_task.task_name}'. Status: {response.status_code}")
                return {"task_name": planned_task.task_name, "result": result}
            
            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
                error_msg = f"Error calling agent for task '{planned_task.task_name}': {e}"
                logger.error(error_msg)
                raw_response = e.response.text if hasattr(e, 'response') and e.response else "No response body."
                status_code = e.response.status_code if hasattr(e, 'response') else 500
                return {"task_name": planned_task.task_name, "result": error_msg, "raw_response": raw_response, "status_code": status_code}
    
    # This block is reached only if all semantic retries in the loop fail
    last_attempt_info = failed_attempts[-1] if failed_attempts else {}
    final_error_msg = f"Agent returned empty results for task '{planned_task.task_name}' after {len(failed_attempts)} attempts."
    logger.error(final_error_msg)
    return {
        "task_name": planned_task.task_name,
        "result": final_error_msg,
        "raw_response": str(last_attempt_info.get('result', 'No result from last attempt.')),
        "status_code": 500
    }

async def execute_batch(state: State):
    """Executes a single batch of tasks from the plan."""
    if not state.get('task_plan'):
        logger.info("No task plan to execute.")
        return {}

    current_batch_plan = state['task_plan'][0]
    remaining_plan = state['task_plan'][1:]
    logger.info(f"Executing batch of {len(current_batch_plan)} tasks.")
    
    task_agent_pairs_map = {pair.task_name: pair for pair in state['task_agent_pairs']}
    
    async def try_task_with_fallbacks(planned_task: PlannedTask):
        original_task_pair = task_agent_pairs_map.get(planned_task.task_name)
        if not original_task_pair:
            error_msg = f"Could not find original task pair for '{planned_task.task_name}' to get fallbacks."
            logger.error(error_msg)
            return {"task_name": planned_task.task_name, "result": error_msg}

        agents_to_try = [original_task_pair.primary] + original_task_pair.fallbacks
        final_error_result = None
        
        for agent_to_try in agents_to_try:
            max_retries = 3 if agent_to_try.id == original_task_pair.primary.id else 1
            last_error = None
            for i in range(max_retries):
                logger.info(f"Attempting task '{planned_task.task_name}' with agent '{agent_to_try.name}' (Attempt {i+1})...")
                
                task_result = await run_agent(planned_task, agent_to_try, state.get('completed_tasks', []), last_error=last_error)
                
                result_data = task_result.get('result', {})
                is_error = isinstance(result_data, str) and "Error:" in result_data
                
                if not is_error:
                    logger.info(f"Task '{planned_task.task_name}' succeeded with agent '{agent_to_try.name}'.")
                    return task_result
                
                final_error_result = task_result
                raw_response = task_result.get('raw_response', 'No raw response available.')
                logger.warning(f"Agent '{agent_to_try.name}' failed for task '{planned_task.task_name}'. Error: {result_data}")
                logger.warning(f"Raw response from failed agent: {raw_response}")
                
                status_code = task_result.get("status_code")
                if status_code and 400 <= status_code < 500:
                    last_error = raw_response
                else:
                    break
            
        logger.error(f"All agents failed for task '{planned_task.task_name}'. Returning final error.")
        return final_error_result

    batch_results = await asyncio.gather(*(try_task_with_fallbacks(planned_task) for planned_task in current_batch_plan))
    
    task_desc_map = {task.task_name: task.task_description for task in current_batch_plan}
    completed_tasks_with_desc = []
    for res in batch_results:
        task_name = res['task_name']
        completed_tasks_with_desc.append(CompletedTask(
            task_name=task_name,
            task_description=task_desc_map.get(task_name, "N/A"),
            result=res.get('result', {})
        ))

    completed_tasks = state.get('completed_tasks', []) + completed_tasks_with_desc
    logger.info("Batch execution complete.")
    
    return {
        "task_plan": remaining_plan,
        "completed_tasks": completed_tasks,
        "latest_completed_tasks": completed_tasks_with_desc
    }

def evaluate_agent_response(state: State):
    """
    Critically evaluates the result of the last executed task to ensure it is
    logically correct and satisfies the user's intent before proceeding.
    """
    latest_tasks = state.get("latest_completed_tasks", [])
    if not latest_tasks:
        # No new tasks to evaluate
        return {"pending_user_input": False, "question_for_user": None}

    llm = ChatCerebras(model="gpt-oss-120b")
    task_to_evaluate = latest_tasks[-1] # Evaluate the most recent task

    # If the agent itself reported an error, we don't need to evaluate it further
    if isinstance(task_to_evaluate.get('result'), str) and "Error:" in task_to_evaluate.get('result', ''):
        return {"pending_user_input": False, "question_for_user": None}

    prompt = f"""
    You are a meticulous Quality Assurance AI. Your job is to determine if an agent's output is a successful and logical fulfillment of its assigned task.

    **Original User Prompt:** "{state['original_prompt']}"
    **Task Description:** "{task_to_evaluate['task_description']}"
    **Agent's Result:**
    ```json
    {json.dumps(task_to_evaluate['result'], indent=2)}
    ```

    **Instructions:**
    1.  **Check for Logical Consistency:** Does the `Agent's Result` make sense in the context of the `Task Description`? (e.g., If the task was to find a "technology company," is the result actually a tech company, not a newspaper?).
    2.  **Check for Completeness:** Is the result empty, or does it contain placeholders like "N/A" or 0.0 when a real value was expected?
    3.  **Check for Unverified Assumptions:** Does the result rely on information not present in the original prompt or task description?

    **Decision:**
    - If the result is logically sound and complete, respond with `{{"status": "complete"}}`.
    - If the result is logically flawed, incomplete, or based on a wrong assumption, respond with `{{"status": "user_input_required", "question": "Formulate a clear, direct question to the user to correct the course of the plan."}}`. For example, "The news search returned an article about the lumber industry, not a tech company. Could you specify a tech company you're interested in?"
    """
    try:
        evaluation = llm.invoke_json(prompt, AgentResponseEvaluation)
        if evaluation.status == "user_input_required":
            logger.warning(f"Result for task '{task_to_evaluate['task_name']}' is unsatisfactory. Pausing for user input.")
            # We add the failed task's result to the parsing feedback to prevent loops
            error_feedback = f"The previous attempt for a similar task resulted in an incorrect output: {task_to_evaluate['result']}. Please generate a more precise task to avoid this error."
            return {
                "pending_user_input": True,
                "question_for_user": evaluation.question,
                "parsing_error_feedback": error_feedback
            }
    except Exception as e:
        logger.error(f"Failed to evaluate agent response for task '{task_to_evaluate['task_name']}': {e}")
    
    return {"pending_user_input": False, "question_for_user": None}

def ask_user(state: State):
    """
    Formats the question for the user and prepares it as the final response.
    This is a terminal node that ends the graph's execution for the current run.
    """
    # Get any existing question or create a default one based on parsing failures
    question = state.get("question_for_user")
    
    if not question:
        # Generate a default question based on the context
        parsing_error = state.get("parsing_error_feedback")
        original_prompt = state.get("original_prompt", "")
        
        if parsing_error:
            question = f"I couldn't find suitable agents for your request: '{original_prompt}'. Could you please provide more specific details about what you'd like me to help you with?"
        else:
            question = f"I need more information to help you with: '{original_prompt}'. Could you please provide more specific details about what you'd like me to do?"
    
    logger.info(f"Asking user for clarification: {question}")
    
    return {
        "pending_user_input": True,
        "question_for_user": question,
        "final_response": None  # Clear any previous final response
    }


def aggregate_responses(state: State):
    logger.info("Aggregating final response.")
    llm = ChatCerebras(model="gpt-oss-120b")
    
    prompt = f"""
    You are an expert project manager's assistant. Your job is to synthesize the results from a team of AI agents into a single, clean, and coherent final report for the user.
    The user's original request was:
    "{state['original_prompt']}"

    The following tasks were completed, with these results:
    ---
    {state['completed_tasks']}
    ---
    Please generate a final, human-readable response that directly answers the user's original request based on the collected results.
    """
    logger.debug(f"Aggregation prompt:\\n{prompt}")
    
    final_response = llm.invoke(prompt).content
    logger.info("Final response generated.")
    return {"final_response": final_response}

# --- Routing Functions ---
def route_after_search(state: State):
    """Route after agent directory search based on whether agents were found"""
    if state.get("parsing_error_feedback"):
        if state.get("parse_retry_count", 0) >= 3:
            logger.warning("Max parse retries reached. Asking user for clarification.")
            return "ask_user"
        else:
            logger.info("Retrying parse_prompt.")
            return "parse_prompt"
    return "rank_agents"

def route_after_validation(state: State):
    """This router acts as the gate after the plan is validated."""
    if state.get("replan_reason"):
        logger.info("Routing back to plan_execution for a replan.")
        return "plan_execution"
    if state.get("pending_user_input"):
        logger.info("Routing to ask_user due to failed plan validation.")
        return "ask_user"
    else:
        logger.info("Plan is valid. Routing to execute_batch.")
        return "execute_batch"

def should_continue_or_finish(state: State):
    """This router runs after execution and evaluation to decide the next step."""
    if state.get("pending_user_input"):
        # If the evaluation failed and we need user input, go to ask_user
        return "ask_user"
    if not state.get('task_plan'):
        # If the plan is empty and evaluation passed, we are done
        logger.info("Execution plan is complete. Routing to aggregate_responses.")
        return "aggregate_responses"
    else:
        # If there are more tasks and evaluation passed, continue to next batch
        logger.info("Plan has more batches. Routing back to validation for the next batch.")
        return "validate_plan_for_execution"

# --- Build the State Graph ---
builder = StateGraph(State)

builder.add_node("parse_prompt", parse_prompt)
builder.add_node("agent_directory_search", agent_directory_search)
builder.add_node("rank_agents", rank_agents)
builder.add_node("plan_execution", plan_execution)
builder.add_node("validate_plan_for_execution", validate_plan_for_execution)
builder.add_node("execute_batch", execute_batch)
builder.add_node("evaluate_agent_response", evaluate_agent_response) # New Node
builder.add_node("ask_user", ask_user)
builder.add_node("aggregate_responses", aggregate_responses)

builder.add_edge(START, "parse_prompt")
builder.add_edge("parse_prompt", "agent_directory_search")
builder.add_edge("rank_agents", "plan_execution")
builder.add_edge("plan_execution", "validate_plan_for_execution")
builder.add_edge("execute_batch", "evaluate_agent_response") # <-- New Edge
builder.add_edge("ask_user", END)
builder.add_edge("aggregate_responses", END)

builder.add_conditional_edges("agent_directory_search", route_after_search, {
    "parse_prompt": "parse_prompt", 
    "rank_agents": "rank_agents",
    "ask_user": "ask_user"
})

builder.add_conditional_edges("validate_plan_for_execution", route_after_validation, {
    "execute_batch": "execute_batch",
    "plan_execution": "plan_execution",
    "ask_user": "ask_user"
})

# The 'should_continue_or_finish' router now runs AFTER evaluation
builder.add_conditional_edges("evaluate_agent_response", should_continue_or_finish, {
    "validate_plan_for_execution": "validate_plan_for_execution",
    "aggregate_responses": "aggregate_responses",
    "ask_user": "ask_user"
})

# Compile the graph
graph = builder.compile()

def create_graph_with_checkpointer(checkpointer):
    """Create a graph with a specific checkpointer for memory/persistence"""
    return builder.compile(checkpointer=checkpointer)