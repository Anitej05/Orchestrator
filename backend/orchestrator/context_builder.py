"""
Context Builder Module

Builds comprehensive orchestrator context from conversation history and execution state.
This enables the orchestrator to make intelligent decisions using full context.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ContextBuilder:
    """Builds comprehensive context for orchestrator decision-making."""
    
    @staticmethod
    def extract_user_preferences(messages: List[Any]) -> Dict[str, Any]:
        """
        Extracts user preferences and patterns from conversation history.
        
        Returns:
        {
            'focus_areas': [areas mentioned multiple times],
            'interaction_style': 'detailed|concise|conversational',
            'complexity_preference': 'simple|moderate|complex',
            'time_sensitivity': boolean,
            'data_format_preference': 'text|structured|visual',
        }
        """
        
        preferences = {
            'focus_areas': [],
            'interaction_style': 'conversational',
            'complexity_preference': 'moderate',
            'time_sensitivity': False,
            'data_format_preference': 'structured'
        }
        
        if not messages:
            return preferences
        
        # Analyze recent user messages (human type)
        user_messages = [
            msg.content for msg in messages 
            if hasattr(msg, 'type') and msg.type == "human"
        ]
        
        # Detect patterns
        all_content = " ".join(user_messages).lower()
        
        # Time sensitivity indicators
        if any(word in all_content for word in ['urgent', 'asap', 'quick', 'fast', 'immediately', 'now']):
            preferences['time_sensitivity'] = True
        
        # Format preference
        if any(word in all_content for word in ['table', 'chart', 'graph', 'visual', 'show me']):
            preferences['data_format_preference'] = 'visual'
        elif any(word in all_content for word in ['json', 'structured', 'format', 'clean']):
            preferences['data_format_preference'] = 'structured'
        
        # Interaction style
        if any(word in all_content for word in ['detailed', 'explain', 'comprehensive', 'thorough']):
            preferences['interaction_style'] = 'detailed'
        elif any(word in all_content for word in ['short', 'quick', 'brief', 'summary']):
            preferences['interaction_style'] = 'concise'
        
        return preferences
    
    @staticmethod
    def extract_request_patterns(messages: List[Any]) -> List[Dict[str, str]]:
        """
        Extracts patterns from user requests to understand intent.
        
        Returns list of:
        {
            'type': 'search|analyze|compare|fetch|combine',
            'subject': 'what is being requested',
            'frequency': 'how many times requested'
        }
        """
        
        patterns = []
        
        user_messages = [
            msg.content for msg in messages 
            if hasattr(msg, 'type') and msg.type == "human"
        ]
        
        # Detect request types
        request_type_keywords = {
            'search': ['search', 'find', 'look', 'get information'],
            'analyze': ['analyze', 'analyze', 'break down', 'explain'],
            'compare': ['compare', 'difference', 'vs', 'versus'],
            'fetch': ['fetch', 'get', 'retrieve', 'pull'],
            'combine': ['combine', 'merge', 'together', 'both']
        }
        
        detected_types = set()
        all_content = " ".join(user_messages).lower()
        
        for req_type, keywords in request_type_keywords.items():
            if any(kw in all_content for kw in keywords):
                detected_types.add(req_type)
        
        return [{'type': t, 'frequency': all_content.count(t)} for t in detected_types]
    
    @staticmethod
    def extract_failed_tasks(completed_tasks: List[Dict[str, Any]]) -> List[str]:
        """Extract names of tasks that failed or had errors."""
        
        failed = []
        for task in completed_tasks:
            status = task.get('status', 'unknown')
            if status in ['failed', 'error']:
                failed.append(task.get('task_name', 'unknown'))
        
        return failed
    
    @staticmethod
    def build_full_context(
        messages: List[Any],
        current_plan: List[List[Any]],
        task_agent_pairs: List[Any],
        completed_tasks: List[Dict[str, Any]],
        task_errors: Optional[Dict[str, str]] = None,
        available_agents: Optional[Dict[str, Any]] = None,
        user_requirements: Optional[Dict[str, float]] = None,
        plan_modifications: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds comprehensive context for orchestrator decisions.
        
        This context is used by:
        - analyze_user_update: Decide how to handle new user input
        - rank_agents: Select appropriate agents
        - plan_execution: Create execution plan
        - validate_plan: Check task dependencies
        """
        
        logger.info("Building comprehensive orchestrator context")
        
        # Format conversation history
        formatted_messages = []
        for msg in messages[-15:]:  # Last 15 messages for context window
            role = "User" if hasattr(msg, 'type') and msg.type == "human" else "Assistant"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            formatted_messages.append(f"{role}: {content[:300]}")  # Truncate long content
        
        conversation_history = "\n".join(formatted_messages)
        
        # Extract patterns and preferences
        user_preferences = ContextBuilder.extract_user_preferences(messages)
        request_patterns = ContextBuilder.extract_request_patterns(messages)
        failed_tasks = ContextBuilder.extract_failed_tasks(completed_tasks)
        
        # Build current plan overview
        current_tasks = []
        for batch_idx, batch in enumerate(current_plan):
            for task in batch:
                task_dict = task.model_dump() if hasattr(task, 'model_dump') else task
                current_tasks.append({
                    'task_name': task_dict.get('task_name'),
                    'description': task_dict.get('task_description'),
                    'batch': batch_idx
                })
        
        # Execution progress
        completed_task_names = [t.get('task_name', '') for t in completed_tasks]
        remaining_task_names = [t['task_name'] for t in current_tasks if t['task_name'] not in completed_task_names]
        
        # Build modification timeline
        modification_timeline = []
        if plan_modifications:
            for mod in plan_modifications[-5:]:  # Last 5
                mod_dict = mod.model_dump() if hasattr(mod, 'model_dump') else mod
                modification_timeline.append({
                    'type': mod_dict.get('modification_type'),
                    'tasks': mod_dict.get('affected_tasks'),
                    'reasoning': mod_dict.get('reasoning', '')[:150],
                    'timestamp': mod_dict.get('timestamp')
                })
        
        context = {
            # Conversation context
            'conversation': {
                'total_turns': len(messages),
                'recent_history': conversation_history,
                'user_preferences': user_preferences,
                'request_patterns': request_patterns
            },
            
            # Plan context
            'plan': {
                'total_tasks': len(current_tasks),
                'batches': len(current_plan),
                'tasks': current_tasks,
                'modifications': modification_timeline
            },
            
            # Execution context
            'execution': {
                'completed_tasks': len(completed_task_names),
                'completed_task_names': completed_task_names,
                'remaining_tasks': len(remaining_task_names),
                'remaining_task_names': remaining_task_names,
                'failed_tasks': failed_tasks,
                'task_errors': task_errors or {}
            },
            
            # Available resources
            'resources': {
                'available_agents': list(available_agents.keys()) if available_agents else [],
                'user_requirements': user_requirements or {},
                'task_agent_pairs': [
                    {
                        'task': pair.task_name if hasattr(pair, 'task_name') else pair.get('task_name'),
                        'agent': pair.primary.name if (hasattr(pair, 'primary') and hasattr(pair.primary, 'name')) else 'unknown'
                    }
                    for pair in task_agent_pairs
                ] if task_agent_pairs else []
            }
        }
        
        logger.info(f"Context built: {len(messages)} messages, {len(current_tasks)} current tasks, {len(completed_task_names)} completed")
        
        return context
    
    @staticmethod
    def format_context_for_llm(context: Dict[str, Any]) -> str:
        """
        Formats the full context into a readable string for LLM prompts.
        """
        
        formatted = f"""
=== ORCHESTRATOR CONTEXT ===

CONVERSATION HISTORY:
{context['conversation']['recent_history']}

USER PREFERENCES:
- Style: {context['conversation']['user_preferences']['interaction_style']}
- Complexity: {context['conversation']['user_preferences']['complexity_preference']}
- Time Sensitive: {context['conversation']['user_preferences']['time_sensitivity']}
- Format Preference: {context['conversation']['user_preferences']['data_format_preference']}

CURRENT EXECUTION PLAN:
- Total Tasks: {context['plan']['total_tasks']}
- Batches: {context['plan']['batches']}
- Tasks: {json.dumps(context['plan']['tasks'], indent=2)}

EXECUTION PROGRESS:
- Completed: {context['execution']['completed_tasks']} tasks
- Remaining: {context['execution']['remaining_tasks']} tasks
- Failed: {len(context['execution']['failed_tasks'])} tasks
- Remaining Task Names: {', '.join(context['execution']['remaining_task_names'])}

AVAILABLE AGENTS:
{', '.join(context['resources']['available_agents'])}

PREVIOUS MODIFICATIONS:
{json.dumps(context['plan']['modifications'], indent=2)}

=== END CONTEXT ===
"""
        
        return formatted

