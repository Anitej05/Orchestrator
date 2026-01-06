"""
Multi-stage planner for spreadsheet operations.

Implements: Propose → Revise → Execute workflow
Provides better error correction and plan history tracking.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

import pandas as pd

from .actions import SpreadsheetAction, ActionParser, ActionExecutor, actions_to_dicts
from .models import ObservationData
from .simulate import simulate_operation

logger = logging.getLogger(__name__)


# ============================================================================
# PLAN MODELS
# ============================================================================

class PlanStage(str, Enum):
    """Stages in the planning workflow"""
    PROPOSING = "proposing"
    REVISING = "revising"
    SIMULATING = "simulating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionPlan:
    """A plan with multiple actions"""
    
    def __init__(
        self,
        plan_id: str,
        instruction: str,
        actions: List[SpreadsheetAction],
        reasoning: str,
        stage: PlanStage = PlanStage.PROPOSING,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.plan_id = plan_id
        self.instruction = instruction
        self.actions = actions
        self.reasoning = reasoning
        self.stage = stage
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.simulation_result = None
        self.execution_result = None
        self.revisions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary"""
        return {
            "plan_id": self.plan_id,
            "instruction": self.instruction,
            "actions": actions_to_dicts(self.actions),
            "reasoning": self.reasoning,
            "stage": self.stage.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "simulation_result": self.simulation_result,
            "execution_result": self.execution_result,
            "revisions": self.revisions
        }


class PlanHistory:
    """Track plan history for error correction"""
    
    def __init__(self):
        self.plans: List[ExecutionPlan] = []
        self.failed_patterns: List[Dict[str, Any]] = []
    
    def add_plan(self, plan: ExecutionPlan):
        """Add plan to history"""
        self.plans.append(plan)
    
    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get plan by ID"""
        for plan in self.plans:
            if plan.plan_id == plan_id:
                return plan
        return None
    
    def get_failed_plans(self) -> List[ExecutionPlan]:
        """Get all failed plans"""
        return [p for p in self.plans if p.stage == PlanStage.FAILED]
    
    def get_successful_plans(self) -> List[ExecutionPlan]:
        """Get all successful plans"""
        return [p for p in self.plans if p.stage == PlanStage.COMPLETED]
    
    def record_failure_pattern(self, error_type: str, context: Dict[str, Any]):
        """Record a failure pattern for future avoidance"""
        self.failed_patterns.append({
            "error_type": error_type,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_similar_failures(self, error_msg: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar past failures"""
        # Simple keyword matching
        keywords = error_msg.lower().split()
        similar = []
        
        for pattern in self.failed_patterns:
            # Search in both instruction and error fields
            pattern_text = (str(pattern.get("instruction", "")) + " " + 
                          str(pattern.get("error", "")) + " " +
                          str(pattern.get("context", ""))).lower()
            if any(kw in pattern_text for kw in keywords if len(kw) > 3):
                similar.append(pattern)
        
        return similar[:top_k]

    # Backward-compatible helper used in tests
    def add_failure(self, instruction: str, error: str, actions: Optional[List[Dict[str, Any]]] = None):
        self.failed_patterns.append({
            "instruction": instruction,
            "error": error,
            "actions": actions or [],
            "timestamp": datetime.now().isoformat()
        })


# ============================================================================
# MULTI-STAGE PLANNER
# ============================================================================

class MultiStagePlanner:
    """
    Multi-stage planner implementing: Propose → Revise → Simulate → Execute
    """
    
    def __init__(self, llm_agent=None):
        """
        Initialize planner.
        
        Args:
            llm_agent: Optional LLM agent for intelligent planning
        """
        self.llm_agent = llm_agent
        self.history = PlanHistory()
    
    # ========================================================================
    # STAGE 1: PROPOSE
    # ========================================================================
    
    async def propose_plan(
        self,
        df: pd.DataFrame,
        instruction: str,
        df_context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Stage 1: Generate initial plan from instruction.
        
        Args:
            df: DataFrame to operate on
            instruction: User's natural language instruction
            df_context: Optional DataFrame context (columns, sample, etc.)
        
        Returns:
            ExecutionPlan with proposed actions
        """
        logger.info(f"📋 STAGE 1: PROPOSING plan for: {instruction[:60]}...")
        
        # Build DataFrame context
        if not df_context:
            df_context = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                "sample": df.head(2).to_dict(orient="records") if len(df) > 0 else []
            }
        
        # If LLM agent available, use it for intelligent planning
        if self.llm_agent:
            plan = await self._propose_plan_with_llm(df, instruction, df_context)
        else:
            # Fallback: simple heuristic-based planning
            plan = self._propose_plan_heuristic(df, instruction, df_context)
        
        # Add to history
        self.history.add_plan(plan)
        
        logger.info(f"✅ Proposed {len(plan.actions)} actions")
        return plan
    
    async def _propose_plan_with_llm(
        self,
        df: pd.DataFrame,
        instruction: str,
        df_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """Use LLM to propose intelligent plan"""
        
        # Build prompt for LLM
        prompt = f"""You are a spreadsheet operations planner. Generate a plan using structured actions.

=== CRITICAL CONSTRAINTS ===
- ASSUME: Orchestrator already classified this as 'transform' task type
- FOCUS: Safety validation and step decomposition ONLY
- DO NOT: Re-interpret user intent or suggest alternate routes
- DO NOT: Guess whether user wants analysis vs modification
- Your ONLY job: Break instruction into safe, executable steps

=== DATAFRAME CONTEXT ===
Shape: {df_context['shape']} (rows, columns)
Columns: {', '.join(df_context['columns'])}
Data types: {df_context['dtypes']}
Sample data: {df_context['sample']}

=== INSTRUCTION ===
{instruction}

=== AVAILABLE ACTIONS ===
1. filter: Filter rows by condition
   {{"action_type": "filter", "column": "Sales", "operator": ">", "value": 100}}

2. sort: Sort by column(s)
   {{"action_type": "sort", "columns": ["Date", "Sales"], "ascending": [True, False]}}

3. add_column: Add calculated column
   {{"action_type": "add_column", "new_column": "Total", "formula": "Quantity * Price"}}

4. rename_column: Rename columns
   {{"action_type": "rename_column", "mapping": {{"old": "new"}}}}

5. drop_column: Remove columns
   {{"action_type": "drop_column", "columns": ["TempCol"]}}

6. group_by: Group and aggregate
   {{"action_type": "group_by", "group_columns": ["Category"], "agg_column": "Sales", "agg_function": "sum"}}

7. fill_na: Fill missing values
   {{"action_type": "fill_na", "columns": ["Price"], "method": "mean"}}

8. drop_duplicates: Remove duplicates
   {{"action_type": "drop_duplicates", "subset": ["ID"], "keep": "first"}}

9. add_serial: Add serial number column
   {{"action_type": "add_serial", "column_name": "Sl.No.", "start": 1}}

10. append_summary_row: Append a summary row with aggregations
    {{"action_type": "append_summary_row", "aggregations": {{"Age": "mean", "Total Amount": "sum"}}}}

11. compare_files: Compare multiple spreadsheet files (requires 2+ files uploaded)
    {{"action_type": "compare_files", "file_ids": ["file1_id", "file2_id"], "comparison_mode": "schema_and_key", "key_columns": ["ID"]}}
    - comparison_mode: "schema_only", "schema_and_key", "full_diff"
    - Use when instruction mentions: "compare files", "find differences", "what changed between files", "diff files"

12. merge_files: Merge multiple spreadsheet files (requires 2+ files uploaded)
    {{"action_type": "merge_files", "file_ids": ["file1_id", "file2_id"], "merge_type": "join", "join_type": "inner", "key_columns": ["ID"]}}
    - merge_type: "join" (SQL-like), "union" (stack matching columns), "concat" (stack all)
    - join_type: "inner", "outer", "left", "right" (for join merge)
    - Use when instruction mentions: "merge files", "combine files", "join files", "union files", "concatenate files"

=== RESPONSE FORMAT (JSON) ===
{{
    "reasoning": "Step-by-step explanation of the plan",
    "actions": [
        {{"action_type": "...", ...}},
        {{"action_type": "...", ...}}
    ]
}}

=== RULES ===
- Use ONLY the action types listed above
- Reference actual column names from the DataFrame
- Break complex operations into multiple simple actions
- Validate column names exist
- Return valid JSON only
"""
        
        try:
            # Get LLM response
            from .llm_agent import query_agent
            
            providers = query_agent.providers
            if not providers:
                raise ValueError("No LLM providers available")
            
            # Try first provider
            provider = providers[0]
            client = provider["client"]
            model = provider["model"]
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            parsed = json.loads(response_text.strip())
            
            reasoning = parsed.get("reasoning", "")
            action_dicts = parsed.get("actions", [])
            
            # Parse actions
            actions = ActionParser.parse_multiple(action_dicts)
            
            # Create plan
            plan_id = f"plan-{int(time.time() * 1000)}"
            plan = ExecutionPlan(
                plan_id=plan_id,
                instruction=instruction,
                actions=actions,
                reasoning=reasoning,
                stage=PlanStage.PROPOSING
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"LLM planning failed: {e}, falling back to heuristic", exc_info=True)
            return self._propose_plan_heuristic(df, instruction, df_context)
    
    def _propose_plan_heuristic(
        self,
        df: pd.DataFrame,
        instruction: str,
        df_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """Fallback: Simple heuristic-based planning"""
        
        instruction_lower = instruction.lower()
        actions = []
        reasoning = "Heuristic-based plan: "
        
        # Detect common patterns
        if "serial" in instruction_lower or "sl.no" in instruction_lower or "sl no" in instruction_lower:
            actions.append(ActionParser.parse({
                "action_type": "add_serial",
                "column_name": "Sl.No.",
                "start": 1,
                "position": 0
            }))
            reasoning += "Add serial number. "

        # Append summary row (mean/sum) heuristic
        if ("append" in instruction_lower or "add" in instruction_lower) and "row" in instruction_lower and (
            "average" in instruction_lower or "avg" in instruction_lower or "total" in instruction_lower or "sum" in instruction_lower
        ):
            import re
            from difflib import get_close_matches

            def best_match_column(phrase: str) -> Optional[str]:
                phrase = re.sub(r"[^a-z0-9 _-]", " ", (phrase or "").lower()).strip()
                if not phrase:
                    return None
                cols = df.columns.tolist()
                lower_map = {str(c).lower(): c for c in cols}
                # Direct containment first
                for lc, orig in lower_map.items():
                    if lc and lc in phrase:
                        return orig
                # Fuzzy match
                matches = get_close_matches(phrase, list(lower_map.keys()), n=1, cutoff=0.55)
                return lower_map[matches[0]] if matches else None

            agg_map: Dict[str, str] = {}

            # Look for phrases like "average Age" / "avg Age"
            avg_phrases = re.findall(r"(?:average|avg)\s+([a-z0-9 _-]{1,50})", instruction_lower)
            for ph in avg_phrases:
                col = best_match_column(ph)
                if col:
                    agg_map[str(col)] = "mean"

            # Look for phrases like "total Total Amount" / "sum Total Amount"
            sum_phrases = re.findall(r"(?:total|sum)\s+([a-z0-9 _-]{1,50})", instruction_lower)
            for ph in sum_phrases:
                col = best_match_column(ph)
                if col:
                    agg_map[str(col)] = "sum"

            if agg_map:
                actions.append(ActionParser.parse({
                    "action_type": "append_summary_row",
                    "aggregations": agg_map,
                    "label_column": None,
                    "label_value": None,
                    "description": "Append a summary row with requested aggregates"
                }))
                reasoning += "Append a summary row with aggregates. "
        
        if "sort" in instruction_lower:
            # Try to find column name
            for col in df.columns:
                if col.lower() in instruction_lower:
                    actions.append(ActionParser.parse({
                        "action_type": "sort",
                        "columns": [col],
                        "ascending": "desc" not in instruction_lower
                    }))
                    reasoning += f"Sort by {col}. "
                    break
        
        if "filter" in instruction_lower or "where" in instruction_lower:
            reasoning += "Filter operation detected (manual action required). "
        
        if not actions:
            reasoning += "Could not parse instruction into actions. Manual intervention needed."
        
        plan_id = f"plan-{int(time.time() * 1000)}"
        plan = ExecutionPlan(
            plan_id=plan_id,
            instruction=instruction,
            actions=actions,
            reasoning=reasoning,
            stage=PlanStage.PROPOSING
        )
        
        return plan
    
    # ========================================================================
    # STAGE 2: REVISE
    # ========================================================================
    
    async def revise_plan(
        self,
        plan: ExecutionPlan,
        feedback: str,
        df: pd.DataFrame
    ) -> ExecutionPlan:
        """
        Stage 2: Revise plan based on feedback or errors.
        
        Args:
            plan: Original plan
            feedback: User feedback or error message
            df: DataFrame for validation
        
        Returns:
            Revised ExecutionPlan
        """
        logger.info(f"🔄 STAGE 2: REVISING plan based on feedback")
        
        plan.stage = PlanStage.REVISING
        plan.revisions.append({
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        })
        
        # If LLM available, use intelligent revision
        if self.llm_agent:
            revised_plan = await self._revise_plan_with_llm(plan, feedback, df)
        else:
            # Simple revision: just log feedback
            revised_plan = plan
            revised_plan.reasoning += f" [REVISED: {feedback}]"
        
        logger.info(f"✅ Plan revised")
        return revised_plan
    
    async def _revise_plan_with_llm(
        self,
        plan: ExecutionPlan,
        feedback: str,
        df: pd.DataFrame
    ) -> ExecutionPlan:
        """Use LLM to revise plan intelligently"""
        
        # Check for similar past failures
        similar_failures = self.history.get_similar_failures(feedback)
        
        prompt = f"""Revise the execution plan based on feedback.

=== ORIGINAL PLAN ===
Instruction: {plan.instruction}
Reasoning: {plan.reasoning}
Actions: {json.dumps(actions_to_dicts(plan.actions), indent=2)}

=== FEEDBACK ===
{feedback}

=== SIMILAR PAST FAILURES ===
{json.dumps(similar_failures, indent=2) if similar_failures else "None"}

=== DATAFRAME ===
Columns: {', '.join(df.columns.tolist())}

Generate a REVISED plan that addresses the feedback. Return JSON with 'reasoning' and 'actions' fields.
"""
        
        try:
            from .llm_agent import query_agent
            
            providers = query_agent.providers
            if not providers:
                raise ValueError("No LLM providers")
            
            provider = providers[0]
            response = provider["client"].chat.completions.create(
                model=provider["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            
            parsed = json.loads(response_text.strip())
            
            # Create revised plan
            revised_actions = ActionParser.parse_multiple(parsed.get("actions", []))
            
            plan.actions = revised_actions
            plan.reasoning = parsed.get("reasoning", plan.reasoning)
            plan.stage = PlanStage.REVISING
            
            return plan
            
        except Exception as e:
            logger.error(f"LLM revision failed: {e}")
            return plan
    
    # ========================================================================
    # STAGE 3: SIMULATE
    # ========================================================================
    
    def simulate_plan(
        self,
        plan: ExecutionPlan,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Stage 3: Simulate plan execution without modifying data.
        
        Args:
            plan: Plan to simulate
            df: DataFrame to simulate on
        
        Returns:
            Simulation result with preview and warnings
        """
        logger.info(f"🧪 STAGE 3: SIMULATING plan with {len(plan.actions)} actions")
        
        plan.stage = PlanStage.SIMULATING
        
        # Execute actions on copy
        current_df = df.copy()
        simulation_log = []
        all_warnings = []
        
        for i, action in enumerate(plan.actions):
            step_log = {
                "step": i + 1,
                "action_type": action.action_type,
                "description": action.description,
                "before_shape": current_df.shape
            }
            
            # Validate
            validation_error = action.validate_against_df(current_df)
            if validation_error:
                step_log["validation_error"] = validation_error
                step_log["success"] = False
                simulation_log.append(step_log)
                break
            
            # Execute
            result_df, error = ActionExecutor.execute_action(action, current_df, validate=False)
            
            if error:
                step_log["error"] = error
                step_log["success"] = False
                simulation_log.append(step_log)
                break
            
            current_df = result_df
            step_log["after_shape"] = current_df.shape
            step_log["success"] = True
            
            # Check for warnings
            from .simulate import detect_potential_issues
            warnings = detect_potential_issues(df, current_df)
            if warnings:
                step_log["warnings"] = warnings
                all_warnings.extend(warnings)
            
            simulation_log.append(step_log)
        
        # Build result
        simulation_result = {
            "success": all([s.get("success", False) for s in simulation_log]) if simulation_log else False,
            "preview": {
                "shape": current_df.shape,
                "columns": current_df.columns.tolist(),
                "rows": current_df.head(10).to_dict(orient="records") if len(current_df) > 0 else []
            },
            "observation": {
                "before_shape": df.shape,
                "after_shape": current_df.shape,
                "changes_summary": f"Rows: {df.shape[0]} -> {current_df.shape[0]}, Cols: {df.shape[1]} -> {current_df.shape[1]}"
            },
            "simulation_log": simulation_log,
            "warnings": all_warnings
        }
        
        plan.simulation_result = simulation_result
        
        logger.info(f"✅ Simulation {'successful' if simulation_result['success'] else 'failed'}")
        return simulation_result
    
    # ========================================================================
    # STAGE 4: EXECUTE
    # ========================================================================
    
    def execute_plan(
        self,
        plan: ExecutionPlan,
        df: pd.DataFrame,
        force: bool = False
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Stage 4: Execute plan on actual DataFrame.
        
        Args:
            plan: Plan to execute
            df: DataFrame to modify
            force: Execute even if simulation failed
        
        Returns:
            (modified_df, execution_result)
        """
        logger.info(f"⚡ STAGE 4: EXECUTING plan")
        
        # Check if simulation passed
        if plan.simulation_result and not plan.simulation_result.get("success") and not force:
            error_msg = "Simulation failed. Use force=True to execute anyway."
            logger.warning(f"⚠️ {error_msg}")
            return df, {"success": False, "error": error_msg}
        
        plan.stage = PlanStage.EXECUTING
        
        # Execute actions
        result_df, execution_log = ActionExecutor.execute_actions(
            plan.actions,
            df,
            stop_on_error=True
        )
        
        # Check if all succeeded
        all_success = all(step.get("success", False) for step in execution_log)
        
        if all_success:
            plan.stage = PlanStage.COMPLETED
        else:
            plan.stage = PlanStage.FAILED
            # Record failure pattern
            failed_step = next((s for s in execution_log if not s.get("success")), None)
            if failed_step:
                self.history.record_failure_pattern(
                    error_type=failed_step.get("action_type", "unknown"),
                    context={"instruction": plan.instruction, "error": failed_step.get("error")}
                )
        
        execution_result = {
            "success": all_success,
            "actions_executed": len([s for s in execution_log if s.get("success")]),
            "final_shape": result_df.shape,
            "success": all_success,
            "execution_log": execution_log,
            "final_shape": result_df.shape
        }
        
        plan.execution_result = execution_result
        
        logger.info(f"{'✅' if all_success else '❌'} Execution {'completed' if all_success else 'failed'}")
        return result_df, execution_result


# Global planner instance
planner = MultiStagePlanner()
