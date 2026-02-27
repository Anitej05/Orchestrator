"""
End-to-End Orchestrator Stress Test v3 — Multi-Agent
Clean workspace, fresh thread, prompt designed to naturally involve multiple agents.
"""

import asyncio
import json
import time
import uuid
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


async def run_stress_test():
    import websockets
    
    thread_id = str(uuid.uuid4())
    
    # This prompt is designed to naturally involve:
    # - Browser Agent: "go to Wikipedia and extract..."
    # - Spreadsheet Agent: "create an Excel spreadsheet"
    # - Document Agent: "create a Word document"
    # - Coding Agent / Python: analysis and charting
    # - Tools: web search for news
    PROMPT = """I need help with a research project about the world's largest tech companies.

1. Go to the Wikipedia page for "List of largest technology companies by revenue" and extract the revenue data for the top 10 companies.

2. Search the web for the latest news about any major acquisitions or partnerships these companies announced this month.

3. Put all the revenue data into a properly formatted Excel spreadsheet with columns for Company, Revenue, Industry, and Headquarters.

4. Write a Python analysis script that reads the data and creates a bar chart comparing the revenues. Save the chart.

5. Compile everything into a professional Word document report with an executive summary, the data table, and key findings.

Give me all the deliverables when done."""

    print(f"{'='*80}")
    print(f"  ORBIMESH ORCHESTRATOR - MULTI-AGENT STRESS TEST v3")
    print(f"{'='*80}")
    print(f"  Thread ID:     {thread_id}")
    print(f"  Prompt:        {len(PROMPT)} chars")
    print(f"  Started:       {time.strftime('%H:%M:%S')}")
    print(f"{'='*80}\n")
    
    uri = "ws://localhost:8000/ws/chat"
    
    start_time = time.time()
    agents_invoked = set()
    tools_used = set()
    action_log = []
    errors = []
    iteration = 0
    
    final_response = None
    try:
        async with websockets.connect(uri, ping_interval=30, ping_timeout=120, max_size=None, open_timeout=30) as ws:
            message = {
                "prompt": PROMPT,
                "thread_id": thread_id,
                "owner": {"user_id": "stress_test_user", "sub": "stress_test_user"},
                "planning_mode": False,
            }
            
            print("[SEND] Prompt sent to orchestrator.\n")
            await ws.send(json.dumps(message))
            
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=600)  # 10 min timeout
                    msg = json.loads(raw)
                    
                    node = msg.get("node", "unknown")
                    data = msg.get("data", {})
                    elapsed = time.time() - start_time
                    
                    if not isinstance(data, dict):
                        data = {}

                    # ---- BRAIN DECISION ----
                    if node == "omni_brain":
                        decision = data.get("decision", {})
                        if not isinstance(decision, dict):
                            decision = {}
                        action_type = decision.get("action_type", "")
                        resource_id = decision.get("resource_id", "") or ""
                        reasoning = decision.get("reasoning", "") or ""
                        plan = decision.get("execution_plan")
                        parallel = decision.get("parallel_actions")
                        user_resp = decision.get("user_response", "")
                        payload = decision.get("payload", {}) or {}
                        code = payload.get("code", "") if isinstance(payload, dict) else ""
                        phase_complete = decision.get("phase_complete", False)
                        
                        # Track agents and tools
                        if action_type == "agent" and resource_id:
                            agents_invoked.add(resource_id)
                        elif action_type == "tool" and resource_id:
                            tools_used.add(resource_id)
                        if parallel and isinstance(parallel, list):
                            for pa in parallel:
                                if isinstance(pa, dict):
                                    pa_type = pa.get("action_type", "")
                                    pa_res = pa.get("resource_id", "")
                                    if pa_type == "tool" and pa_res:
                                        tools_used.add(pa_res)
                                    elif pa_type == "agent" and pa_res:
                                        agents_invoked.add(pa_res)

                        iteration += 1
                        label = f"{action_type}:{resource_id}" if resource_id else action_type
                        
                        print(f"  [{elapsed:6.1f}s]  BRAIN  [{iteration}] --> {label}")
                        if reasoning:
                            print(f"           Reason: {reasoning[:160]}")
                        if plan:
                            print(f"           Plan: {len(plan)} phases")
                            for p in plan:
                                if isinstance(p, dict):
                                    print(f"             Phase {p.get('phase_id','?')}: {p.get('name','?')} -- {p.get('goal','')[:80]}")
                        if parallel:
                            print(f"           Parallel: {len(parallel)} actions")
                            for i, pa in enumerate(parallel):
                                if isinstance(pa, dict):
                                    pa_label = f"{pa.get('action_type','')}:{pa.get('resource_id','')}"
                                    pa_payload = pa.get("payload", {})
                                    snippet = ""
                                    if isinstance(pa_payload, dict):
                                        snippet = pa_payload.get("query", pa_payload.get("prompt", pa_payload.get("instruction", pa_payload.get("url", ""))))
                                    print(f"             [{i}] {pa_label} -- {str(snippet)[:80]}")
                        if action_type == "python" and code:
                            print(f"           Code: {code[:120].replace(chr(10), ' | ')}...")
                        if action_type == "terminal":
                            cmd = payload.get("command", "")
                            print(f"           Command: {cmd[:100]}")
                        if action_type == "agent":
                            prompt = payload.get("prompt", payload.get("instruction", ""))
                            print(f"           Prompt: {str(prompt)[:120]}")
                        if phase_complete:
                            verified = decision.get("phase_goal_verified", "")
                            print(f"           ** PHASE COMPLETE: {str(verified)[:100]}")
                        if action_type == "finish" and user_resp:
                            final_response = user_resp
                            print(f"           Final response ready ({len(user_resp)} chars)")
                        print()

                    # ---- HANDS EXECUTION RESULT ----
                    elif node == "omni_hands":
                        exec_result = data.get("execution_result", {})
                        if not isinstance(exec_result, dict):
                            exec_result = {}
                        success = exec_result.get("success", False)
                        action_id = exec_result.get("action_id", "")
                        output = exec_result.get("output")
                        error_msg = exec_result.get("error_message", "")
                        
                        summary = ""
                        if isinstance(output, dict):
                            par_results = output.get("parallel_results")
                            if par_results and isinstance(par_results, list):
                                summaries = []
                                for pr in par_results:
                                    if isinstance(pr, dict):
                                        pr_success = pr.get("success", False)
                                        pr_res = pr.get("resource_id", "")
                                        tag = "OK" if pr_success else "FAIL"
                                        summaries.append(f"[{tag}:{pr_res}]")
                                summary = " ".join(summaries)
                            else:
                                result_data = output.get("result", output)
                                if isinstance(result_data, dict):
                                    summary = result_data.get("result_summary", str(result_data)[:150])
                                else:
                                    summary = str(result_data)[:150]
                        elif isinstance(output, str):
                            summary = output[:150]
                        
                        icon = "  OK  " if success else " FAIL "
                        print(f"  [{elapsed:6.1f}s]  HANDS  [{icon}] {action_id}")
                        if summary:
                            print(f"           Result: {str(summary)[:200]}")
                        if error_msg:
                            print(f"           Error: {error_msg[:150]}")
                            errors.append(f"Hands:{action_id}: {error_msg[:100]}")
                        
                        action_log.append((round(elapsed,1), action_id, success, str(summary)[:80]))
                        print()

                    # ---- TODO LIST ----
                    elif node == "todo_list_update":
                        todo = data.get("todo_list", [])
                        if todo:
                            statuses = {}
                            for t in todo:
                                if isinstance(t, dict):
                                    s = t.get("status", "?")
                                    statuses[s] = statuses.get(s, 0) + 1
                            status_str = ", ".join(f"{v} {k}" for k, v in statuses.items())
                            print(f"  [{elapsed:6.1f}s]  TODO   {len(todo)} tasks ({status_str})")

                    # ---- TASK EVENTS ----
                    elif node == "task_started":
                        task_name = data.get("task_name", msg.get("task_name", "?"))
                        agent = data.get("agent_name", msg.get("agent_name", "?"))
                        agents_invoked.add(agent)
                        print(f"  [{elapsed:6.1f}s]  TASK>> {task_name} -> {agent}")
                    elif node == "task_completed":
                        task_name = data.get("task_name", msg.get("task_name", "?"))
                        exec_time = data.get("execution_time", msg.get("execution_time", 0))
                        print(f"  [{elapsed:6.1f}s]  TASK OK {task_name} ({exec_time:.1f}s)")
                    elif node == "task_failed":
                        task_name = data.get("task_name", msg.get("task_name", "?"))
                        error = str(data.get("error", msg.get("error", "?")))[:120]
                        print(f"  [{elapsed:6.1f}s]  TASK!! {task_name} - {error}")
                        errors.append(f"TaskFail:{task_name}: {error}")

                    # ---- OTHER ----
                    elif node == "__start__":
                        print(f"  [{elapsed:6.1f}s]  START  Orchestration initialized\n")
                    elif node == "__end__":
                        fr = data.get("final_response", "")
                        if fr:
                            final_response = fr
                        print(f"  [{elapsed:6.1f}s]  END    Orchestration complete")
                        break
                    elif node == "__error__":
                        error_msg = msg.get("error", "Unknown error")
                        print(f"  [{elapsed:6.1f}s]  ERROR  {error_msg[:200]}")
                        errors.append(f"WS Error: {error_msg[:100]}")
                        break
                    elif node == "__user_input_required__":
                        q = data.get("question_for_user", "?")
                        print(f"  [{elapsed:6.1f}s]  INPUT? {q[:150]}")
                        break
                    elif node == "action_approval_required":
                        reason = data.get("approval_reason", "?")
                        print(f"  [{elapsed:6.1f}s]  APPROVE? {reason[:150]}")
                        break
                    elif node == "workflow_complete":
                        print(f"  [{elapsed:6.1f}s]  WORKFLOW COMPLETE")
                    else:
                        print(f"  [{elapsed:6.1f}s]  {node:8s}")

                except asyncio.TimeoutError:
                    print(f"\n  TIMEOUT after 10 minutes!")
                    errors.append("Timeout after 10 minutes")
                    break
                except Exception as recv_err:
                    print(f"\n  RECV ERROR: {recv_err}")
                    errors.append(str(recv_err))
                    break
    except Exception as conn_err:
        print(f"\n  CONNECTION ERROR: {conn_err}")
        errors.append(str(conn_err))
    
    total_time = time.time() - start_time
    
    # ========================================================================
    #  SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"  TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Total Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Brain Iterations:   {iteration}")
    print(f"  Agents Invoked:     {agents_invoked or 'None'}")
    print(f"  Tools Used:         {tools_used or 'None'}")
    print(f"  Errors:             {len(errors)}")
    for e in errors:
        print(f"    - {e}")
    has_resp = bool(final_response)
    print(f"  Final Response:     {'Yes (' + str(len(final_response)) + ' chars)' if has_resp else 'No'}")
    print(f"{'='*80}")

    if final_response:
        print(f"\n{'='*80}")
        print(f"  FINAL RESPONSE (first 1500 chars)")
        print(f"{'='*80}")
        print(final_response[:1500])
        if len(final_response) > 1500:
            print(f"\n  ... ({len(final_response) - 1500} more chars)")
        print(f"{'='*80}")

    print(f"\n  ACTION LOG:")
    print(f"  {'Time':>7s}  {'Action':<35s}  {'OK?':<6s}  Summary")
    print(f"  {'-'*7}  {'-'*35}  {'-'*6}  {'-'*40}")
    for t, action, ok, summary in action_log:
        print(f"  {t:6.1f}s  {action:<35s}  {'YES' if ok else 'NO':<6s}  {summary[:50]}")
    print()

    return has_resp and len(errors) == 0


if __name__ == "__main__":
    result = asyncio.run(run_stress_test())
    
    print(f"{'='*80}")
    if result:
        print(f"  TEST PASSED")
    else:
        print(f"  TEST COMPLETED WITH ISSUES")
    print(f"{'='*80}")
