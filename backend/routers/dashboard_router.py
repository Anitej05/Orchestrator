"""
Dashboard Router - Handles dashboard metrics and analytics endpoints.

Extracted from main.py to improve code organization and maintainability.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from database import SessionLocal
from models import UserThread, Workflow, Agent, StatusEnum

router = APIRouter(prefix="/api/metrics", tags=["Dashboard"])
logger = logging.getLogger("uvicorn.error")

CONVERSATION_HISTORY_DIR = "conversation_history"


# --- Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/dashboard")
async def get_dashboard_metrics(request: Request, db: Session = Depends(get_db)):
    """Get comprehensive dashboard metrics for the current user"""
    try:
        user_id = request.headers.get("X-User-ID")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not provided")
        
        # Get counts
        conversation_count = db.query(UserThread).filter(UserThread.user_id == user_id).count()
        workflow_count = db.query(Workflow).filter(Workflow.user_id == user_id).count()
        agent_count = db.query(Agent).filter(Agent.status == StatusEnum.active).count()
        
        # Time periods
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        yesterday = now - timedelta(days=1)
        
        # Recent activity
        recent_activity = db.query(UserThread).filter(
            UserThread.user_id == user_id,
            UserThread.created_at >= yesterday
        ).count()
        
        # Conversation trend (last 7 days)
        conversation_trend = []
        for i in range(6, -1, -1):
            date = now - timedelta(days=i)
            count = db.query(UserThread).filter(
                UserThread.user_id == user_id,
                UserThread.created_at >= date,
                UserThread.created_at < date + timedelta(days=1)
            ).count()
            conversation_trend.append({"date": date.strftime('%b %d'), "count": count})
        
        # Workflow status distribution
        active_workflows = db.query(Workflow).filter(Workflow.user_id == user_id, Workflow.status == 'active').count()
        archived_workflows = db.query(Workflow).filter(Workflow.user_id == user_id, Workflow.status == 'archived').count()
        
        workflow_status = []
        if active_workflows > 0:
            workflow_status.append({"name": "Active", "value": active_workflows})
        if archived_workflows > 0:
            workflow_status.append({"name": "Archived", "value": archived_workflows})
        
        # Recent conversations
        recent_conversations = db.query(UserThread).filter(UserThread.user_id == user_id).order_by(UserThread.updated_at.desc()).limit(5).all()
        recent_conv_list = [
            {
                "id": conv.thread_id,
                "title": conv.title or "Untitled Conversation",
                "date": conv.created_at.strftime('%Y-%m-%d'),
                "status": "completed"
            }
            for conv in recent_conversations
        ]
        
        # Cost and performance metrics
        cost_today = 0.0
        cost_week = 0.0
        cost_month = 0.0
        total_cost = 0.0
        total_tasks = 0
        successful_tasks = 0
        failed_tasks = 0
        agent_usage = {}
        agent_costs = {}
        hourly_usage = [0] * 24
        
        all_conversations = db.query(UserThread).filter(UserThread.user_id == user_id).all()
        
        for conv in all_conversations:
            history_file = os.path.join(CONVERSATION_HISTORY_DIR, f"{conv.thread_id}.json")
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                        task_pairs = history_data.get('task_agent_pairs', [])
                        
                        for pair in task_pairs:
                            agent_name = pair.get('primary', {}).get('name', 'Unknown')
                            agent_id = pair.get('primary', {}).get('id')
                            
                            if agent_id:
                                agent = db.query(Agent).filter(Agent.id == agent_id).first()
                                if agent:
                                    cost = agent.price_per_call_usd
                                    total_cost += cost
                                    
                                    if agent_name not in agent_usage:
                                        agent_usage[agent_name] = 0
                                        agent_costs[agent_name] = 0.0
                                    agent_usage[agent_name] += 1
                                    agent_costs[agent_name] += cost
                                    
                                    if conv.created_at >= today_start:
                                        cost_today += cost
                                    if conv.created_at >= week_start:
                                        cost_week += cost
                                    if conv.created_at >= month_start:
                                        cost_month += cost
                            
                            total_tasks += 1
                        
                        if history_data.get('final_response'):
                            successful_tasks += len(task_pairs)
                        
                        hour = conv.created_at.hour
                        hourly_usage[hour] += 1
                        
                except Exception as e:
                    logger.warning(f"Could not parse history file {history_file}: {e}")
        
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        avg_response_time = 2.5  # placeholder
        
        top_agents = sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        top_agents_list = [
            {"name": name, "calls": calls, "cost": agent_costs.get(name, 0.0), "cost_per_call": agent_costs.get(name, 0.0) / calls if calls > 0 else 0}
            for name, calls in top_agents
        ]
        
        hourly_pattern = [{"hour": f"{i:02d}:00", "count": hourly_usage[i]} for i in range(24)]
        
        # Cost trend
        cost_trend = []
        for i in range(6, -1, -1):
            date = now - timedelta(days=i)
            day_cost = 0.0
            day_conversations = db.query(UserThread).filter(
                UserThread.user_id == user_id,
                UserThread.created_at >= date,
                UserThread.created_at < date + timedelta(days=1)
            ).all()
            
            for conv in day_conversations:
                history_file = os.path.join(CONVERSATION_HISTORY_DIR, f"{conv.thread_id}.json")
                if os.path.exists(history_file):
                    try:
                        with open(history_file, 'r') as f:
                            history_data = json.load(f)
                            task_pairs = history_data.get('task_agent_pairs', [])
                            for pair in task_pairs:
                                agent_id = pair.get('primary', {}).get('id')
                                if agent_id:
                                    agent = db.query(Agent).filter(Agent.id == agent_id).first()
                                    if agent:
                                        day_cost += agent.price_per_call_usd
                    except:
                        pass
            
            cost_trend.append({"date": date.strftime('%b %d'), "cost": round(day_cost, 4)})
        
        return {
            "total_conversations": conversation_count,
            "total_workflows": workflow_count,
            "total_agents": agent_count,
            "recent_activity": recent_activity,
            "conversation_trend": conversation_trend,
            "workflow_status": workflow_status,
            "recent_conversations": recent_conv_list,
            "cost_metrics": {
                "today": round(cost_today, 4),
                "week": round(cost_week, 4),
                "month": round(cost_month, 4),
                "total": round(total_cost, 4),
                "avg_per_conversation": round(total_cost / conversation_count, 4) if conversation_count > 0 else 0
            },
            "cost_trend": cost_trend,
            "performance_metrics": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": round(success_rate, 2),
                "avg_response_time": avg_response_time,
                "avg_tasks_per_conversation": round(total_tasks / conversation_count, 2) if conversation_count > 0 else 0
            },
            "top_agents": top_agents_list,
            "hourly_usage": hourly_pattern
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")
