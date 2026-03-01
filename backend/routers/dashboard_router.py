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

from database import get_db
from models import UserThread, Workflow, Agent, StatusEnum, LLMTelemetryRecord, AgentExecutionRecord

router = APIRouter(prefix="/api/metrics", tags=["Dashboard"])
logger = logging.getLogger("uvicorn.error")

_DASHBOARD_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERSATION_HISTORY_DIR = os.path.join(_DASHBOARD_BACKEND_DIR, "conversation_history")


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
        
        # Cost and performance metrics fetched from SQLAlchemy DB natively
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
        
        llm_records = db.query(LLMTelemetryRecord).filter(LLMTelemetryRecord.user_id == user_id).all()
        agent_records = db.query(AgentExecutionRecord).filter(AgentExecutionRecord.user_id == user_id).all()
        all_conversations = db.query(UserThread).filter(UserThread.user_id == user_id).all()
        
        for record in llm_records:
            total_cost += record.cost_usd
            
            if record.timestamp >= today_start:
                cost_today += record.cost_usd
            if record.timestamp >= week_start:
                cost_week += record.cost_usd
            if record.timestamp >= month_start:
                cost_month += record.cost_usd
                
            agent_name = record.agent_name or "Unknown"
            if agent_name not in agent_costs:
                agent_costs[agent_name] = 0.0
            agent_costs[agent_name] += record.cost_usd

        for agent_rec in agent_records:
            agent_name = agent_rec.agent_name or "Unknown"
            total_tasks += 1
            if agent_rec.success:
                successful_tasks += 1
            else:
                failed_tasks += 1
                
            if agent_name not in agent_usage:
                agent_usage[agent_name] = 0
                if agent_name not in agent_costs:
                    agent_costs[agent_name] = 0.0
            agent_usage[agent_name] += 1
            
        for conv in all_conversations:
            hour = conv.created_at.hour
            hourly_usage[hour] += 1
        
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
            day_end = date + timedelta(days=1)
            day_cost = sum(r.cost_usd for r in llm_records if date <= r.timestamp < day_end)
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
