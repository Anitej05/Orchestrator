#!/usr/bin/env python3
"""
Database Migration Script: Populate plan_graph for existing workflows
and initialize analytics tables

This script:
1. Reads all workflow JSON files from conversation_history/
2. Extracts plan_graph from each conversation
3. Updates existing workflows in database with plan_graph
4. Initializes conversation analytics for tracked conversations
5. Initializes plan history records for past executions
6. Generates system tags and initializes tagging system

Usage:
    python migrate_workflows_plan_graph.py
    python migrate_workflows_plan_graph.py --db-url postgresql://user:pass@localhost/dbname
    python migrate_workflows_plan_graph.py --dry-run
"""

import os
import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine, text, func
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    logger.error("SQLAlchemy not installed. Run: pip install sqlalchemy")
    sys.exit(1)

# Import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from models import (
        Base, UserThread, Workflow, ConversationPlan, ConversationAnalytics,
        AgentUsageAnalytics, UserActivitySummary, ConversationTag,
        ConversationTagAssignment
    )
except ImportError as e:
    logger.error(f"Failed to import models: {e}")
    logger.info("Make sure models.py exists and SQLAlchemy models are properly defined")
    sys.exit(1)


class WorkflowMigration:
    """Handles database migration for plan graphs and analytics"""
    
    def __init__(self, db_url: Optional[str] = None, dry_run: bool = False):
        """
        Initialize migration handler
        
        Args:
            db_url: Database URL (uses DATABASE_URL env var if not provided)
            dry_run: If True, prints changes without committing to database
        """
        self.dry_run = dry_run
        self.stats = {
            'total_conversations': 0,
            'workflows_updated': 0,
            'plans_migrated': 0,
            'analytics_created': 0,
            'tags_created': 0,
            'errors': 0
        }
        
        # Get database URL
        if not db_url:
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                logger.error("DATABASE_URL environment variable not set")
                sys.exit(1)
        
        # Create database engine
        try:
            self.engine = create_engine(db_url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info(f"Connected to database: {db_url.split('@')[-1] if '@' in db_url else 'local'}")
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
    
    def get_conversation_history_path(self) -> Path:
        """Get path to conversation_history directory"""
        current_dir = Path(__file__).parent.parent
        history_path = current_dir / 'conversation_history'
        
        if not history_path.exists():
            logger.warning(f"Conversation history directory not found: {history_path}")
            return None
        
        return history_path
    
    def load_conversation_json(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load a single conversation JSON file"""
        history_path = self.get_conversation_history_path()
        if not history_path:
            return None
        
        file_path = history_path / f"{thread_id}.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversation {thread_id}: {e}")
            self.stats['errors'] += 1
        
        return None
    
    def extract_plan_graph(self, conversation_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract plan_graph from conversation data"""
        if not conversation_data:
            return None
        
        # Direct plan_graph field
        if 'plan_graph' in conversation_data:
            return conversation_data['plan_graph']
        
        # Nested in plan object
        if 'plan' in conversation_data and isinstance(conversation_data['plan'], dict):
            if 'graph' in conversation_data['plan']:
                return conversation_data['plan']['graph']
        
        # Nested in task_plan
        if 'task_plan' in conversation_data and isinstance(conversation_data['task_plan'], dict):
            if 'graph' in conversation_data['task_plan']:
                return conversation_data['task_plan']['graph']
        
        return None
    
    def migrate_existing_workflows(self, session: Session) -> int:
        """
        Migrate existing workflows with plan_graph data
        
        Returns:
            Number of workflows updated
        """
        logger.info("=" * 60)
        logger.info("MIGRATING EXISTING WORKFLOWS")
        logger.info("=" * 60)
        
        updated_count = 0
        
        try:
            # Query all active workflows
            workflows = session.query(Workflow).filter_by(status='active').all()
            logger.info(f"Found {len(workflows)} active workflows")
            
            for workflow in workflows:
                try:
                    # Load conversation data
                    conversation = self.load_conversation_json(workflow.workflow_id)
                    if not conversation:
                        continue
                    
                    # Extract plan_graph
                    plan_graph = self.extract_plan_graph(conversation)
                    
                    if plan_graph and not workflow.plan_graph:
                        if not self.dry_run:
                            workflow.plan_graph = plan_graph
                            session.add(workflow)
                        
                        updated_count += 1
                        logger.info(f"  ✓ Workflow {workflow.workflow_id}: plan_graph added")
                    
                except Exception as e:
                    logger.warning(f"  ✗ Error processing workflow {workflow.workflow_id}: {e}")
                    self.stats['errors'] += 1
            
            if not self.dry_run:
                session.commit()
                logger.info(f"✅ Updated {updated_count} workflows")
            else:
                logger.info(f"[DRY RUN] Would update {updated_count} workflows")
            
            self.stats['workflows_updated'] = updated_count
            return updated_count
        
        except Exception as e:
            logger.error(f"Error migrating workflows: {e}")
            session.rollback()
            self.stats['errors'] += 1
            return 0
    
    def migrate_plan_history(self, session: Session) -> int:
        """
        Create plan history records from conversation data
        
        Returns:
            Number of plan history records created
        """
        logger.info("=" * 60)
        logger.info("CREATING PLAN HISTORY")
        logger.info("=" * 60)
        
        created_count = 0
        history_path = self.get_conversation_history_path()
        
        if not history_path:
            logger.warning("Skipping plan history migration - conversation_history not found")
            return 0
        
        try:
            # Find all conversation JSON files
            json_files = list(history_path.glob('*.json'))
            logger.info(f"Found {len(json_files)} conversation files")
            
            for json_file in json_files:
                try:
                    thread_id = json_file.stem
                    conversation = self.load_conversation_json(thread_id)
                    
                    if not conversation:
                        continue
                    
                    # Check if user_thread exists
                    user_thread = session.query(UserThread).filter_by(thread_id=thread_id).first()
                    if not user_thread:
                        continue
                    
                    # Extract plan data
                    task_agent_pairs = conversation.get('task_agent_pairs', {})
                    task_plan = conversation.get('task_plan', {})
                    plan_graph = self.extract_plan_graph(conversation)
                    
                    if task_agent_pairs or task_plan:
                        plan_id = str(uuid.uuid4())
                        plan_status = conversation.get('execution_status', 'completed')
                        
                        # Determine status based on conversation state
                        if plan_status == 'running':
                            status = 'executing'
                        elif plan_status == 'completed':
                            status = 'completed'
                        elif plan_status == 'error':
                            status = 'failed'
                        else:
                            status = 'draft'
                        
                        plan = ConversationPlan(
                            plan_id=plan_id,
                            thread_id=thread_id,
                            user_id=user_thread.user_id,
                            plan_version=1,
                            task_agent_pairs=task_agent_pairs,
                            task_plan=task_plan,
                            plan_graph=plan_graph,
                            status=status,
                            result=conversation.get('task_statuses'),
                            error_message=conversation.get('error'),
                            created_at=user_thread.created_at,
                            updated_at=user_thread.updated_at
                        )
                        
                        if not self.dry_run:
                            session.add(plan)
                        
                        created_count += 1
                        logger.info(f"  ✓ Plan created for conversation {thread_id}")
                
                except Exception as e:
                    logger.warning(f"  ✗ Error processing conversation {json_file.name}: {e}")
                    self.stats['errors'] += 1
            
            if not self.dry_run:
                session.commit()
                logger.info(f"✅ Created {created_count} plan history records")
            else:
                logger.info(f"[DRY RUN] Would create {created_count} plan history records")
            
            self.stats['plans_migrated'] = created_count
            return created_count
        
        except Exception as e:
            logger.error(f"Error migrating plan history: {e}")
            session.rollback()
            self.stats['errors'] += 1
            return 0
    
    def initialize_conversation_analytics(self, session: Session) -> int:
        """
        Initialize analytics for all conversations
        
        Returns:
            Number of analytics records created
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING CONVERSATION ANALYTICS")
        logger.info("=" * 60)
        
        created_count = 0
        history_path = self.get_conversation_history_path()
        
        if not history_path:
            logger.warning("Skipping conversation analytics - conversation_history not found")
            return 0
        
        try:
            json_files = list(history_path.glob('*.json'))
            
            for json_file in json_files:
                try:
                    thread_id = json_file.stem
                    conversation = self.load_conversation_json(thread_id)
                    
                    if not conversation:
                        continue
                    
                    user_thread = session.query(UserThread).filter_by(thread_id=thread_id).first()
                    if not user_thread:
                        continue
                    
                    # Check if analytics already exist
                    existing = session.query(ConversationAnalytics).filter_by(thread_id=thread_id).first()
                    if existing:
                        continue
                    
                    # Extract metrics
                    messages = conversation.get('messages', [])
                    total_messages = len(messages)
                    
                    task_agent_pairs = conversation.get('task_agent_pairs', {})
                    total_agents = len(task_agent_pairs)
                    
                    has_plan = bool(conversation.get('task_plan') or conversation.get('plan'))
                    
                    # Calculate timing
                    message_times = []
                    for msg in messages:
                        if 'timestamp' in msg:
                            try:
                                message_times.append(datetime.fromisoformat(msg['timestamp']))
                            except:
                                pass
                    
                    conversation_duration = 0
                    if len(message_times) > 1:
                        conversation_duration = int((message_times[-1] - message_times[0]).total_seconds())
                    
                    analytics = ConversationAnalytics(
                        thread_id=thread_id,
                        user_id=user_thread.user_id,
                        total_messages=total_messages,
                        total_agents_used=total_agents,
                        plan_attempts=1 if has_plan else 0,
                        successful_plans=1 if (has_plan and conversation.get('execution_status') == 'completed') else 0,
                        failed_executions=1 if conversation.get('execution_status') == 'error' else 0,
                        conversation_duration_seconds=conversation_duration,
                        created_at=user_thread.created_at,
                        updated_at=user_thread.updated_at
                    )
                    
                    if not self.dry_run:
                        session.add(analytics)
                    
                    created_count += 1
                    logger.info(f"  ✓ Analytics initialized for {thread_id}")
                
                except Exception as e:
                    logger.warning(f"  ✗ Error initializing analytics for {json_file.name}: {e}")
                    self.stats['errors'] += 1
            
            if not self.dry_run:
                session.commit()
                logger.info(f"✅ Created {created_count} analytics records")
            else:
                logger.info(f"[DRY RUN] Would create {created_count} analytics records")
            
            self.stats['analytics_created'] = created_count
            return created_count
        
        except Exception as e:
            logger.error(f"Error initializing conversation analytics: {e}")
            session.rollback()
            self.stats['errors'] += 1
            return 0
    
    def initialize_system_tags(self, session: Session) -> int:
        """
        Create system tags for conversation organization
        
        Returns:
            Number of tags created
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING SYSTEM TAGS")
        logger.info("=" * 60)
        
        created_count = 0
        
        system_tags = [
            {'name': 'Urgent', 'color': '#EF4444', 'description': 'Requires immediate attention'},
            {'name': 'Completed', 'color': '#10B981', 'description': 'Conversation completed successfully'},
            {'name': 'Failed', 'color': '#DC2626', 'description': 'Execution failed or had errors'},
            {'name': 'In Progress', 'color': '#F59E0B', 'description': 'Conversation execution is running'},
            {'name': 'Important', 'color': '#8B5CF6', 'description': 'Mark as important'},
            {'name': 'Bug Report', 'color': '#6366F1', 'description': 'Bug or issue report'},
            {'name': 'Feature Request', 'color': '#06B6D4', 'description': 'Feature request or enhancement'},
            {'name': 'Documentation', 'color': '#14B8A6', 'description': 'Documentation and how-to guides'},
        ]
        
        try:
            # Get all unique users
            users = session.query(func.distinct(UserThread.user_id)).all()
            user_ids = [u[0] for u in users]
            
            logger.info(f"Found {len(user_ids)} unique users")
            
            for user_id in user_ids:
                for tag_def in system_tags:
                    try:
                        # Check if tag already exists
                        existing = session.query(ConversationTag).filter_by(
                            user_id=user_id,
                            tag_name=tag_def['name']
                        ).first()
                        
                        if existing:
                            continue
                        
                        tag = ConversationTag(
                            tag_id=str(uuid.uuid4()),
                            user_id=user_id,
                            tag_name=tag_def['name'],
                            tag_color=tag_def['color'],
                            tag_description=tag_def['description'],
                            is_system=True
                        )
                        
                        if not self.dry_run:
                            session.add(tag)
                        
                        created_count += 1
                    
                    except Exception as e:
                        logger.warning(f"  ✗ Error creating tag {tag_def['name']} for user {user_id}: {e}")
                        self.stats['errors'] += 1
            
            if not self.dry_run:
                session.commit()
                logger.info(f"✅ Created {created_count} system tags")
            else:
                logger.info(f"[DRY RUN] Would create {created_count} system tags")
            
            self.stats['tags_created'] = created_count
            return created_count
        
        except Exception as e:
            logger.error(f"Error initializing system tags: {e}")
            session.rollback()
            self.stats['errors'] += 1
            return 0
    
    def run_migration(self):
        """Execute complete migration"""
        logger.info("\n")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " WORKFLOW PLAN GRAPH & ANALYTICS MIGRATION ".center(58) + "║")
        logger.info("╚" + "=" * 58 + "╝")
        
        if self.dry_run:
            logger.warning("[DRY RUN MODE] No changes will be committed to database\n")
        
        session = self.SessionLocal()
        
        try:
            # Run all migrations
            self.migrate_existing_workflows(session)
            self.migrate_plan_history(session)
            self.initialize_conversation_analytics(session)
            self.initialize_system_tags(session)
            
        finally:
            session.close()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Workflows updated:        {self.stats['workflows_updated']}")
        logger.info(f"Plans migrated:           {self.stats['plans_migrated']}")
        logger.info(f"Analytics created:        {self.stats['analytics_created']}")
        logger.info(f"System tags created:      {self.stats['tags_created']}")
        logger.info(f"Errors encountered:       {self.stats['errors']}")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.warning("\n⚠️  This was a DRY RUN. No changes were committed.")
            logger.info("Run without --dry-run flag to apply changes.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate workflows and initialize analytics tables'
    )
    parser.add_argument(
        '--db-url',
        help='Database URL (uses DATABASE_URL env var if not provided)',
        default=None
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print changes without committing to database'
    )
    
    args = parser.parse_args()
    
    migration = WorkflowMigration(db_url=args.db_url, dry_run=args.dry_run)
    migration.run_migration()


if __name__ == '__main__':
    main()
