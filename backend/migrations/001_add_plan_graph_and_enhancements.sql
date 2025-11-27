-- ============================================================================
-- Migration: Add plan_graph to workflows and create enhanced storage tables
-- Date: 2025-11-25
-- Description: Implements plan graph persistence, plan history, search, and analytics
-- ============================================================================

-- ============================================================================
-- 1. ALTER WORKFLOWS TABLE - Add plan_graph field
-- ============================================================================
ALTER TABLE workflows 
ADD COLUMN plan_graph JSON DEFAULT NULL;

CREATE INDEX idx_workflows_plan_graph ON workflows(user_id) 
WHERE plan_graph IS NOT NULL;

-- ============================================================================
-- 2. ENHANCE USER_THREADS TABLE - Add metadata for better querying
-- ============================================================================
ALTER TABLE user_threads 
ADD COLUMN preview_text TEXT DEFAULT NULL,
ADD COLUMN last_message_at TIMESTAMP DEFAULT NULL,
ADD COLUMN agent_ids JSON DEFAULT NULL,
ADD COLUMN execution_status VARCHAR(50) DEFAULT 'idle',
ADD COLUMN message_count INT DEFAULT 0,
ADD COLUMN has_plan BOOLEAN DEFAULT FALSE;

-- Create indexes for common queries
CREATE INDEX idx_user_threads_last_message ON user_threads(user_id, last_message_at DESC);
CREATE INDEX idx_user_threads_execution_status ON user_threads(user_id, execution_status);
CREATE INDEX idx_user_threads_has_plan ON user_threads(user_id, has_plan);

-- ============================================================================
-- 3. CREATE PLAN HISTORY TABLE - Track plan iterations
-- ============================================================================
CREATE TABLE IF NOT EXISTS conversation_plans (
    id SERIAL PRIMARY KEY,
    plan_id VARCHAR(255) NOT NULL UNIQUE,
    thread_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    plan_version INT NOT NULL DEFAULT 1,
    task_agent_pairs JSON NOT NULL,
    task_plan JSON NOT NULL,
    plan_graph JSON DEFAULT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'draft',  -- 'draft', 'executing', 'completed', 'failed'
    result JSON DEFAULT NULL,  -- Execution result
    execution_time_ms INT DEFAULT NULL,
    error_message TEXT DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (thread_id) REFERENCES user_threads(thread_id) ON DELETE CASCADE,
    INDEX idx_conversation_plans_user (user_id),
    INDEX idx_conversation_plans_thread (thread_id),
    INDEX idx_conversation_plans_status (status),
    INDEX idx_conversation_plans_created (user_id, created_at DESC)
);

-- ============================================================================
-- 4. CREATE CONVERSATION SEARCH TABLE - Full-text search capability
-- ============================================================================
CREATE TABLE IF NOT EXISTS conversation_search (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    message_index INT NOT NULL,
    message_content TEXT NOT NULL,
    message_role VARCHAR(50),  -- 'user', 'assistant', 'agent'
    message_timestamp TIMESTAMP DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (thread_id) REFERENCES user_threads(thread_id) ON DELETE CASCADE,
    INDEX idx_conversation_search_user (user_id),
    INDEX idx_conversation_search_thread (thread_id),
    INDEX idx_conversation_search_role (message_role),
    FULLTEXT INDEX idx_conversation_search_content (message_content)
);

-- ============================================================================
-- 5. CREATE CONVERSATION TAGGING SYSTEM
-- ============================================================================
CREATE TABLE IF NOT EXISTS conversation_tags (
    id SERIAL PRIMARY KEY,
    tag_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    tag_name VARCHAR(100) NOT NULL,
    tag_color VARCHAR(7) DEFAULT '#808080',  -- Hex color for UI
    tag_description TEXT DEFAULT NULL,
    is_system BOOLEAN DEFAULT FALSE,  -- System tags: 'urgent', 'completed', etc.
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_conversation_tags_user (user_id),
    UNIQUE KEY unique_user_tag (user_id, tag_name)
);

-- Create junction table for many-to-many relationship
CREATE TABLE IF NOT EXISTS conversation_tag_assignments (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,
    tag_id VARCHAR(255) NOT NULL,
    assigned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (thread_id) REFERENCES user_threads(thread_id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES conversation_tags(tag_id) ON DELETE CASCADE,
    UNIQUE KEY unique_thread_tag (thread_id, tag_id),
    INDEX idx_tag_assignments_tag (tag_id)
);

-- ============================================================================
-- 6. CREATE ANALYTICS TABLES - User activity and performance tracking
-- ============================================================================

-- Table: Conversation Analytics - Tracks metrics per conversation
CREATE TABLE IF NOT EXISTS conversation_analytics (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    total_messages INT DEFAULT 0,
    total_agents_used INT DEFAULT 0,
    plan_attempts INT DEFAULT 0,  -- How many different plans were tried
    successful_plans INT DEFAULT 0,  -- How many completed successfully
    total_execution_time_ms INT DEFAULT 0,
    failed_executions INT DEFAULT 0,
    avg_response_time_ms DECIMAL(10, 2) DEFAULT 0,
    conversation_duration_seconds INT DEFAULT 0,  -- From first to last message
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (thread_id) REFERENCES user_threads(thread_id) ON DELETE CASCADE,
    INDEX idx_conversation_analytics_user (user_id),
    INDEX idx_conversation_analytics_updated (user_id, updated_at DESC)
);

-- Table: Agent Usage Analytics - Tracks which agents are used most
CREATE TABLE IF NOT EXISTS agent_usage_analytics (
    id SERIAL PRIMARY KEY,
    analytics_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    execution_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    avg_execution_time_ms DECIMAL(10, 2) DEFAULT 0,
    last_used_at TIMESTAMP DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_agent (user_id, agent_id),
    INDEX idx_agent_usage_user (user_id),
    INDEX idx_agent_usage_popularity (user_id, execution_count DESC)
);

-- Table: User Activity Summary - Daily/Monthly rollups
CREATE TABLE IF NOT EXISTS user_activity_summary (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    activity_date DATE NOT NULL,
    total_conversations_started INT DEFAULT 0,
    total_workflows_executed INT DEFAULT 0,
    total_plans_created INT DEFAULT 0,
    successful_executions INT DEFAULT 0,
    failed_executions INT DEFAULT 0,
    total_execution_time_ms INT DEFAULT 0,
    agents_used INT DEFAULT 0,
    api_calls_made INT DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_user_date (user_id, activity_date),
    INDEX idx_user_activity_user (user_id),
    INDEX idx_user_activity_date (activity_date)
);

-- Table: Workflow Execution Analytics - Detailed metrics per execution
CREATE TABLE IF NOT EXISTS workflow_execution_analytics (
    id SERIAL PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    workflow_id VARCHAR(255) NOT NULL,
    total_steps INT DEFAULT 0,
    completed_steps INT DEFAULT 0,
    failed_steps INT DEFAULT 0,
    total_duration_ms INT DEFAULT 0,
    retry_count INT DEFAULT 0,
    error_type VARCHAR(100) DEFAULT NULL,
    success_rate DECIMAL(5, 2) DEFAULT 0,  -- (completed_steps / total_steps) * 100
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE,
    INDEX idx_workflow_execution_user (user_id),
    INDEX idx_workflow_execution_workflow (workflow_id),
    INDEX idx_workflow_execution_created (user_id, created_at DESC)
);

-- ============================================================================
-- 7. CREATE MATERIALIZED VIEW - User Statistics Dashboard
-- ============================================================================
CREATE VIEW user_statistics_summary AS
SELECT 
    u.user_id,
    COUNT(DISTINCT ut.thread_id) as total_conversations,
    COUNT(DISTINCT CASE WHEN ut.execution_status = 'running' THEN ut.thread_id END) as active_conversations,
    COUNT(DISTINCT w.workflow_id) as total_workflows,
    COUNT(DISTINCT cp.plan_id) as total_plans_created,
    COUNT(DISTINCT CASE WHEN cp.status = 'completed' THEN cp.plan_id END) as successful_plans,
    COALESCE(SUM(uas.execution_count), 0) as total_agent_executions,
    COALESCE(AVG(uas.avg_execution_time_ms), 0) as avg_agent_execution_time,
    MAX(ut.updated_at) as last_activity,
    NOW() as generated_at
FROM (SELECT DISTINCT user_id FROM user_threads) u
LEFT JOIN user_threads ut ON u.user_id = ut.user_id
LEFT JOIN workflows w ON u.user_id = w.user_id
LEFT JOIN conversation_plans cp ON u.user_id = cp.user_id
LEFT JOIN agent_usage_analytics uas ON u.user_id = uas.user_id
GROUP BY u.user_id;

-- ============================================================================
-- 8. UPDATE TRIGGERS - Maintain updated_at timestamps
-- ============================================================================
DELIMITER //

CREATE TRIGGER user_threads_update_timestamp
BEFORE UPDATE ON user_threads
FOR EACH ROW
SET NEW.updated_at = CURRENT_TIMESTAMP;
//

CREATE TRIGGER conversation_plans_update_timestamp
BEFORE UPDATE ON conversation_plans
FOR EACH ROW
SET NEW.updated_at = CURRENT_TIMESTAMP;
//

CREATE TRIGGER conversation_analytics_update_timestamp
BEFORE UPDATE ON conversation_analytics
FOR EACH ROW
SET NEW.updated_at = CURRENT_TIMESTAMP;
//

CREATE TRIGGER agent_usage_analytics_update_timestamp
BEFORE UPDATE ON agent_usage_analytics
FOR EACH ROW
SET NEW.updated_at = CURRENT_TIMESTAMP;
//

CREATE TRIGGER user_activity_summary_update_timestamp
BEFORE UPDATE ON user_activity_summary
FOR EACH ROW
SET NEW.updated_at = CURRENT_TIMESTAMP;
//

CREATE TRIGGER workflow_execution_analytics_update_timestamp
BEFORE UPDATE ON workflow_execution_analytics
FOR EACH ROW
SET NEW.updated_at = CURRENT_TIMESTAMP;
//

DELIMITER ;

-- ============================================================================
-- 9. VERIFY MIGRATIONS
-- ============================================================================
-- Check that all new columns exist in workflows table
SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'workflows' 
AND COLUMN_NAME IN ('plan_graph', 'created_at', 'updated_at')
ORDER BY ORDINAL_POSITION;

-- Check that all new tables exist
SELECT TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME IN (
    'conversation_plans',
    'conversation_search',
    'conversation_tags',
    'conversation_tag_assignments',
    'conversation_analytics',
    'agent_usage_analytics',
    'user_activity_summary',
    'workflow_execution_analytics'
)
ORDER BY TABLE_NAME;

-- ============================================================================
-- END OF MIGRATION
-- ============================================================================
