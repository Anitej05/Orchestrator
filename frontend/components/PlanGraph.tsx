// components/PlanGraph.tsx
/**
 * SIMPLIFIED PlanGraph Component - For Non-Technical Users
 * 
 * Shows workflow as a simple visual timeline:
 * - NO START/END nodes
 * - Just agent name + simple description per step
 * - Color-coded statuses with icons for instant recognition
 * - Clean, minimal design
 */

"use client"

import React, { useEffect, useMemo, useState } from 'react';
import ReactFlow, {
    Controls,
    Background,
    Node,
    Edge,
    Position,
    Handle,
    NodeProps
} from 'reactflow';
import 'reactflow/dist/style.css';
import { CheckCircle, Loader2, Circle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

// Agent Icon Mapping - 15+ agent types for visual recognition
const AGENT_ICONS: Record<string, string> = {
    // Email & Communication
    'gmail': '📧',
    'gmail_agent': '📧',
    'mail': '📬',
    'mail_agent': '📬',
    
    // Documents & Files
    'document': '📄',
    'document_agent': '📄',
    'spreadsheet': '📊',
    'spreadsheet_agent': '📊',
    
    // Web & Browser
    'browser': '🌐',
    'browser_agent': '🌐',
    'web': '🕸️',
    
    // Code & Development
    'coding': '💻',
    'coding_agent': '💻',
    'python': '🐍',
    'code': '⚡',
    
    // System & Terminal
    'terminal': '⌨️',
    'system': '⚙️',
    
    // Business & Finance
    'zoho_books': '💼',
    'zoho_books_agent': '💼',
    'finance': '💰',
    
    // General & Planning
    'general': '🤖',
    'general_agent': '🤖',
    'universal': '🌟',
    'universal_agent': '🌟',
    'plan': '📋',
    'planning': '🗓️',
    
    // Default fallback
    'default': '🔷'
};

interface ActionHistoryEntry {
    iteration: number;
    action_type: string;
    resource_id: string;     // Agent/tool name
    instruction: string;
    success: boolean;
    result_summary: string;
    execution_time_ms: number;
}

interface TodoItem {
    task_id: string;
    description: string;
    status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED';
    priority: number;
}

interface PlanGraphProps {
    // NEW SYSTEM: Use these for real-time updates
    actionHistory?: ActionHistoryEntry[];    // Detailed execution log
    todoList?: TodoItem[];                   // High-level tasks
    
    // LEGACY SYSTEM: Old format (keep for backward compatibility)
    planData?: {
        pendingTasks: any[];
        completedTasks: any[];
    };
}

interface SimpleNodeData {
    // What to show (KEEP IT SIMPLE!)
    agentName: string;        // "Gmail Agent" (formatted)
    agentIcon: string;        // "📧" (emoji for visual recognition)
    resourceId: string;       // Raw "gmail_agent" (for icon lookup)
    description: string;      // One line: "Sending 5 emails..."
    status: 'pending' | 'running' | 'completed' | 'failed';
    
    // Extra info (hidden until hover)
    executionTime?: number;   // Milliseconds
    error?: string;
    iteration?: number;
}

// ============================================================================
// SIMPLIFIED NODE COMPONENT
// ============================================================================

const SimpleNode: React.FC<NodeProps<SimpleNodeData>> = ({ data }) => {
    const { agentName, agentIcon, description, status, executionTime, error } = data;

    // Status-based styling
    const getNodeStyles = () => {
        switch (status) {
            case 'completed':
                return {
                    bg: 'bg-green-50 dark:bg-green-950/30',
                    border: 'border-green-500',
                    icon: <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />,
                    iconColor: 'text-green-600 dark:text-green-400'
                };
            case 'running':
                return {
                    bg: 'bg-blue-50 dark:bg-blue-950/30',
                    border: 'border-blue-500',
                    icon: <Loader2 className="w-5 h-5 text-blue-600 dark:text-blue-400 animate-spin" />,
                    iconColor: 'text-blue-600 dark:text-blue-400'
                };
            case 'failed':
                return {
                    bg: 'bg-red-50 dark:bg-red-950/30',
                    border: 'border-red-500',
                    icon: <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />,
                    iconColor: 'text-red-600 dark:text-red-400'
                };
            case 'pending':
            default:
                return {
                    bg: 'bg-gray-50 dark:bg-gray-900/30',
                    border: 'border-gray-300 dark:border-gray-700',
                    icon: <Circle className="w-5 h-5 text-gray-400" />,
                    iconColor: 'text-gray-400'
                };
        }
    };

    const styles = getNodeStyles();

    return (
        <div
            className={cn(
                "min-w-[280px] max-w-[320px] rounded-lg border-2 shadow-sm transition-all",
                styles.bg,
                styles.border,
                "hover:shadow-md"
            )}
            title={error || `${agentName} - ${status}`}
        >
            <Handle type="target" position={Position.Top} className="!bg-gray-400" />
            
            <div className="p-3">
                {/* Header: Icon + Agent Name + Status */}
                <div className="flex items-center gap-3 mb-2">
                    {/* Agent Icon - Large & Prominent */}
                    <div className="text-2xl leading-none flex-shrink-0">
                        {agentIcon}
                    </div>
                    
                    {/* Agent Name */}
                    <div className="flex-1 min-w-0">
                        <div className="font-semibold text-sm text-gray-900 dark:text-gray-100 truncate">
                            {agentName}
                        </div>
                    </div>
                    
                    {/* Status Icon */}
                    <div className="flex-shrink-0">
                        {styles.icon}
                    </div>
                </div>
                
                {/* Description - Simple one line */}
                <div className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 mb-1">
                    {description}
                </div>
                
                {/* Execution Time - Subtle */}
                {executionTime !== undefined && executionTime > 0 && (
                    <div className="text-xs text-gray-500 dark:text-gray-500 text-right">
                        {(executionTime / 1000).toFixed(1)}s
                    </div>
                )}
                
                {/* Error - Only if failed */}
                {error && status === 'failed' && (
                    <div className="mt-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/50 p-2 rounded">
                        {error}
                    </div>
                )}
            </div>
            
            <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
        </div>
    );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function PlanGraph({
    actionHistory = [],
    todoList = [],
    planData
}: PlanGraphProps) {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);

    const nodeTypes = useMemo(() => ({ simple: SimpleNode }), []);

    // Use stringified versions in dependency array to avoid infinite loops from object reference changes
    useEffect(() => {
        // Priority: action_history > todoList > legacy planData
        if (actionHistory && actionHistory.length > 0) {
            buildFromActionHistory(actionHistory);
        } else if (todoList && todoList.length > 0) {
            buildFromTodoList(todoList);
        } else if (planData && (planData.pendingTasks?.length > 0 || planData.completedTasks?.length > 0)) {
            buildFromLegacyPlan(planData);
        } else {
            setNodes([]);
            setEdges([]);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [JSON.stringify(actionHistory), JSON.stringify(todoList), JSON.stringify(planData)]);

    /**
     * Build graph from action_history (BEST - most detailed)
     * FILTERS to show only agents/tools, not simple transactions
     */
    const buildFromActionHistory = (history: ActionHistoryEntry[]) => {
        const newNodes: Node[] = [];
        const newEdges: Edge[] = [];

        // Filter to show only significant agent/tool actions
        const significantActions = history.filter(action => isSignificantAction(action));

        significantActions.forEach((action, index) => {
            const agentIcon = getAgentIcon(action.resource_id);
            const agentName = formatAgentName(action.resource_id);
            const description = createDetailedDescription(action);
            
            // Determine status
            let status: SimpleNodeData['status'] = 'completed';
            if (index === significantActions.length - 1 && !action.success) {
                status = 'running'; // Last action might still be running
            } else if (!action.success) {
                status = 'failed';
            }

            const nodeData: SimpleNodeData = {
                agentName,
                agentIcon,
                resourceId: action.resource_id,
                description,
                status,
                executionTime: action.execution_time_ms,
                iteration: action.iteration,
                error: !action.success ? action.result_summary : undefined
            };

            newNodes.push({
                id: `action-${action.iteration}`,
                type: 'simple',
                data: nodeData,
                position: { x: 0, y: index * 150 }
            });

            // Create edge from previous node
            if (index > 0) {
                newEdges.push({
                    id: `edge-${index - 1}-${index}`,
                    source: `action-${significantActions[index - 1].iteration}`,
                    target: `action-${action.iteration}`,
                    animated: status === 'running',
                    style: { stroke: getEdgeColor(status) }
                });
            }
        });

        setNodes(newNodes);
        setEdges(newEdges);
    };

    /**
     * Build graph from todo_list (OKAY - task-level)
     */
    const buildFromTodoList = (todos: TodoItem[]) => {
        const newNodes: Node[] = [];
        const newEdges: Edge[] = [];

        // Sort by priority
        const sortedTodos = [...todos].sort((a, b) => a.priority - b.priority);

        sortedTodos.forEach((todo, index) => {
            // Try to extract agent from description
            const extractedAgent = extractAgentFromDescription(todo.description);
            const agentIcon = getAgentIcon(extractedAgent || 'default');
            const agentName = extractedAgent ? formatAgentName(extractedAgent) : 'Task';
            const description = truncate(todo.description, 50);
            
            // Map todo status to node status
            let status: SimpleNodeData['status'] = 'pending';
            if (todo.status === 'IN_PROGRESS') status = 'running';
            else if (todo.status === 'COMPLETED') status = 'completed';
            else if (todo.status === 'FAILED') status = 'failed';

            const nodeData: SimpleNodeData = {
                agentName,
                agentIcon,
                resourceId: extractedAgent || 'unknown',
                description,
                status
            };

            newNodes.push({
                id: todo.task_id,
                type: 'simple',
                data: nodeData,
                position: { x: 0, y: index * 150 }
            });

            // Create edge from previous node
            if (index > 0) {
                newEdges.push({
                    id: `edge-${sortedTodos[index - 1].task_id}-${todo.task_id}`,
                    source: sortedTodos[index - 1].task_id,
                    target: todo.task_id,
                    animated: status === 'running',
                    style: { stroke: getEdgeColor(status) }
                });
            }
        });

        setNodes(newNodes);
        setEdges(newEdges);
    };

    /**
     * Build from legacy plan data (FALLBACK)
     */
    const buildFromLegacyPlan = (plan: any) => {
        const newNodes: Node[] = [];
        const newEdges: Edge[] = [];
        let index = 0;

        // Add completed tasks
        if (plan.completedTasks && plan.completedTasks.length > 0) {
            plan.completedTasks.forEach((task: any) => {
                const resourceId = task.agent || 'unknown';
                const agentIcon = getAgentIcon(resourceId);
                const agentName = formatAgentName(resourceId);
                const description = truncate(task.task || task.description || 'Completed task', 50);

                const nodeData: SimpleNodeData = {
                    agentName,
                    agentIcon,
                    resourceId,
                    description,
                    status: 'completed'
                };

                newNodes.push({
                    id: `completed-${index}`,
                    type: 'simple',
                    data: nodeData,
                    position: { x: 0, y: index * 150 }
                });

                if (index > 0) {
                    newEdges.push({
                        id: `edge-${index - 1}-${index}`,
                        source: `completed-${index - 1}`,
                        target: `completed-${index}`,
                        style: { stroke: getEdgeColor('completed') }
                    });
                }
                index++;
            });
        }

        // Add pending tasks
        if (plan.pendingTasks && plan.pendingTasks.length > 0) {
            plan.pendingTasks.forEach((task: any) => {
                const resourceId = task.agent || 'unknown';
                const agentIcon = getAgentIcon(resourceId);
                const agentName = formatAgentName(resourceId);
                const description = truncate(task.task || task.description || 'Pending task', 50);

                const nodeData: SimpleNodeData = {
                    agentName,
                    agentIcon,
                    resourceId,
                    description,
                    status: 'pending'
                };

                newNodes.push({
                    id: `pending-${index}`,
                    type: 'simple',
                    data: nodeData,
                    position: { x: 0, y: index * 150 }
                });

                if (index > 0) {
                    const prevId = index - 1 < plan.completedTasks?.length 
                        ? `completed-${index - 1}` 
                        : `pending-${index - 1}`;
                    newEdges.push({
                        id: `edge-${index - 1}-${index}`,
                        source: prevId,
                        target: `pending-${index}`,
                        style: { stroke: getEdgeColor('pending') }
                    });
                }
                index++;
            });
        }

        setNodes(newNodes);
        setEdges(newEdges);
    };

    // ========================================================================
    // HELPER FUNCTIONS
    // ========================================================================

    /**
     * Get appropriate icon for an agent/resource
     */
    function getAgentIcon(resourceId: string): string {
        if (!resourceId) return AGENT_ICONS.default;
        
        const lowerResource = resourceId.toLowerCase();
        
        // Direct match
        if (AGENT_ICONS[lowerResource]) {
            return AGENT_ICONS[lowerResource];
        }
        
        // Keyword matching for flexibility
        if (lowerResource.includes('mail') || lowerResource.includes('gmail')) {
            return '📧';
        }
        if (lowerResource.includes('doc') || lowerResource.includes('document')) {
            return '📄';
        }
        if (lowerResource.includes('sheet') || lowerResource.includes('spreadsheet')) {
            return '📊';
        }
        if (lowerResource.includes('browse') || lowerResource.includes('web')) {
            return '🌐';
        }
        if (lowerResource.includes('code') || lowerResource.includes('coding')) {
            return '💻';
        }
        if (lowerResource.includes('python')) {
            return '🐍';
        }
        if (lowerResource.includes('terminal') || lowerResource.includes('shell')) {
            return '⌨️';
        }
        if (lowerResource.includes('book') || lowerResource.includes('finance') || lowerResource.includes('zoho')) {
            return '💼';
        }
        if (lowerResource.includes('plan')) {
            return '📋';
        }
        
        // Default fallback
        return AGENT_ICONS.default;
    }

    /**
     * Format agent names for display
     * "gmail_agent" → "Gmail Agent"
     */
    function formatAgentName(resourceId: string): string {
        if (!resourceId) return 'Agent';
        
        return resourceId
            .replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

    /**
     * Check if action is significant (agent/tool invocation, not simple transaction)
     * STRICT FILTER: Only show successful agent and major tool operations
     */
    function isSignificantAction(action: ActionHistoryEntry): boolean {
        const resourceId = action.resource_id?.toLowerCase() || '';
        const instruction = action.instruction?.toLowerCase() || '';
        
        // MUST be successful - no failed operations
        if (!action.success) {
            return false;
        }
        
        // MUST be an agent or major tool
        const isAgent = resourceId.includes('_agent');
        const isMajorTool = resourceId.includes('spreadsheet') || 
                           resourceId.includes('gmail') || 
                           resourceId.includes('document') || 
                           resourceId.includes('browser') || 
                           resourceId.includes('coding') || 
                           resourceId.includes('zoho') ||
                           resourceId.includes('mail');
        
        if (!isAgent && !isMajorTool) {
            return false;
        }
        
        // Exclude: simple operations, intermediate steps, system operations
        const excludeKeywords = ['print', 'log', 'debug', 'wait', 'sleep', 'file path', 'check', 'verify'];
        const hasExcludeKeyword = excludeKeywords.some(kw => 
            resourceId.includes(kw) || instruction.includes(kw)
        );
        
        if (hasExcludeKeyword) {
            return false;
        }
        
        // Additional check: must have meaningful execution time
        if (action.execution_time_ms < 50) {
            return false;
        }
        
        return true;
    }

    /**
     * Create detailed description showing what the tool specifically does
     * Parses result_summary to extract clean, user-friendly text
     * Examples: "Analyzes CSV file and extracts invoice data"
     *           "Sends 5 emails to clients with reports"
     */
    function createDetailedDescription(action: ActionHistoryEntry): string {
        let description = '';
        
        // Try to parse result_summary if it exists
        if (action.result_summary && action.result_summary.length > 0) {
            // Extract content from various formats:
            // Format 1: result: {'task_summary': "The spreadsheet..."}
            // Format 2: {"task_summary": "Text here"}
            // Format 3: Plain text
            
            const taskSummaryMatch = action.result_summary.match(/['"]task_summary['"]\s*:\s*['"]([^'"]+)['"]/i);
            if (taskSummaryMatch && taskSummaryMatch[1]) {
                description = taskSummaryMatch[1];
            } else {
                // Try to extract meaningful text after common prefixes
                const cleanText = action.result_summary
                    .replace(/^result:\s*/i, '')
                    .replace(/^\{[^}]*\}\s*/, '')
                    .replace(/^['"]/, '')
                    .replace(/['"]$/, '')
                    .trim();
                
                if (cleanText && cleanText.length > 10) {
                    description = cleanText;
                }
            }
        }
        
        // If we got a clean description, use it
        if (description && description.length > 10) {
            return truncate(description, 70);
        }
        
        // Try instruction as fallback
        if (action.instruction && action.instruction.length > 20) {
            return truncate(action.instruction, 70);
        }
        
        // Generate description based on resource type
        const resourceId = action.resource_id?.toLowerCase() || '';
        if (resourceId.includes('spreadsheet')) {
            return 'Analyzing spreadsheet data';
        } else if (resourceId.includes('gmail') || resourceId.includes('mail')) {
            return 'Managing email operations';
        } else if (resourceId.includes('document')) {
            return 'Processing document content';
        } else if (resourceId.includes('browser')) {
            return 'Browsing and extracting web data';
        } else if (resourceId.includes('coding') || resourceId.includes('python')) {
            return 'Executing code analysis';
        }
        
        return 'Processing your request';
    }

    /**
     * Extract agent name from todo description
     * "Use gmail_agent to send emails" → "gmail_agent"
     */
    function extractAgentFromDescription(desc: string): string | null {
        // Look for common patterns
        const agentMatch = desc.match(/\b(\w+_agent)\b/i);
        if (agentMatch) return agentMatch[1].toLowerCase();
        
        // Check for specific keywords
        if (desc.toLowerCase().includes('email') || desc.toLowerCase().includes('gmail')) return 'gmail_agent';
        if (desc.toLowerCase().includes('spreadsheet') || desc.toLowerCase().includes('excel')) return 'spreadsheet_agent';
        if (desc.toLowerCase().includes('document') || desc.toLowerCase().includes('doc')) return 'document_agent';
        if (desc.toLowerCase().includes('browse') || desc.toLowerCase().includes('web')) return 'browser_agent';
        
        return null;
    }

    /**
     * Truncate text to max length
     */
    function truncate(text: string, maxLength: number): string {
        if (!text) return '';
        return text.length > maxLength ? text.slice(0, maxLength) + '...' : text;
    }

    /**
     * Get edge color based on status
     */
    function getEdgeColor(status: string): string {
        switch (status) {
            case 'completed': return '#10B981'; // green
            case 'running': case 'in_progress': return '#3B82F6'; // blue
            case 'failed': return '#EF4444'; // red
            case 'pending': default: return '#9CA3AF'; // gray
        }
    }

    // ========================================================================
    // RENDER
    // ========================================================================

    // Empty state
    if (!nodes || nodes.length === 0) {
        return (
            <div className="w-full h-[500px] rounded-lg border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
                <div className="text-center text-gray-500 dark:text-gray-400">
                    <div className="text-4xl mb-3">📋</div>
                    <div className="text-sm">No workflow to display</div>
                    <div className="text-xs mt-1">Start a conversation to see the execution flow</div>
                </div>
            </div>
        );
    }

    // Calculate progress
    const completedCount = nodes.filter((n: Node<SimpleNodeData>) => n.data.status === 'completed').length;
    const totalCount = nodes.length;
    const progress = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

    return (
        <div className="w-full h-[500px] rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 flex flex-col">
            {/* Progress Header */}
            <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-800">
                <div className="flex items-center justify-between mb-2">
                    <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        Workflow Progress
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                        {completedCount} / {totalCount} completed
                    </div>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                    <div 
                        className="bg-blue-500 dark:bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${progress}%` }}
                    />
                </div>
            </div>

            {/* ReactFlow Graph */}
            <div className="flex-1">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    fitView
                    attributionPosition="bottom-left"
                >
                    <Controls className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800" />
                    <Background 
                        color="#e5e7eb" 
                        className="dark:bg-gray-900" 
                        gap={16} 
                    />
                </ReactFlow>
            </div>
        </div>
    );
}
