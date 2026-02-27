// components/PlanGraph.tsx
/**
 * PlanGraph Component - Workflow Visualization
 * 
 * Supports two execution models:
 * 1. NEW SYSTEM (todo_list): Sequential task execution (linear flow)
 * 2. LEGACY SYSTEM (task_plan): Parallel batch execution (multi-level flow)
 * 
 * Automatically detects the input format and renders appropriate visualization.
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
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { CheckCircle, Clock, PlayCircle, Loader2, XCircle, Zap, AlertTriangle, ArrowRight } from "lucide-react"
import { MdMail, MdTableChart, MdDescription } from "react-icons/md"
import { cn } from "@/lib/utils"
import type { TaskStatus } from "@/lib/types"

interface LegacyPlanTask {
    task: string;
    description: string;
    agent: string;
    short_description?: string;
    agent_image_url?: string;
}

interface TodoListTask {
    id: string;
    title: string;
    description?: string;
    instructions?: string;
    status: 'pending' | 'in-progress' | 'completed' | 'failed';
    assigned_to?: string;
}

interface CompletedTask {
    task: string;
    result: string;
}

interface Plan {
    pendingTasks: (LegacyPlanTask | LegacyPlanTask[])[];
    completedTasks: CompletedTask[];
    // Optional: Support todo_list directly in plan
    todoList?: TodoListTask[];
}

interface PlanGraphProps {
    planData: Plan;
    todoList?: TodoListTask[]; // Can be passed directly
    taskStatuses?: Record<string, TaskStatus>;
}

interface CustomNodeData {
    label: string;
    description: string;
    agent: string;
    status: 'completed' | 'pending' | 'running' | 'failed' | 'in-progress' | 'start';
    executionTime?: number;
    isDialogue?: boolean;
    error?: string;
    short_description?: string;
    icon_name?: string;
    taskId?: string; // For todo_list integration
    isTodoItem?: boolean; // Indicates this is from todo_list
}

// Map icon names to React Icon components
const iconMap: Record<string, React.ComponentType<any>> = {
    'MdMail': MdMail,
    'MdTableChart': MdTableChart,
    'MdDescription': MdDescription,
};

const getIconComponent = (iconName?: string) => {
    if (!iconName) return null;
    return iconMap[iconName] || null;
};

// Custom Node Component with real-time status updates
const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ data }) => {
    const isCompleted = data.status === 'completed';
    const isRunning = data.status === 'in-progress' || data.status === 'running';
    const isFailed = data.status === 'failed';
    const isPending = data.status === 'pending';
    const isDialogue = data.isDialogue && isRunning;

    // Get node color based on status - matches project color scheme
    const getNodeColor = () => {
        if (isDialogue) return 'bg-status-pending-light dark:bg-status-pending-dark/20 border-status-pending-dark dark:border-status-pending';
        if (isRunning) return 'bg-status-active-light dark:bg-status-active-dark/20 border-status-active-dark dark:border-status-active animate-pulse';
        if (isCompleted) return 'bg-status-success-light dark:bg-status-success-dark/20 border-status-success-dark dark:border-status-success';
        if (isFailed) return 'bg-status-error-light dark:bg-status-error-dark/20 border-status-error-dark dark:border-status-error';
        return 'bg-border-color-light dark:bg-bg-card border-border-color dark:border-border-color-strong';
    };

    const getStatusIcon = () => {
        if (isCompleted) return <CheckCircle className="w-4 h-4 text-status-success" />;
        if (isRunning) return <Loader2 className="w-4 h-4 text-status-active animate-spin" />;
        if (isFailed) return <XCircle className="w-4 h-4 text-status-error" />;
        if (isDialogue) return <AlertTriangle className="w-4 h-4 text-status-pending-dark animate-pulse" />;
        return <Clock className="w-4 h-4 text-text-tertiary" />;
    };

    return (
        <div className={cn(
            'px-4 py-3 rounded-lg border-2 bg-bg-card shadow-lg min-w-72 max-w-80',
            getNodeColor()
        )}>
            <Handle type="target" position={Position.Top} />

            <div className="flex items-start gap-3 mb-2">
                <div className="flex-shrink-0 mt-1">
                    {getStatusIcon()}
                </div>
                <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-text-primary truncate text-sm">
                        {data.label}
                    </h4>
                    <p className="text-xs text-text-tertiary mt-1">
                        {data.agent}
                    </p>
                </div>
            </div>

            {data.description && (
                <p className="text-xs text-text-secondary mb-2 line-clamp-2">
                    {data.description}
                </p>
            )}

            {data.executionTime && (
                <div className="flex items-center gap-1 text-xs text-text-tertiary">
                    <Clock className="w-3 h-3" />
                    <span>{data.executionTime}s</span>
                </div>
            )}

            {data.error && (
                <div className="mt-2 p-2 bg-status-error-light rounded text-xs text-status-error-dark">
                    {data.error}
                </div>
            )}

            <Handle type="source" position={Position.Bottom} />
        </div>
    );
};

export default function PlanGraph({ planData, todoList: externalTodoList, taskStatuses = {} }: PlanGraphProps) {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);
    const [executionModel, setExecutionModel] = useState<'todo-list' | 'batch-plan'>('batch-plan');

    const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);

    useEffect(() => {
        // Detect which execution model to use
        const todoList = externalTodoList || planData.todoList || [];
        const hasTodoList = todoList && todoList.length > 0;
        const hasTaskPlan = planData.pendingTasks && planData.pendingTasks.length > 0;

        if (hasTodoList) {
            setExecutionModel('todo-list');
            buildTodoListGraph(todoList, taskStatuses);
        } else if (hasTaskPlan) {
            setExecutionModel('batch-plan');
            buildBatchPlanGraph(planData.pendingTasks, taskStatuses);
        } else {
            setNodes([]);
            setEdges([]);
        }
    }, [planData, externalTodoList, taskStatuses]);

    /**
     * Build graph for NEW TODO_LIST system (sequential execution)
     */
    const buildTodoListGraph = (todoList: TodoListTask[], taskStatuses: Record<string, TaskStatus>) => {
        if (!todoList || todoList.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const initialNodes: Node[] = [];
        const initialEdges: Edge[] = [];

        // Add START node
        initialNodes.push({
            id: 'start',
            type: 'custom',
            data: {
                label: 'Start',
                description: 'Workflow begins',
                agent: 'System',
                status: 'start'
            },
            position: { x: 200, y: 0 }
        });

        // Sequential layout: tasks in a vertical line
        const xPos = 200;
        const ySpacing = 180; // Vertical spacing between tasks

        todoList.forEach((task, index) => {
            const taskStatus = taskStatuses[task.title];
            // Map todo_list status to component status
            const statusMap: Record<string, any> = {
                'pending': 'pending',
                'in-progress': 'in-progress',
                'completed': 'completed',
                'failed': 'failed'
            };
            const status = statusMap[task.status] || 'pending';

            const nodeId = `todo-task-${task.id}`;
            const yPos = (index + 1) * ySpacing;

            initialNodes.push({
                id: nodeId,
                type: 'custom',
                data: {
                    label: task.title,
                    description: task.description || task.instructions || '',
                    agent: task.assigned_to || 'Unassigned',
                    status: status,
                    taskId: task.id,
                    isTodoItem: true,
                    executionTime: taskStatus?.executionTime,
                    error: taskStatus?.error
                },
                position: { x: xPos, y: yPos }
            });

            // Edge from previous task (or start)
            const sourceId = index === 0 ? 'start' : `todo-task-${todoList[index - 1].id}`;
            initialEdges.push({
                id: `edge-${sourceId}-${nodeId}`,
                source: sourceId,
                target: nodeId,
                animated: status === 'pending' || status === 'in-progress',
                style: {
                    strokeWidth: 2,
                    stroke: getEdgeColor(status),
                    transition: 'stroke 0.3s ease-in-out'
                }
            });
        });

        setNodes(initialNodes);
        setEdges(initialEdges);
    };

    /**
     * Build graph for LEGACY BATCH_PLAN system (parallel execution)
     */
    const buildBatchPlanGraph = (pendingTasks: (LegacyPlanTask | LegacyPlanTask[])[], taskStatuses: Record<string, TaskStatus>) => {
        if (!pendingTasks || pendingTasks.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const initialNodes: Node[] = [];
        const initialEdges: Edge[] = [];

        const yOffset = 220;
        const xNodeWidth = 320;
        let currentY = 1;
        let previousLayerIds: string[] = [];

        pendingTasks.forEach((batchOrTask, rankIndex) => {
            // Normalize to array of tasks
            const batch = Array.isArray(batchOrTask) ? batchOrTask : [batchOrTask];
            const currentLayerIds: string[] = [];
            const batchSize = batch.length;

            const startX = 400 - ((batchSize - 1) * 320) / 2;

            batch.forEach((task: any, indexInBatch: number) => {
                const taskName = task.task;
                const taskStatus = taskStatuses[taskName];
                const status = taskStatus?.status || task.status || 'pending';

                const nodeId = `task-${taskKey}`;
                currentLayerIds.push(nodeId);

                const xPos = startX + (indexInBatch * 320);
                const yPos = (rankIndex + 1) * yOffset;

                initialNodes.push({
                    id: nodeId,
                    type: 'custom',
                    data: {
                        label: taskName,
                        description: task.description || 'Completed',
                        agent: taskStatus?.agentName || task.agent || 'N/A',
                        short_description: task.short_description,
                        icon_name: task.icon_name,
                        status: status,
                        executionTime: taskStatus?.executionTime,
                        error: taskStatus?.error,
                        isDialogue: taskStatus?.is_dialogue
                    },
                    position: { x: xPos, y: yPos }
                });

                // Edges from previous layer
                previousLayerIds.forEach(prevId => {
                    const edgeColor = getEdgeColor(status);
                    initialEdges.push({
                        id: `edge-${prevId}-${nodeId}`,
                        source: prevId,
                        target: nodeId,
                        animated: status === 'pending' || status === 'running' || status === 'in-progress',
                        style: {
                            strokeWidth: 2,
                            stroke: edgeColor,
                            transition: 'stroke 0.3s ease-in-out'
                        }
                    });
                });
            });

            previousLayerIds = currentLayerIds;
            currentY++;
        });

        setNodes(initialNodes);
        setEdges(initialEdges);
    };

    /**
     * Get edge color based on task status
     */
    const getEdgeColor = (status: string): string => {
        switch (status) {
            case 'in-progress':
            case 'running':
                return 'var(--color-status-active)';
            case 'completed':
                return 'var(--color-status-success)';
            case 'failed':
                return 'var(--color-status-error)';
            case 'pending':
            default:
                return 'var(--color-text-tertiary)';
        }
    };

    // Empty state
    if (!planData || (planData.pendingTasks.flat().length === 0 && (!externalTodoList || externalTodoList.length === 0))) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-bg-subtle rounded-lg border-2 border-dashed border-border-color-medium">
                <div className="text-center text-text-tertiary p-8">
                    <div className="mb-4">
                        <svg
                            className="w-16 h-16 mx-auto text-text-disabled animate-pulse"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={1.5}
                                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"
                            />
                        </svg>
                    </div>
                    <p className="font-semibold text-orbimesh-section-header mb-2 text-text-secondary">Workflow Plan</p>
                    <p className="text-orbimesh-section-subtitle text-text-tertiary">
                        {executionModel === 'todo-list'
                            ? 'No tasks in the sequence yet'
                            : 'The execution plan graph will appear here once tasks are identified'}
                    </p>
                </div>
            </div>
        );
    }

    // Calculate progress
    const todoListToUse = externalTodoList || planData.todoList || [];
    const totalTasks = todoListToUse.length || planData.pendingTasks.flat().length;
    const completedCount = Object.values(taskStatuses).filter((t: any) => t.status === 'completed').length;
    const runningCount = Object.values(taskStatuses).filter((t: any) => t.status === 'running' || t.status === 'in-progress').length;
    const failedCount = Object.values(taskStatuses).filter((t: any) => t.status === 'failed').length;
    const progress = totalTasks > 0 ? (completedCount / totalTasks) * 100 : 0;

    return (
        <div className="w-full h-full rounded-lg border border-border-color bg-bg-card flex flex-col">
            {/* Model Badge */}
            <div className="px-4 py-2 border-b border-border-color bg-bg-subtle flex items-center justify-between">
                <Badge
                    variant={executionModel === 'todo-list' ? 'default' : 'secondary'}
                    className="text-xs"
                >
                    {executionModel === 'todo-list' ? '📝 Sequential (Todo List)' : '⚡ Parallel (Batch Plan)'}
                </Badge>
            </div>

            {/* Progress Bar */}
            {totalTasks > 0 && progress < 100 && (
                <div className="px-4 py-2 border-b border-border-color bg-bg-subtle">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-orbimesh-metadata-label font-medium text-text-secondary">
                            Progress: {completedCount}/{totalTasks} tasks
                        </span>
                        <span className="text-orbimesh-file-meta text-text-tertiary">
                            {runningCount > 0 && `${runningCount} running`}
                            {failedCount > 0 && ` • ${failedCount} failed`}
                        </span>
                    </div>
                    <div className="w-full bg-border-color-light rounded-full h-2 overflow-hidden">
                        <div
                            className="bg-status-success h-2 rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Completion Banner */}
            {progress === 100 && totalTasks > 0 && (
                <div className="px-4 py-3 bg-status-success-light border-b border-status-success-border">
                    <div className="flex items-center gap-2">
                        <CheckCircle className="w-5 h-5 text-status-success-dark" />
                        <span className="text-orbimesh-section-subtitle font-semibold text-status-success-dark">
                            All tasks completed successfully!
                        </span>
                        <Zap className="w-4 h-4 text-status-pending animate-pulse" />
                    </div>
                </div>
            )}

            {/* Graph */}
            <div className="flex-1 bg-bg-subtle">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    fitView
                    proOptions={{ hideAttribution: true }}
                    className="bg-bg-subtle"
                >
                    <Controls className="bg-bg-card border border-border-color" />
                    <Background
                        color="var(--color-border-medium)"
                        className="bg-bg-subtle"
                    />
                </ReactFlow>
            </div>
        </div>
    );
}
