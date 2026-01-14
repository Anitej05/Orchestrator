// components/PlanGraph.tsx
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
import { CheckCircle, Clock, PlayCircle, Loader2, XCircle, Zap, AlertTriangle } from "lucide-react"
import { cn } from "@/lib/utils"
import type { TaskStatus } from "@/lib/types"

interface PlanTask {
    task: string;
    description: string;
    agent: string;
}

interface CompletedTask {
    task: string;
    result: string;
}

interface Plan {
    pendingTasks: (PlanTask | PlanTask[])[];
    completedTasks: CompletedTask[];
}

interface PlanGraphProps {
    planData: Plan;
    taskStatuses?: Record<string, TaskStatus>; // NEW: Real-time task status tracking
}

interface CustomNodeData {
    label: string;
    description: string;
    agent: string;
    status: 'completed' | 'pending' | 'running' | 'failed' | 'start';
    executionTime?: number;
    isDialogue?: boolean;
    error?: string;
}

// Custom Node Component with real-time status updates
const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ data }) => {
    if (data.status === 'start') {
        return (
            <div className="flex items-center justify-center p-4 bg-gray-800 dark:bg-gray-700 text-white rounded-full shadow-lg w-28 h-28 border-2 border-gray-600 dark:border-gray-500">
                <div className="text-center">
                    <PlayCircle className="w-6 h-6 mx-auto mb-1" />
                    <div className="font-bold text-sm">Start</div>
                </div>
                <Handle type="source" position={Position.Bottom} className="!bg-gray-600 dark:!bg-gray-500" />
            </div>
        );
    }

    const isCompleted = data.status === 'completed';
    const isRunning = data.status === 'running';
    const isFailed = data.status === 'failed';
    const isPending = data.status === 'pending';
    const isDialogue = data.isDialogue && isRunning;

    // Get node color based on status - matches project color scheme
    const getNodeColor = () => {
        if (isDialogue) return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-400 dark:border-yellow-500';

        switch (data.status) {
            case 'pending': return 'bg-gray-50 dark:bg-gray-800 border-gray-300 dark:border-gray-600';
            case 'running': return 'bg-blue-50 dark:bg-blue-900/20 border-blue-400 dark:border-blue-500';
            case 'completed': return 'bg-green-50 dark:bg-green-900/20 border-green-400 dark:border-green-500';
            case 'failed': return 'bg-red-50 dark:bg-red-900/20 border-red-400 dark:border-red-500';
            default: return 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600';
        }
    };

    return (
        <div
            className={cn(
                "p-4 rounded-lg border-2 shadow-md transition-all duration-500 ease-in-out",
                getNodeColor(),
                isRunning && "workflow-node-running ring-2 ring-blue-400 dark:ring-blue-500",
                isCompleted && "workflow-node-completed",
                isFailed && "workflow-node-failed"
            )}
            style={{
                minWidth: 280,
                maxWidth: 400,
                width: 'auto',
                minHeight: 100,
                height: 'auto',
                whiteSpace: 'normal',
                wordBreak: 'break-word',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'flex-start',
                gap: '0.5rem',
                transform: isCompleted ? 'scale(1)' : 'scale(1)',
            }}
        >
            <Handle type="target" position={Position.Top} className="!bg-gray-400" />

            {/* Status Icon + Task Name */}
            <div className="flex items-start gap-2">
                {isRunning && <Loader2 className="w-4 h-4 mt-0.5 text-blue-600 dark:text-blue-400 animate-spin flex-shrink-0" />}
                {isCompleted && <CheckCircle className="w-4 h-4 mt-0.5 text-green-600 dark:text-green-400 flex-shrink-0" />}
                {isFailed && <XCircle className="w-4 h-4 mt-0.5 text-red-600 dark:text-red-400 flex-shrink-0" />}
                {isPending && <Clock className="w-4 h-4 mt-0.5 text-gray-400 dark:text-gray-500 flex-shrink-0" />}
                <p className="font-semibold text-sm text-gray-800 dark:text-gray-200 leading-tight">{data.label}</p>
            </div>

            {/* Description */}
            <p className="text-xs text-gray-500 dark:text-gray-400 my-2 line-clamp-2">{data.description}</p>

            {/* Agent Badge - only show if agent has meaningful content */}
            {data.agent &&
                typeof data.agent === 'string' &&
                data.agent.trim().length > 0 &&
                data.agent !== 'N/A' &&
                data.agent !== 'Unknown Agent' &&
                data.agent !== 'null' &&
                data.agent !== 'undefined' && (
                    <div className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-800/90 dark:bg-gray-700/90 border border-gray-700 dark:border-gray-600 backdrop-blur-sm">
                        <span className="text-xs font-medium text-gray-200 dark:text-gray-300">
                            {data.agent.trim()}
                        </span>
                    </div>
                )}

            {/* Execution Time (for completed tasks) */}
            {isCompleted && data.executionTime && (
                <div className="flex items-center gap-1 pt-1">
                    <Zap className="w-3 h-3 text-green-600 dark:text-green-400" />
                    <p className="text-xs text-green-600 dark:text-green-400 font-medium">
                        {data.executionTime.toFixed(2)}s
                    </p>
                </div>
            )}

            {/* Error Message (for failed tasks) */}
            {isFailed && data.error && (
                <div className="flex items-start gap-1 pt-1">
                    <AlertTriangle className="w-3 h-3 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                    <p className="text-xs text-red-600 dark:text-red-400 font-medium line-clamp-2">
                        {data.error.substring(0, 80)}
                    </p>
                </div>
            )}

            {/* Running Indicator */}
            {isRunning && (
                <div className="flex items-center gap-1 pt-1">
                    <Clock className="w-3 h-3 text-blue-600 dark:text-blue-400 animate-pulse" />
                    <p className="text-xs text-blue-600 dark:text-blue-400 font-medium">
                        Executing...
                    </p>
                </div>
            )}

            <Handle
                type="source"
                position={Position.Bottom}
                className="!bg-gray-400 !w-3 !h-3 !border-2 !border-white"
                style={{ bottom: -6 }}
            />
        </div>
    );
};

export default function PlanGraph({ planData, taskStatuses = {} }: PlanGraphProps) {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);

    const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);

    useEffect(() => {
        const { pendingTasks, completedTasks } = planData;

        // ONLY use pendingTasks as the source of truth for the plan structure
        // Real-time status updates come from taskStatuses prop

        console.log('PlanGraph update:', {
            pendingTasksCount: pendingTasks.length,
            completedTasksCount: completedTasks.length,
            taskStatusesCount: Object.keys(taskStatuses).length
        });

        if (pendingTasks.length === 0 && completedTasks.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const initialNodes: Node[] = [];
        const initialEdges: Edge[] = [];

        // Add Start Node
        initialNodes.push({
            id: 'start',
            type: 'custom',
            data: { label: 'Start', status: 'start' },
            position: { x: 400, y: 0 }, // Center at 400
        });

        const yOffset = 220; // Vertical spacing between ranks
        const xNodeWidth = 320; // Width of node + gap

        // --- Process Tasks ---
        // Iterate through batches (ranks)
        // planData.pendingTasks is now (PlanTask | PlanTask[])[]

        // Ensure pendingTasks is treated as an array of batches
        // If the backend sends a flat list, we treat it as single-item batches? 
        // Or we rely on the sidebar to have formatted it correctly.
        // We will assume pendingTasks is (PlanTask | PlanTask[])[]

        let currentY = 1;
        let previousLayerIds: string[] = ['start']; // Connect first layer to 'start'

        pendingTasks.forEach((batchOrTask, rankIndex) => {
            // Normalize to array of tasks (batch)
            // If it's not an array, wrap it (supporting flat lists for backward compat)
            const batch = Array.isArray(batchOrTask) ? batchOrTask : [batchOrTask];

            const currentLayerIds: string[] = [];
            const batchSize = batch.length;

            // Calculate starting X to center the batch
            // Center is 400. 
            // Total width = (batchSize * width) - gap? No, standard grid logic.
            // Let's assume node width 280, gap 40 -> 320 stride.
            // Center X = 400.
            // Start X = 400 - ((batchSize - 1) * 320) / 2

            const startX = 400 - ((batchSize - 1) * 320) / 2;

            batch.forEach((task: any, indexInBatch: number) => {
                const taskName = task.task;

                // Check if we have real-time status for this task
                const taskStatus = taskStatuses[taskName];
                // Fallback to 'pending' if no status found
                const status = taskStatus?.status || task.status || 'pending';

                const nodeId = `task-${taskName}`;
                currentLayerIds.push(nodeId);

                // Position
                const xPos = startX + (indexInBatch * 320);
                const yPos = (rankIndex + 1) * yOffset;

                // Create Node
                initialNodes.push({
                    id: nodeId,
                    type: 'custom',
                    data: {
                        label: taskName,
                        description: task.description || 'Completed',
                        agent: taskStatus?.agentName || task.agent || 'N/A',
                        status: status,
                        executionTime: taskStatus?.executionTime,
                        error: taskStatus?.error,
                        isDialogue: taskStatus?.is_dialogue,
                    },
                    position: { x: xPos, y: yPos },
                });

                // Create Edges from Previous Layer
                previousLayerIds.forEach(prevId => {
                    const shouldAnimate = status === 'pending' || status === 'running';
                    const isDialogue = taskStatus?.is_dialogue && status === 'running';

                    const edgeColor = isDialogue ? '#eab308' : // yellow-500
                        status === 'running' ? '#3b82f6' :  // blue-500
                            status === 'completed' ? '#22c55e' :  // green-500
                                status === 'failed' ? '#ef4444' :     // red-500
                                    '#6b7280';                             // gray-500

                    initialEdges.push({
                        id: `edge-${prevId}-${nodeId}`,
                        source: prevId,
                        target: nodeId,
                        animated: shouldAnimate,
                        style: {
                            strokeWidth: 2,
                            stroke: edgeColor,
                            transition: 'stroke 0.3s ease-in-out',
                        },
                        className: shouldAnimate ? 'workflow-edge-animated' : ''
                    });
                });
            });

            // Current layer becomes previous layer for next iteration
            previousLayerIds = currentLayerIds;
            currentY++;
        });

        // Add End Node if we have tasks
        if (pendingTasks.length > 0) {
            // Calculate final Y based on the last layer
            const finalY = (pendingTasks.length + 1) * yOffset;

            initialNodes.push({
                id: 'end',
                type: 'output', // Use default output node
                data: { label: 'End' },
                position: { x: 400, y: finalY },
                targetPosition: Position.Top,
                style: {
                    background: '#ef4444',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    width: 100,
                    fontSize: '14px',
                    fontWeight: 'bold',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: 50
                }
            });

            // Connect Last Layer to End
            previousLayerIds.forEach(prevId => {
                // Check if ALL tasks are completed
                // We need to flatten pendingTasks to check all statuses
                const allTasksFlat = planData.pendingTasks.flat();
                const allCompleted = allTasksFlat.every((t: any) => {
                    const s = taskStatuses[t.task]?.status || t.status;
                    return s === 'completed';
                });

                initialEdges.push({
                    id: `edge-${prevId}-end`,
                    source: prevId,
                    target: 'end',
                    animated: false,
                    style: { stroke: allCompleted ? '#22c55e' : '#6b7280', strokeWidth: 2 },
                    type: 'smoothstep',
                });
            });
        }

        setNodes(initialNodes);
        setEdges(initialEdges);

    }, [planData, taskStatuses]);

    if (!planData || (planData.pendingTasks.flat().length === 0 && planData.completedTasks.length === 0)) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-700">
                <div className="text-center text-gray-500 dark:text-gray-400 p-8">
                    <div className="mb-4">
                        <svg
                            className="w-16 h-16 mx-auto text-gray-400 dark:text-gray-600 animate-pulse"
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
                    <p className="font-semibold text-lg mb-2 text-gray-700 dark:text-gray-300">Workflow Plan</p>
                    <p className="text-sm text-gray-400 dark:text-gray-500">
                        The execution plan graph will appear here once tasks are identified
                    </p>
                </div>
            </div>
        );
    }

    // Calculate progress
    const totalTasks = planData.pendingTasks.flat().length;
    const completedCount = Object.values(taskStatuses).filter((t: any) => t.status === 'completed').length;
    const runningCount = Object.values(taskStatuses).filter((t: any) => t.status === 'running').length;
    const failedCount = Object.values(taskStatuses).filter((t: any) => t.status === 'failed').length;
    const progress = totalTasks > 0 ? (completedCount / totalTasks) * 100 : 0;

    return (
        <div className="w-full h-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 flex flex-col">
            {/* Progress Bar */}
            {totalTasks > 0 && progress < 100 && (
                <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                            Progress: {completedCount} of {totalTasks} tasks completed
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-500">
                            {runningCount > 0 && `${runningCount} running`}
                            {failedCount > 0 && ` • ${failedCount} failed`}
                        </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                        <div
                            className="bg-gradient-to-r from-green-500 to-green-600 dark:from-green-600 dark:to-green-700 h-2 rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Completion Banner */}
            {progress === 100 && totalTasks > 0 && (
                <div className="px-4 py-3 bg-green-50 dark:bg-green-900/20 border-b border-green-200 dark:border-green-800">
                    <div className="flex items-center gap-2">
                        <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                        <span className="text-sm font-semibold text-green-800 dark:text-green-300">
                            All tasks completed successfully!
                        </span>
                        <Zap className="w-4 h-4 text-yellow-500 dark:text-yellow-400 animate-pulse" />
                    </div>
                </div>
            )}

            {/* Graph */}
            <div className="flex-1 bg-gray-50 dark:bg-gray-900">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    fitView
                    proOptions={{ hideAttribution: true }}
                    className="bg-gray-50 dark:bg-gray-900"
                >
                    <Controls className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700" />
                    <Background
                        color="#9ca3af"
                        className="bg-gray-50 dark:bg-gray-900"
                    />
                </ReactFlow>
            </div>
        </div>
    );
}