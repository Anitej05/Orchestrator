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
import { MdMail, MdTableChart, MdDescription } from "react-icons/md"
import { cn } from "@/lib/utils"
import type { TaskStatus } from "@/lib/types"

interface PlanTask {
    task: string;
    description: string;
    agent: string;
    short_description?: string;  // AI-generated summary from backend
    agent_image_url?: string;    // Agent avatar URL
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
    short_description?: string;  // Short task summary
    icon_name?: string;          // React icon name (e.g., "MdMail")
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

// Hook to fetch agent image URL from database (COMMENTED OUT - using React Icons instead)
/*
const useAgentImageUrl = (agentId: string, initialUrl?: string) => {
    const [imageUrl, setImageUrl] = useState<string | undefined>(initialUrl);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        // If we already have a URL, use it
        if (initialUrl) {
            setImageUrl(initialUrl);
            return;
        }

        // Otherwise, fetch from API
        if (!agentId) return;

        setLoading(true);
        fetch(`/api/agents/${encodeURIComponent(agentId)}/image-url`)
            .then(res => {
                if (!res.ok) throw new Error('Failed to fetch image URL');
                return res.json();
            })
            .then(data => {
                if (data.image_url) {
                    setImageUrl(data.image_url);
                }
            })
            .catch(err => {
                console.debug(`Could not fetch image for agent ${agentId}:`, err);
            })
            .finally(() => setLoading(false));
    }, [agentId, initialUrl]);

    return imageUrl;
};
*/

// Custom Node Component with real-time status updates
const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ data }) => {
    const isCompleted = data.status === 'completed';
    const isRunning = data.status === 'running';
    const isFailed = data.status === 'failed';
    const isPending = data.status === 'pending';
    const isDialogue = data.isDialogue && isRunning;

    // COMMENTED OUT: Removed image URL fetching - now using React Icons instead
    // const agentImageUrl = useAgentImageUrl(data.agent, data.agent_image_url);

    // Get node color based on status - matches project color scheme
    const getNodeColor = () => {
        if (isDialogue) return 'bg-status-pending-light dark:bg-status-pending-dark/20 border-status-pending-dark dark:border-status-pending';

        switch (data.status) {
            case 'pending': return 'bg-bg-subtle dark:bg-gray-800 border-border-color-medium dark:border-gray-700';
            case 'running': return 'bg-status-active-light dark:bg-status-active-dark/20 border-status-active-border dark:border-status-active';
            case 'completed': return 'bg-status-success-light dark:bg-status-success-dark/20 border-status-success-border dark:border-status-success';
            case 'failed': return 'bg-status-error-light dark:bg-status-error/20 border-status-error dark:border-status-error';
            default: return 'bg-bg-card dark:bg-gray-800 border-border-color-medium dark:border-gray-700';
        }
    };

    return (
        <div
            className={cn(
                "relative p-4 rounded-lg border-2 shadow-md transition-all duration-500 ease-in-out",
                getNodeColor(),
                isRunning && "workflow-node-running ring-2 ring-status-active",
                isCompleted && "workflow-node-completed",
                isFailed && "workflow-node-failed"
            )}
            style={{
                minWidth: 380,
                maxWidth: 480,
                width: 'auto',
                minHeight: 120,
                height: 'auto',
                whiteSpace: 'normal',
                wordBreak: 'break-word',
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'flex-start',
                gap: '1rem',
                transform: isCompleted ? 'scale(1)' : 'scale(1)',
            }}
        >
            {/* Completion Tick Mark - top right */}
            {isCompleted && (
                <div className="absolute -top-2 -right-2 bg-status-success-dark rounded-full p-1.5 z-10 shadow-md">
                    <CheckCircle className="w-5 h-5 text-white" />
                </div>
            )}

            <Handle type="target" position={Position.Top} className="!bg-border-color-medium" />

            {/* Left Section: Agent Icon (48×48px using React Icons) */}
            <div className="shrink-0 flex items-center justify-center w-16 h-16 rounded-lg bg-gradient-to-br from-status-active-light to-status-active-border dark:from-gray-700 dark:to-gray-600 shadow-sm border border-border-light dark:border-gray-600">
                {(() => {
                    const IconComponent = getIconComponent(data.icon_name);
                    if (IconComponent) {
                        return <IconComponent className="w-8 h-8 text-status-active-dark dark:text-gray-300" />;
                    }
                    // Fallback placeholder
                    return (
                        <svg className="w-8 h-8 text-text-tertiary dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                    );
                })()}
            </div>

            {/* Right Section: Task Information */}
            <div className="flex-1 flex flex-col justify-between min-w-0 gap-1">
                
                {/* Task Name */}
                <p className="font-semibold text-orbimesh-task-name text-text-primary dark:text-gray-100 leading-tight truncate">
                    {data.label}
                </p>

                {/* Short Description */}
                <p className="text-orbimesh-task-description text-text-secondary dark:text-gray-300 text-xs leading-relaxed line-clamp-2">
                    {data.short_description || data.description}
                </p>

                {/* Agent Name */}
                {data.agent && (
                    <p className="text-xs text-text-tertiary dark:text-gray-400 truncate">
                        {data.agent}
                    </p>
                )}

                {/* Execution Time / Status Badge */}
                <div className="flex items-center gap-2 pt-1">
                    {isCompleted && data.executionTime && (
                        <div className="flex items-center gap-1">
                            <Zap className="w-3 h-3 text-status-success-dark" />
                            <p className="text-orbimesh-file-meta text-status-success-dark font-medium text-xs">
                                {data.executionTime.toFixed(2)}s
                            </p>
                        </div>
                    )}
                    
                    {isRunning && (
                        <div className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-status-active-dark animate-pulse" />
                            <p className="text-orbimesh-file-meta text-status-active-dark font-medium text-xs">
                                Executing...
                            </p>
                        </div>
                    )}
                    
                    {isFailed && data.error && (
                        <div className="flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3 text-status-error" />
                            <p className="text-orbimesh-file-meta text-status-error font-medium text-xs line-clamp-1">
                                {data.error.substring(0, 50)}...
                            </p>
                        </div>
                    )}
                </div>
            </div>

            <Handle
                type="source"
                position={Position.Bottom}
                className="!bg-border-color-medium !w-3 !h-3 !border-2 !border-bg-card"
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
        let previousLayerIds: string[] = []; // First layer has no predecessors (no start node)

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
                        short_description: task.short_description,
                        icon_name: task.icon_name,
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

                    const edgeColor = isDialogue ? 'var(--color-status-pending)' :
                        status === 'running' ? 'var(--color-status-active)' :
                            status === 'completed' ? 'var(--color-status-success)' :
                                status === 'failed' ? 'var(--color-status-error)' :
                                    'var(--color-text-tertiary)';

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

        setNodes(initialNodes);
        setEdges(initialEdges);

    }, [planData, taskStatuses]);

    if (!planData || (planData.pendingTasks.flat().length === 0 && planData.completedTasks.length === 0)) {
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
        <div className="w-full h-full rounded-lg border border-border-color bg-bg-card flex flex-col">
            {/* Progress Bar */}
            {totalTasks > 0 && progress < 100 && (
                <div className="px-4 py-2 border-b border-border-color bg-bg-subtle">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-orbimesh-metadata-label font-medium text-text-secondary">
                            Progress: {completedCount} of {totalTasks} tasks completed
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
