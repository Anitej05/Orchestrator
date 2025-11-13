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
    pendingTasks: PlanTask[];
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
    error?: string;
}

// Custom Node Component with real-time status updates
const CustomNode: React.FC<NodeProps<CustomNodeData>> = ({ data }) => {
    if (data.status === 'start') {
        return (
            <div className="flex items-center justify-center p-4 bg-blue-500 text-white rounded-full shadow-lg w-28 h-28">
                <div className="text-center">
                    <PlayCircle className="w-6 h-6 mx-auto mb-1" />
                    <div className="font-bold text-sm">Start</div>
                </div>
                <Handle type="source" position={Position.Bottom} className="!bg-blue-500" />
            </div>
        );
    }

    const isCompleted = data.status === 'completed';
    const isRunning = data.status === 'running';
    const isFailed = data.status === 'failed';
    const isPending = data.status === 'pending';

    // Get node color based on status
    const getNodeColor = () => {
        switch (data.status) {
            case 'pending': return 'bg-gray-50 border-gray-300';
            case 'running': return 'bg-yellow-50 border-yellow-400';
            case 'completed': return 'bg-green-50 border-green-400';
            case 'failed': return 'bg-red-50 border-red-400';
            default: return 'bg-white border-gray-300';
        }
    };

    return (
        <div
            className={cn(
                "p-3 rounded-md border-2 shadow-sm transition-all duration-500 ease-in-out",
                getNodeColor(),
                isRunning && "workflow-node-running ring-2 ring-yellow-400",
                isCompleted && "workflow-node-completed",
                isFailed && "workflow-node-failed"
            )}
            style={{
                minWidth: 220,
                maxWidth: 320,
                width: 'auto',
                minHeight: 64,
                height: 'auto',
                whiteSpace: 'normal',
                wordBreak: 'break-word',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                transform: isCompleted ? 'scale(1)' : 'scale(1)',
            }}
        >
            <Handle type="target" position={Position.Top} className="!bg-gray-400" />
            
            {/* Status Icon + Task Name */}
            <div className="flex items-center mb-2">
                {isRunning && <Loader2 className="w-4 h-4 mr-2 text-yellow-600 animate-spin flex-shrink-0" />}
                {isCompleted && <CheckCircle className="w-4 h-4 mr-2 text-green-600 flex-shrink-0" />}
                {isFailed && <XCircle className="w-4 h-4 mr-2 text-red-600 flex-shrink-0" />}
                {isPending && <Clock className="w-4 h-4 mr-2 text-gray-400 flex-shrink-0" />}
                <p className="font-semibold text-sm text-gray-800">{data.label}</p>
            </div>
            
            {/* Description */}
            <p className="text-xs text-gray-500 my-1">{data.description}</p>
            
            {/* Agent Badge */}
            <Badge variant="outline" className="mt-1">{data.agent}</Badge>
            
            {/* Execution Time (for completed tasks) */}
            {isCompleted && data.executionTime && (
                <div className="flex items-center gap-2 mt-2">
                    <p className="text-xs text-green-600 font-medium flex items-center gap-1">
                        <Zap className="w-3 h-3" />
                        {data.executionTime}s
                    </p>
                </div>
            )}
            
            {/* Error Message (for failed tasks) */}
            {isFailed && data.error && (
                <p className="text-xs text-red-600 mt-2 font-medium">
                    ⚠️ {data.error.substring(0, 50)}...
                </p>
            )}
            
            {/* Running Indicator */}
            {isRunning && (
                <p className="text-xs text-yellow-600 mt-2 font-medium animate-pulse">
                    ⏱️ Executing...
                </p>
            )}
            
            <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
        </div>
    );
};

export default function PlanGraph({ planData, taskStatuses = {} }: PlanGraphProps) {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);

    const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);

    useEffect(() => {
        const { pendingTasks, completedTasks } = planData;
        const allTasks = [
            ...completedTasks.map(t => ({...t, status: 'completed' as const})),
            ...pendingTasks.map(t => ({...t, status: 'pending' as const})),
        ];

        if (allTasks.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const initialNodes: Node[] = [];
        const initialEdges: Edge[] = [];

        const yOffset = 150;
        const xPos = 100;

        // Add a Start node
        initialNodes.push({
            id: 'start',
            type: 'custom',
            data: { label: 'Start', status: 'start' },
            position: { x: xPos, y: 0 },
        });

        let previousNodeId = 'start';

        allTasks.forEach((task, index) => {
            const nodeId = `task-${index}`;
            const taskName = task.task;
            
            // Check if we have real-time status for this task
            const taskStatus = taskStatuses[taskName];
            const status = taskStatus?.status || task.status;
            
            initialNodes.push({
                id: nodeId,
                type: 'custom',
                data: {
                    label: taskName,
                    description: (task as PlanTask).description || 'Completed',
                    agent: taskStatus?.agentName || (task as PlanTask).agent || 'N/A',
                    status: status,
                    executionTime: taskStatus?.executionTime,
                    error: taskStatus?.error,
                },
                position: { x: xPos, y: (index + 1) * yOffset },
            });

            // Animate edge when task is running or pending
            const shouldAnimate = status === 'pending' || status === 'running';
            const edgeColor = status === 'running' ? '#facc15' : 
                             status === 'completed' ? '#22c55e' : 
                             status === 'failed' ? '#ef4444' : '#9ca3af';
            
            initialEdges.push({
                id: `edge-${previousNodeId}-${nodeId}`,
                source: previousNodeId,
                target: nodeId,
                animated: shouldAnimate,
                style: { 
                    strokeWidth: 2,
                    stroke: edgeColor,
                    transition: 'stroke 0.3s ease-in-out',
                },
                className: shouldAnimate ? 'workflow-edge-animated' : ''
            });

            previousNodeId = nodeId;
        });

        setNodes(initialNodes);
        setEdges(initialEdges);

    }, [planData, taskStatuses]);

    if (!planData || (planData.pendingTasks.length === 0 && planData.completedTasks.length === 0)) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg border-2 border-dashed border-gray-300">
                <div className="text-center text-gray-500 p-8">
                    <div className="mb-4">
                        <svg 
                            className="w-16 h-16 mx-auto text-gray-400 animate-pulse" 
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
                    <p className="font-semibold text-lg mb-2">Workflow Plan</p>
                    <p className="text-sm text-gray-400">
                        The execution plan graph will appear here once tasks are identified
                    </p>
                </div>
            </div>
        );
    }

    // Calculate progress
    const totalTasks = planData.pendingTasks.length + planData.completedTasks.length;
    const completedCount = planData.completedTasks.length;
    const runningCount = Object.values(taskStatuses).filter(t => t.status === 'running').length;
    const failedCount = Object.values(taskStatuses).filter(t => t.status === 'failed').length;
    const progress = totalTasks > 0 ? (completedCount / totalTasks) * 100 : 0;

    return (
        <div className="w-full h-full rounded-lg border bg-white flex flex-col">
            {/* Progress Bar */}
            {totalTasks > 0 && progress < 100 && (
                <div className="px-4 py-2 border-b bg-gray-50">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-medium text-gray-600">
                            Progress: {completedCount} of {totalTasks} tasks completed
                        </span>
                        <span className="text-xs text-gray-500">
                            {runningCount > 0 && `${runningCount} running`}
                            {failedCount > 0 && ` • ${failedCount} failed`}
                        </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                        <div 
                            className="bg-gradient-to-r from-green-500 to-green-600 h-2 rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>
            )}
            
            {/* Completion Banner */}
            {progress === 100 && totalTasks > 0 && (
                <div className="px-4 py-3 bg-gradient-to-r from-green-50 to-emerald-50 border-b border-green-200">
                    <div className="flex items-center gap-2">
                        <CheckCircle className="w-5 h-5 text-green-600" />
                        <span className="text-sm font-semibold text-green-800">
                            All tasks completed successfully!
                        </span>
                        <Zap className="w-4 h-4 text-yellow-500 animate-pulse" />
                    </div>
                </div>
            )}
            
            {/* Graph */}
            <div className="flex-1">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    fitView
                    proOptions={{ hideAttribution: true }}
                >
                    <Controls />
                    <Background />
                </ReactFlow>
            </div>
        </div>
    );
}