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
import { CheckCircle, Clock, PlayCircle } from "lucide-react"

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
    planData: Plan
}

interface CustomNodeData {
    label: string;
    description: string;
    agent: string;
    status: 'completed' | 'pending' | 'start';
}

// Custom Node Component for a polished look
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

    return (
        <div
            className="p-3 bg-white rounded-md border shadow-sm"
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
            }}
        >
            <Handle type="target" position={Position.Top} className="!bg-gray-400" />
            <div className="flex items-center mb-2">
                {isCompleted ? (
                    <CheckCircle className="w-4 h-4 mr-2 text-green-600 flex-shrink-0" />
                ) : (
                    <Clock className="w-4 h-4 mr-2 text-yellow-600" />
                )}
                <p className="font-semibold text-sm text-gray-800">{data.label}</p>
            </div>
            <p className="text-xs text-gray-500 my-1">{data.description}</p>
            <Badge variant="outline">{data.agent}</Badge>
            <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
        </div>
    );
};

export default function PlanGraph({ planData }: PlanGraphProps) {
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
            initialNodes.push({
                id: nodeId,
                type: 'custom',
                data: {
                    label: task.task,
                    description: (task as PlanTask).description || 'Completed',
                    agent: (task as PlanTask).agent || 'N/A',
                    status: task.status,
                },
                position: { x: xPos, y: (index + 1) * yOffset },
            });

            initialEdges.push({
                id: `edge-${previousNodeId}-${nodeId}`,
                source: previousNodeId,
                target: nodeId,
                animated: task.status === 'pending',
                style: { strokeWidth: 2 }
            });

            previousNodeId = nodeId;
        });

        setNodes(initialNodes);
        setEdges(initialEdges);

    }, [planData]);

    if (!planData || (planData.pendingTasks.length === 0 && planData.completedTasks.length === 0)) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-gray-100 rounded-lg border-2 border-dashed">
                <div className="text-center text-gray-500">
                    <p className="font-semibold">Orchestrator's Plan</p>
                    <p className="text-sm mt-2">The execution plan graph will be displayed here.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="w-full h-full rounded-lg border bg-white">
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
    );
}