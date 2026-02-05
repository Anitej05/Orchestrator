
"use client"

import React, { useEffect, useState } from 'react';
import {
    CheckCircle,
    Clock,
    AlertTriangle,
    Loader2,
    Terminal,
    FileText,
    ChevronDown,
    ChevronUp
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { TaskItem } from "@/lib/types";

interface PlanChecklistProps {
    todoList: TaskItem[];
    currentTaskId?: string;
    isExecuting?: boolean;
}

export default function PlanChecklist({ todoList, currentTaskId, isExecuting }: PlanChecklistProps) {
    // Auto-scroll to active task
    const activeRef = React.useRef<HTMLDivElement>(null);
    const [expandedResults, setExpandedResults] = useState<Record<string, boolean>>({});

    useEffect(() => {
        if (activeRef.current) {
            activeRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, [currentTaskId, todoList.length]);

    const toggleResult = (id: string) => {
        setExpandedResults(prev => ({ ...prev, [id]: !prev[id] }));
    };

    if (!todoList || todoList.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center p-8 text-gray-400">
                <Loader2 className="w-8 h-8 animate-spin mb-2" />
                <p>Initializing Plan...</p>
            </div>
        );
    }

    return (
        <ScrollArea className="h-full w-full pr-4">
            <div className="space-y-3 pb-4">
                {todoList.map((task, index) => {
                    const isActive = task.id === currentTaskId || task.status === 'in_progress';
                    const isCompleted = task.status === 'completed';
                    const isFailed = task.status === 'failed';
                    const isPending = task.status === 'pending';
                    const hasResult = !!task.result;
                    const hasSnippet = !!task.code_snippet;

                    return (
                        <Card
                            key={task.id}
                            ref={isActive ? activeRef : null}
                            className={cn(
                                "p-3 transition-all duration-300 border-l-4",
                                isActive ? "border-l-blue-500 bg-blue-50/50 dark:bg-blue-900/10 shadow-md scale-[1.01]" :
                                    isCompleted ? "border-l-green-500 opacity-80" :
                                        isFailed ? "border-l-red-500" :
                                            "border-l-gray-300 dark:border-l-gray-700 opacity-60 hover:opacity-100"
                            )}
                        >
                            <div className="flex items-start gap-3">
                                {/* Icon Status */}
                                <div className="mt-1 flex-shrink-0">
                                    {isActive && <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />}
                                    {isCompleted && <CheckCircle className="w-5 h-5 text-green-500" />}
                                    {isFailed && <AlertTriangle className="w-5 h-5 text-red-500" />}
                                    {isPending && <Clock className="w-5 h-5 text-gray-400" />}
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex justify-between items-start">
                                        <p className={cn("font-medium text-sm leading-tight", isCompleted && "line-through text-gray-500")}>
                                            {task.description}
                                        </p>
                                        <Badge variant="outline" className="text-[10px] ml-2 flex-shrink-0">
                                            {task.status.toUpperCase()}
                                        </Badge>
                                    </div>

                                    {/* Snippet Preview (Small) */}
                                    {hasSnippet && (
                                        <div className="mt-1 flex items-center gap-1 text-xs text-gray-500 font-mono">
                                            <Terminal className="w-3 h-3" />
                                            <span className="truncate max-w-[200px]">{task.code_snippet}</span>
                                        </div>
                                    )}

                                    {/* Result Expand/Collapse */}
                                    {hasResult && (
                                        <div className="mt-2">
                                            <button
                                                onClick={() => toggleResult(task.id)}
                                                className="flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 hover:underline"
                                            >
                                                {expandedResults[task.id] ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                                {expandedResults[task.id] ? "Hide Result" : "View Result"}
                                            </button>

                                            {expandedResults[task.id] && (
                                                <div className="mt-2 p-2 bg-black/5 dark:bg-black/30 rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap max-h-40">
                                                    {task.result}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </Card>
                    );
                })}
            </div>

            {/* Loading Indicator at bottom if still planning/executing and list is short */}
            {isExecuting && (
                <div className="flex items-center justify-center py-4 text-xs text-gray-400 animate-pulse">
                    <Loader2 className="w-3 h-3 mr-2 animate-spin" />
                    Orchestrator is working...
                </div>
            )}
        </ScrollArea>
    );
}
