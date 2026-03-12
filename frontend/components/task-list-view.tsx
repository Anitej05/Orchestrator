"use client"

import type { FC } from 'react'

// Task List View Component - Display real todo_list from Brain
export interface TaskListViewProps {
    todoList: any[];
    taskStatuses: Record<string, any>;
    pendingTasks: any[];
}

const TaskListView: FC<TaskListViewProps> = ({ todoList, taskStatuses, pendingTasks }) => {
    // Use todoList if available (new system from Brain), otherwise fall back to pendingTasks
    const tasks = todoList.length > 0 ? todoList : pendingTasks.flat();

    const getStatusIcon = (status: any) => {
        const statusStr = typeof status === 'string' ? status : String(status || '');
        switch (statusStr.toLowerCase()) {
            case 'completed':
                return <span className="text-status-success text-lg">✓</span>;
            case 'in_progress':
            case 'in-progress':
                return <span className="text-status-active text-lg animate-pulse">●</span>;
            case 'failed':
                return <span className="text-status-error text-lg">✕</span>;
            case 'blocked':
                return <span className="text-status-warning text-lg">⊘</span>;
            case 'skipped':
                return <span className="text-text-tertiary text-lg">-</span>;
            case 'pending':
            default:
                return <span className="text-text-disabled text-lg">○</span>;
        }
    };

    const getStatusColor = (status: any) => {
        const statusStr = typeof status === 'string' ? status : String(status || '');
        switch (statusStr.toLowerCase()) {
            case 'completed':
                return 'text-status-success';
            case 'in_progress':
            case 'in-progress':
                return 'text-status-active';
            case 'failed':
                return 'text-status-error';
            case 'blocked':
                return 'text-status-warning';
            case 'skipped':
                return 'text-text-tertiary';
            case 'pending':
            default:
                return 'text-text-secondary';
        }
    };

    const getPriorityBadge = (priority?: any) => {
        if (!priority) return null;
        // Ensure priority is a string (handle enum, object, or string)
        const priorityStr = typeof priority === 'string' ? priority : String(priority);
        const priorityLower = priorityStr.toLowerCase();
        const colors: Record<string, string> = {
            'critical': 'bg-status-error-light text-status-error-dark',
            'high': 'bg-status-warning-light text-status-warning-dark',
            'medium': 'bg-status-pending-light text-status-pending-dark',
            'low': 'bg-bg-subtle text-text-tertiary',
        };
        return (
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium capitalize ${colors[priorityLower] || colors['medium']}`}>
                {priorityStr}
            </span>
        );
    };

    const getToolBadge = (assignedTool?: any) => {
        if (!assignedTool) return null;
        const toolStr = typeof assignedTool === 'string' ? assignedTool : String(assignedTool);
        return (
            <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-bg-card text-text-secondary border border-border-color">
                {toolStr}
            </span>
        );
    };

    if (tasks.length === 0) {
        return (
            <div className="text-center text-text-tertiary py-8">
                <p className="text-orbimesh-section-header font-semibold mb-2">No Tasks</p>
                <p className="text-orbimesh-section-subtitle">Tasks will appear here once the workflow creates them</p>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {tasks.map((task: any, index: number) => {
                // Real todo_list structure from Backend TaskItem
                // Backend uses 'task_id' field, not 'id'
                const taskId = task.task_id || task.id;
                const description = task.description || 'Unknown task';
                const status = task.status || 'pending';
                const priority = task.priority;
                const assignedTool = task.assigned_tool;
                const dependencies = task.dependencies || [];
                const result = task.result;
                const error = task.error;

                return (
                    <div
                        key={taskId || index}
                        className="p-4 rounded-lg border border-border-color bg-bg-card hover:bg-bg-hover transition-colors max-w-full overflow-hidden"
                    >
                        <div className="flex items-start gap-3 min-w-0">
                            {/* Status Icon */}
                            <div className="flex-shrink-0 mt-0.5">
                                {getStatusIcon(status)}
                            </div>

                            <div className="flex-1 min-w-0 overflow-hidden">
                                {/* Task Title + ID */}
                                <div className="flex items-start justify-between gap-2 mb-2">
                                    <h4 className={`font-semibold leading-snug break-words ${getStatusColor(status)}`}>
                                        {description}
                                    </h4>
                                    <span className="flex-shrink-0 text-xs text-text-disabled font-mono bg-bg-subtle px-2 py-1 rounded">
                                        #{taskId}
                                    </span>
                                </div>

                                {/* Priority + Tool + Status badges */}
                                <div className="flex flex-wrap items-center gap-2 mb-3">
                                    {getPriorityBadge(priority)}
                                    {getToolBadge(assignedTool)}
                                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium capitalize ${
                                        String(status).toLowerCase().includes('in_progress') || String(status).toLowerCase().includes('in-progress')
                                            ? 'animate-pulse bg-status-active-light text-status-active-dark'
                                            : 'bg-bg-subtle text-text-tertiary'
                                    }`}>
                                        {String(status).replace('_', ' ')}
                                    </span>
                                </div>

                                {/* Dependencies info */}
                                {dependencies && dependencies.length > 0 && (
                                    <div className="text-xs text-text-tertiary mb-2">
                                        <span className="font-medium">Depends on:</span> {dependencies.join(', ')}
                                    </div>
                                )}

                                {/* Error Message */}
                                {error && (
                                    <div className="mt-2 p-2 bg-status-error-light rounded text-xs text-status-error-dark font-medium break-words overflow-hidden">
                                        <span className="font-bold">Error:</span> {error}
                                    </div>
                                )}

                                {/* Result Summary */}
                                {result && String(status).toLowerCase() === 'completed' && (
                                    <div className="mt-2 p-2 bg-status-success-light rounded text-xs text-status-success-dark break-words overflow-hidden">
                                        <span className="font-bold">Result:</span>{' '}
                                        <span className="break-all">
                                            {typeof result === 'string'
                                                ? (result.length > 150 ? result.substring(0, 150) + '...' : result)
                                                : (JSON.stringify(result, null, 2).length > 150
                                                    ? JSON.stringify(result).substring(0, 150) + '...'
                                                    : JSON.stringify(result, null, 2))
                                            }
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default TaskListView;
