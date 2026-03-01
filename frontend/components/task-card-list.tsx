"use client"

import { useMemo, useState } from "react"
import { ChevronDown, ChevronUp, CheckCircle2, Circle, AlertTriangle, Loader2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { ActionHistoryEntry, TaskStatus, TodoItem } from "@/lib/types"

type GenericTask = Record<string, any>

interface NormalizedTask {
  id: string
  title: string
  description: string
  status: string
  priority?: string | number
  agentName?: string
  toolName?: string
  executionTime?: number
  startedAt?: Date
  completedAt?: Date
  createdAt?: string
  updatedAt?: string
  result?: any
  error?: string
}

interface TaskCardListProps {
  todoList?: TodoItem[]
  taskStatuses?: Record<string, TaskStatus>
  actionHistory?: ActionHistoryEntry[]
  fallbackTasks?: GenericTask[]
  emptyTitle?: string
  emptySubtitle?: string
  className?: string
}

function normalizeStatus(status?: string): string {
  const statusText = (status || "pending").toLowerCase()
  if (statusText === "running") return "in_progress"
  if (statusText === "in-progress") return "in_progress"
  return statusText
}

function formatDate(value?: Date | string): string | undefined {
  if (!value) return undefined
  const parsedDate = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(parsedDate.getTime())) return undefined
  return parsedDate.toLocaleTimeString()
}

function formatDuration(executionTime?: number): string | undefined {
  if (typeof executionTime !== "number" || Number.isNaN(executionTime)) return undefined
  if (executionTime > 1000) return `${(executionTime / 1000).toFixed(2)}s`
  return `${executionTime.toFixed(0)}ms`
}

function getStatusIcon(status: string) {
  switch (normalizeStatus(status)) {
    case "completed":
      return <CheckCircle2 className="w-4 h-4 text-status-success" />
    case "in_progress":
      return <Loader2 className="w-4 h-4 text-status-active animate-spin" />
    case "failed":
      return <AlertTriangle className="w-4 h-4 text-status-error" />
    default:
      return <Circle className="w-4 h-4 text-text-disabled" />
  }
}

function getStatusBadgeVariant(status: string): "ui-complete" | "ui-active" | "destructive" | "ui-pending" {
  switch (normalizeStatus(status)) {
    case "completed":
      return "ui-complete"
    case "in_progress":
      return "ui-active"
    case "failed":
      return "destructive"
    default:
      return "ui-pending"
  }
}

function resolveTaskStatus(
  task: GenericTask,
  taskStatuses: Record<string, TaskStatus>,
  index: number
): TaskStatus | undefined {
  const candidateKeys = [
    task.task_id,
    task.id,
    task.taskName,
    task.task_name,
    task.task,
    task.title,
    `${index}`,
  ].filter(Boolean)

  for (const key of candidateKeys) {
    const matchedStatus = taskStatuses[String(key)]
    if (matchedStatus) return matchedStatus
  }

  return undefined
}

function formatResourceName(resourceId: string): string {
  if (!resourceId) return 'Orbimesh Brain'
  
  return resourceId
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}

function normalizeTask(
  task: GenericTask,
  taskStatuses: Record<string, TaskStatus>,
  index: number,
  actionHistory: ActionHistoryEntry[]
): NormalizedTask {
  const taskStatus = resolveTaskStatus(task, taskStatuses, index)
  const latestAction = actionHistory[actionHistory.length - 1]

  const taskId = String(task.task_id || task.id || task.taskName || task.task_name || `task-${index}`)
  
  // Get task title (the main goal/objective)
  const taskTitle = String(
    task.description ||
      task.task_description ||
      task.task ||
      task.task_name ||
      task.taskName ||
      taskStatus?.taskDescription ||
      taskStatus?.taskName ||
      `Task ${index + 1}`
  )

  // Get agent/tool name
  const agentName =
    task.payload?.agent_name ||
    task.assigned_to ||
    task.assigned_agent ||
    task.agent ||
    taskStatus?.agentName ||
    latestAction?.resource_id

  const toolName = task.assigned_tool || task.payload?.tool_name

  // Create display name: Agent first, fallback to tool, fallback to "Orbimesh Brain"
  let displayName = "Orbimesh Brain"
  if (agentName) {
    displayName = formatResourceName(agentName)
  } else if (toolName) {
    displayName = formatResourceName(toolName)
  }

  // Use activity description for live updates (clean one-liner)
  const activityDescription = 
    taskStatus?.activityDescription ||
    (taskTitle ? taskTitle.split('.')[0] : "") ||
    taskTitle

  // Truncate description to 2 lines max (roughly 120 chars)
  const maxLength = 120
  let shortDescription = activityDescription
  if (shortDescription.length > maxLength) {
    shortDescription = shortDescription.substring(0, maxLength).trim() + "..."
  }

  const taskStatusValue = normalizeStatus(
    task.status ||
      taskStatus?.status ||
      "pending"
  )

  return {
    id: taskId,
    title: taskTitle,
    description: shortDescription,
    status: taskStatusValue,
    priority: task.priority,
    agentName: displayName,  // This is now the formatted display name
    toolName,
    executionTime: task.execution_time || taskStatus?.executionTime,
    startedAt: taskStatus?.startedAt,
    completedAt: taskStatus?.completedAt,
    createdAt: typeof task.created_at === "string" ? task.created_at : undefined,
    updatedAt: typeof task.updated_at === "string" ? task.updated_at : undefined,
    result: taskStatus?.resultSummary,
    error: task.error || taskStatus?.error,
  }
}

export default function TaskCardList({
  todoList = [],
  taskStatuses = {},
  actionHistory = [],
  fallbackTasks = [],
  emptyTitle = "No Tasks",
  emptySubtitle = "Tasks will appear here once execution starts",
  className,
}: TaskCardListProps) {
  const [expandedTaskIds, setExpandedTaskIds] = useState<Record<string, boolean>>({})

  const baseTasks = todoList.length > 0 ? todoList : fallbackTasks

  const normalizedTasks = useMemo(
    () => baseTasks.map((task, index) => normalizeTask(task as GenericTask, taskStatuses, index, actionHistory)),
    [baseTasks, taskStatuses, actionHistory]
  )

  // Calculate progress
  const completedCount = normalizedTasks.filter(t => t.status === "completed").length
  const totalCount = normalizedTasks.length

  if (normalizedTasks.length === 0) {
    return (
      <div className="text-center text-text-tertiary py-8">
        <p className="text-orbimesh-section-header font-semibold mb-2">{emptyTitle}</p>
        <p className="text-orbimesh-section-subtitle">{emptySubtitle}</p>
      </div>
    )
  }

  return (
    <div className={cn("space-y-3", className)}>
      {/* Progress counter */}
      {totalCount > 0 && (
        <div className="text-xs text-text-secondary font-medium px-1">
          {completedCount} of {totalCount} tasks completed
        </div>
      )}

      {normalizedTasks.map((task) => {
        const isExpanded = !!expandedTaskIds[task.id]

        return (
          <div key={task.id} className="rounded-lg border border-border-color bg-bg-card overflow-hidden">
            <button
              type="button"
              className="w-full px-4 py-3 text-left hover:bg-bg-hover transition-colors"
              onClick={() =>
                setExpandedTaskIds((previous) => ({
                  ...previous,
                  [task.id]: !previous[task.id],
                }))
              }
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 flex-shrink-0">{getStatusIcon(task.status)}</div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-3 mb-1">
                    <p className="font-semibold text-sm text-text-primary truncate">
                      {task.agentName}
                    </p>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <Badge variant={getStatusBadgeVariant(task.status)} className="capitalize text-xs">
                        {task.status.replace("_", " ")}
                      </Badge>
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-text-tertiary" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-text-tertiary" />
                      )}
                    </div>
                  </div>
                  <p className="text-xs text-text-secondary line-clamp-2">{task.description}</p>
                </div>
              </div>
            </button>

            {isExpanded && (
              <div className="px-4 pb-4 pt-3 border-t border-border-color bg-bg-subtle/40 space-y-3">
                {/* Full task description */}
                <div className="text-sm text-text-primary">
                  <p className="font-medium mb-1">Task:</p>
                  <p className="text-text-secondary">{task.title}</p>
                </div>

                {/* Agent and execution info */}
                <div className="flex flex-wrap items-center gap-2">
                  {task.agentName && (
                    <Badge variant="secondary" className="text-xs">
                      {task.agentName}
                    </Badge>
                  )}
                  {formatDuration(task.executionTime) && (
                    <Badge variant="outline" className="text-xs">
                      ⏱️ {formatDuration(task.executionTime)}
                    </Badge>
                  )}
                </div>

                {/* Error message */}
                {task.error && (
                  <div className="text-xs p-2 rounded bg-status-error-light text-status-error-dark border border-status-error-light break-words">
                    <span className="font-semibold">Error:</span> {String(task.error)}
                  </div>
                )}

                {/* Result summary (user-friendly, not raw JSON) */}
                {task.status === "completed" && task.result && (
                  <div className="text-xs p-2 rounded bg-status-success-light text-status-success-dark border border-status-success-light break-words">
                    <span className="font-semibold">Result:</span> {String(task.result)}
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
