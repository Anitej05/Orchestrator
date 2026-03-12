"use client"

/**
 * TaskCardList — Clean step-by-step execution progress
 *
 * Design: Manus / Claude Code style
 *   ○  pending  — dim numbered circle, gray text
 *   ◉  running  — blue highlighted row, spinner, live text pulses
 *   ✓  done     — green check, dimmed text, 1-line result
 *   ✕  failed   — red triangle, red text, error message
 *
 * Completion: green "All N tasks completed" banner when done.
 */

import { useMemo } from "react"
import { CheckCircle2, AlertTriangle, Loader2, BrainCircuit, Clock, CheckCheck } from "lucide-react"
import { cn } from "@/lib/utils"
import type { ActionHistoryEntry, TaskStatus, TodoItem } from "@/lib/types"

type GenericTask = Record<string, any>

interface NormalizedTask {
  id: string
  title: string
  liveText: string
  status: string
  agentName?: string
  executionTime?: number
  result?: string
  error?: string
}

interface TaskCardListProps {
  todoList?: TodoItem[]
  taskStatuses?: Record<string, TaskStatus>
  actionHistory?: ActionHistoryEntry[]
  fallbackTasks?: GenericTask[]
  brainReasoning?: string
  emptyTitle?: string
  emptySubtitle?: string
  className?: string
}

// ── Helpers ────────────────────────────────────────────────────────────────

function normalizeStatus(s?: string): string {
  const v = (s || "pending").toLowerCase()
  if (v === "running" || v === "in-progress" || v === "in_progress") return "in_progress"
  if (v.includes("complet")) return "completed"
  if (v.includes("fail")) return "failed"
  return v
}

function formatDuration(ms?: number): string | undefined {
  if (typeof ms !== "number" || Number.isNaN(ms)) return undefined
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`
}

function resolveTaskStatus(
  task: GenericTask,
  statuses: Record<string, TaskStatus>,
  index: number
): TaskStatus | undefined {
  const keys = [task.task_id, task.id, task.taskName, task.task_name, task.task, task.title, `${index}`].filter(Boolean)
  for (const k of keys) {
    const s = statuses[String(k)]
    if (s) return s
  }
  return undefined
}

function formatName(id: string): string {
  if (!id) return "Orbimesh Brain"
  return id.replace(/_/g, " ").split(" ").map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(" ")
}

function normalizeTask(
  task: GenericTask,
  statuses: Record<string, TaskStatus>,
  index: number
): NormalizedTask {
  const ts = resolveTaskStatus(task, statuses, index)

  const id = String(task.task_id || task.id || task.taskName || task.task_name || `task-${index}`)

  const title = String(
    task.description || task.task_description || task.task || task.task_name || task.taskName ||
    ts?.taskDescription || ts?.taskName || `Task ${index + 1}`
  )

  const resolvedStatus = normalizeStatus(ts?.status || task.status || "pending")

  const agentId = task.payload?.agent_name || task.assigned_to || task.assigned_agent || task.agent || ts?.agentName
  const agentName = agentId ? formatName(agentId) : undefined

  const liveRaw = ts?.activityDescription || ""
  const liveText = liveRaw.length > 120 ? liveRaw.slice(0, 120) + "…" : liveRaw

  return {
    id,
    title,
    liveText,
    status: resolvedStatus,
    agentName,
    executionTime: task.execution_time || ts?.executionTime,
    result: ts?.resultSummary,
    error: task.error || ts?.error,
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export default function TaskCardList({
  todoList = [],
  taskStatuses = {},
  actionHistory = [],
  fallbackTasks = [],
  brainReasoning,
  emptyTitle = "No Tasks",
  emptySubtitle = "Tasks will appear here once execution starts",
  className,
}: TaskCardListProps) {
  const baseTasks = todoList.length > 0 ? todoList : fallbackTasks

  const tasks = useMemo(
    () => baseTasks.map((t, i) => normalizeTask(t as GenericTask, taskStatuses, i)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(baseTasks), JSON.stringify(taskStatuses)]
  )

  const doneCount = tasks.filter((t) => t.status === "completed").length
  const total = tasks.length
  const allDone = total > 0 && doneCount === total

  // ── Empty state ──────────────────────────────────────────────────────────
  if (tasks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-center">
        <div className="w-10 h-10 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center mb-3">
          <span className="text-lg text-gray-300 dark:text-gray-600">○</span>
        </div>
        <p className="text-sm font-semibold text-gray-500 dark:text-gray-400">{emptyTitle}</p>
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-1 max-w-[200px] leading-relaxed">{emptySubtitle}</p>
      </div>
    )
  }

  // ── Main render ──────────────────────────────────────────────────────────
  return (
    <div className={cn("flex flex-col gap-2", className)}>

      {/* ── All done banner ── */}
      {allDone && (
        <div className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60">
          <CheckCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
          <span className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">
            All {total} task{total !== 1 ? "s" : ""} completed
          </span>
        </div>
      )}

      {/* ── Progress bar (while running) ── */}
      {!allDone && total > 1 && (
        <div className="flex items-center gap-2.5 px-1">
          <div className="flex-1 h-1 rounded-full bg-gray-100 dark:bg-gray-800 overflow-hidden">
            <div
              className="h-full bg-emerald-500 rounded-full transition-all duration-700 ease-out"
              style={{ width: `${Math.round((doneCount / total) * 100)}%` }}
            />
          </div>
          <span className="text-[11px] text-gray-400 dark:text-gray-500 tabular-nums whitespace-nowrap">
            {doneCount} / {total}
          </span>
        </div>
      )}

      {/* ── Brain reasoning ── */}
      {brainReasoning && (
        <div className="flex items-start gap-2 px-2.5 py-2 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-100 dark:border-blue-900/50">
          <BrainCircuit className="w-3.5 h-3.5 text-blue-500 dark:text-blue-400 mt-0.5 flex-shrink-0 animate-pulse" />
          <p className="text-[11px] text-blue-700 dark:text-blue-300 leading-relaxed line-clamp-3">{brainReasoning}</p>
        </div>
      )}

      {/* ── Step list ── */}
      <ol className="flex flex-col gap-0.5">
        {tasks.map((task, index) => {
          const running = task.status === "in_progress"
          const done = task.status === "completed"
          const failed = task.status === "failed"
          const pending = !running && !done && !failed

          return (
            <li
              key={task.id}
              className={cn(
                "flex gap-3 px-3 py-2.5 rounded-lg transition-all duration-300",
                running && "bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800/60 shadow-sm",
                failed && "bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/50",
                done && "opacity-60",
                pending && "opacity-40"
              )}
            >
              {/* ── Status indicator ── */}
              <div className="flex-shrink-0 pt-0.5">
                {done ? (
                  <div className="w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-900/50 flex items-center justify-center">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                  </div>
                ) : running ? (
                  <div className="w-5 h-5 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center">
                    <Loader2 className="w-3 h-3 text-blue-600 dark:text-blue-400 animate-spin" />
                  </div>
                ) : failed ? (
                  <div className="w-5 h-5 rounded-full bg-red-100 dark:bg-red-900/50 flex items-center justify-center">
                    <AlertTriangle className="w-3 h-3 text-red-500 dark:text-red-400" />
                  </div>
                ) : (
                  <div className="w-5 h-5 rounded-full border border-gray-200 dark:border-gray-700 flex items-center justify-center">
                    <span className="text-[10px] text-gray-400 dark:text-gray-600 font-mono leading-none">{index + 1}</span>
                  </div>
                )}
              </div>

              {/* ── Task body ── */}
              <div className="flex-1 min-w-0">
                {/* Title + time */}
                <div className="flex items-start justify-between gap-2">
                  <p className={cn(
                    "text-[13px] leading-snug",
                    running && "font-semibold text-gray-900 dark:text-gray-100",
                    done && "font-medium text-gray-500 dark:text-gray-500",
                    failed && "font-medium text-red-600 dark:text-red-400",
                    pending && "font-medium text-gray-400 dark:text-gray-600",
                  )}>
                    {task.title}
                  </p>
                  {done && task.executionTime && (
                    <span className="flex-shrink-0 flex items-center gap-0.5 text-[10px] text-gray-400 dark:text-gray-600 mt-0.5 whitespace-nowrap">
                      <Clock className="w-2.5 h-2.5" />
                      {formatDuration(task.executionTime)}
                    </span>
                  )}
                </div>

                {/* Agent label (for pending/running, not for done) */}
                {!done && task.agentName && (
                  <p className={cn(
                    "mt-0.5 text-[11px] font-medium",
                    running ? "text-blue-500 dark:text-blue-400" : "text-gray-400 dark:text-gray-600"
                  )}>
                    via {task.agentName}
                  </p>
                )}

                {/* Running: live text */}
                {running && task.liveText && (
                  <div className="mt-0.5 flex items-start gap-1.5">
                    <span className="mt-[5px] w-1.5 h-1.5 rounded-full bg-blue-500 dark:bg-blue-400 animate-pulse flex-shrink-0" />
                    <p className="text-[11px] text-blue-600 dark:text-blue-300 leading-relaxed">{task.liveText}</p>
                  </div>
                )}

                {/* Done: result */}
                {done && task.result && (
                  <p className="mt-0.5 text-[11px] text-gray-400 dark:text-gray-500 line-clamp-2 leading-relaxed">
                    {task.result}
                  </p>
                )}

                {/* Failed: error */}
                {failed && task.error && (
                  <p className="mt-0.5 text-[11px] text-red-500 dark:text-red-400 line-clamp-2 leading-relaxed">
                    {task.error}
                  </p>
                )}
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
