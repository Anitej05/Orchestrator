"use client"

/**
 * TaskPlanViewer — Manus-style live task plan tracker
 *
 * Self-contained: reads directly from the Zustand conversation store.
 * Drop anywhere with <TaskPlanViewer />.
 *
 * Visual states:
 *   ○  pending   — dim numbered circle, muted text
 *   ◉  running   — blue row tint + left accent border, spinner, pulsing live text
 *   ✓  done      — green check, dimmed + struck text, exec-time badge, result snippet
 *   ✕  failed    — red tint, red X, error message
 */

import { useEffect, useRef, useMemo } from "react"
import {
  CheckCircle2,
  AlertTriangle,
  Loader2,
  CheckCheck,
  Clock,
  BrainCircuit,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useConversationStore } from "@/lib/conversation-store"
import type { TaskStatus, TodoItem } from "@/lib/types"

// ── Types ─────────────────────────────────────────────────────────────────────

interface NormalizedTask {
  id: string
  title: string
  status: "pending" | "in_progress" | "completed" | "failed"
  agentName?: string
  liveText?: string
  resultSummary?: string
  executionTime?: number
  error?: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function normalizeStatus(raw?: string): NormalizedTask["status"] {
  const v = (raw || "pending").toLowerCase()
  if (v === "running" || v === "in_progress" || v === "in-progress") return "in_progress"
  if (v.includes("complet") || v === "skipped") return "completed"
  if (v.includes("fail")) return "failed"
  return "pending"
}

function formatDuration(ms?: number): string | undefined {
  if (typeof ms !== "number" || isNaN(ms) || ms <= 0) return undefined
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`
}

function formatAgentName(id?: string): string | undefined {
  if (!id) return undefined
  return id
    .replace(/_agent$/i, "")
    .replace(/_/g, " ")
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ")
}

function normalizeTasks(
  todoList: TodoItem[],
  taskStatuses: Record<string, TaskStatus>
): NormalizedTask[] {
  return todoList.map((task, index) => {
    const taskKey = String(task.task_id || task.id || `task-${index}`)
    const ts = taskStatuses[taskKey]

    const rawStatus = ts?.status || task.status || "pending"
    const status = normalizeStatus(rawStatus)

    const agentRaw =
      ts?.agentName ||
      (task.payload as any)?.agent_name ||
      (task as any).assigned_agent ||
      (task as any).agent_name
    const agentName = formatAgentName(agentRaw)

    const liveRaw = ts?.activityDescription || ""
    const liveText = liveRaw.length > 130 ? liveRaw.slice(0, 130) + "…" : liveRaw || undefined

    return {
      id: taskKey,
      title: String(task.description || ts?.taskName || `Task ${index + 1}`),
      status,
      agentName,
      liveText: status === "in_progress" ? liveText : undefined,
      resultSummary: status === "completed" ? ts?.resultSummary || undefined : undefined,
      executionTime: ts?.executionTime,
      error: status === "failed" ? (ts?.error || (task as any).error || undefined) : undefined,
    }
  })
}

// ── Empty state ────────────────────────────────────────────────────────────────

function EmptyState({ brainReasoning }: { brainReasoning?: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full py-12 px-6 text-center gap-4">
      {brainReasoning ? (
        <div className="w-full max-w-xs">
          <div className="flex items-center justify-center gap-2 mb-3">
            <BrainCircuit className="w-4 h-4 text-brand-primary animate-pulse" />
            <span className="text-xs font-semibold text-brand-primary uppercase tracking-wide">
              Planning
            </span>
          </div>
          <p className="text-[12px] text-text-tertiary leading-relaxed line-clamp-4">
            {brainReasoning}
          </p>
        </div>
      ) : (
        <>
          <div className="w-10 h-10 rounded-full bg-brand-primary-light border border-brand-primary/20 flex items-center justify-center">
            <span className="text-lg text-brand-primary/40 select-none">○</span>
          </div>
          <div>
            <p className="text-sm font-semibold text-text-secondary">No tasks yet</p>
            <p className="text-xs text-text-tertiary mt-1 leading-relaxed max-w-[180px]">
              Tasks will appear here as the agent plans and executes
            </p>
          </div>
        </>
      )}
    </div>
  )
}

// ── Status icon ────────────────────────────────────────────────────────────────

function StatusIcon({ status, index }: { status: NormalizedTask["status"]; index: number }) {
  if (status === "completed") {
    return (
      <div className="w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-900/50 flex items-center justify-center flex-shrink-0">
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
      </div>
    )
  }
  if (status === "in_progress") {
    return (
      <div className="w-5 h-5 rounded-full bg-brand-primary/15 flex items-center justify-center flex-shrink-0">
        <Loader2 className="w-3 h-3 text-brand-primary animate-spin" />
      </div>
    )
  }
  if (status === "failed") {
    return (
      <div className="w-5 h-5 rounded-full bg-red-100 dark:bg-red-900/50 flex items-center justify-center flex-shrink-0">
        <AlertTriangle className="w-3 h-3 text-red-500 dark:text-red-400" />
      </div>
    )
  }
  // pending
  return (
    <div className="w-5 h-5 rounded-full border border-gray-200 dark:border-gray-700 flex items-center justify-center flex-shrink-0">
      <span className="text-[10px] text-gray-400 dark:text-gray-600 font-mono leading-none select-none">
        {index + 1}
      </span>
    </div>
  )
}

// ── Task row ──────────────────────────────────────────────────────────────────

function TaskRow({
  task,
  index,
  isActive,
  rowRef,
}: {
  task: NormalizedTask
  index: number
  isActive: boolean
  rowRef?: React.RefObject<HTMLLIElement>
}) {
  const running = task.status === "in_progress"
  const done = task.status === "completed"
  const failed = task.status === "failed"
  const pending = task.status === "pending"

  return (
    <li
      ref={rowRef as any}
      className={cn(
        "flex gap-3 px-3 py-2.5 rounded-lg transition-all duration-300 relative",
        running && [
          "bg-brand-primary/[0.06] dark:bg-brand-primary/[0.12]",
          "border border-brand-primary/25",
          "shadow-sm",
          // Left accent bar
          "before:absolute before:left-0 before:top-2 before:bottom-2 before:w-0.5",
          "before:rounded-full before:bg-brand-primary",
        ],
        failed && "bg-red-50/60 dark:bg-red-950/20 border border-red-200 dark:border-red-900/40",
        done && "opacity-55",
        pending && "opacity-50",
      )}
    >
      {/* Status icon */}
      <div className="pt-0.5">
        <StatusIcon status={task.status} index={index} />
      </div>

      {/* Body */}
      <div className="flex-1 min-w-0">
        {/* Title row */}
        <div className="flex items-start justify-between gap-2">
          <p
            className={cn(
              "text-[13px] leading-snug break-words",
              running && "font-semibold text-text-primary",
              done && "font-medium text-gray-400 dark:text-gray-500 line-through decoration-gray-300 dark:decoration-gray-600",
              failed && "font-medium text-red-600 dark:text-red-400",
              pending && "font-medium text-gray-400 dark:text-gray-600",
            )}
          >
            {task.title}
          </p>

          {/* Exec time badge on done */}
          {done && task.executionTime && (
            <span className="flex-shrink-0 flex items-center gap-0.5 text-[10px] text-gray-400 dark:text-gray-600 mt-0.5 whitespace-nowrap tabular-nums">
              <Clock className="w-2.5 h-2.5" />
              {formatDuration(task.executionTime)}
            </span>
          )}
        </div>

        {/* Agent label */}
        {task.agentName && !done && (
          <p
            className={cn(
              "mt-0.5 text-[11px] font-medium",
              running ? "text-brand-primary" : "text-text-disabled",
            )}
          >
            via {task.agentName}
          </p>
        )}

        {/* Live text (running only) */}
        {running && task.liveText && (
          <div className="mt-1 flex items-start gap-1.5">
            <span className="mt-[5px] w-1.5 h-1.5 rounded-full bg-brand-primary animate-pulse flex-shrink-0" />
            <p className="text-[11px] text-brand-primary leading-relaxed">
              {task.liveText}
            </p>
          </div>
        )}

        {/* Result snippet (done) */}
        {done && task.resultSummary && (
          <p className="mt-0.5 text-[11px] text-gray-400 dark:text-gray-500 line-clamp-2 leading-relaxed">
            {task.resultSummary}
          </p>
        )}

        {/* Error (failed) */}
        {failed && task.error && (
          <p className="mt-0.5 text-[11px] text-red-500 dark:text-red-400 line-clamp-2 leading-relaxed">
            {task.error}
          </p>
        )}
      </div>
    </li>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export function TaskPlanViewer() {
  const conversationState = useConversationStore()
  const todoList = conversationState.todo_list || []
  const taskStatuses = conversationState.task_statuses || {}
  const brainReasoning = conversationState.brain_reasoning
  const currentExecutingTask = conversationState.current_executing_task
  const convStatus = conversationState.status

  const tasks = useMemo(
    () => normalizeTasks(todoList, taskStatuses),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(todoList), JSON.stringify(taskStatuses)],
  )

  const doneCount = tasks.filter((t) => t.status === "completed").length
  const total = tasks.length
  const allDone = total > 0 && doneCount === total
  const progressPct = total > 0 ? Math.round((doneCount / total) * 100) : 0

  // Auto-scroll active task into view
  const activeRef = useRef<HTMLLIElement>(null)
  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })
  }, [currentExecutingTask])

  const isProcessing =
    convStatus === "processing" ||
    conversationState.metadata?.currentStage === "executing" ||
    conversationState.metadata?.currentStage === "planning"

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-color/40 flex-shrink-0">
        <div className="flex items-center gap-2">
          {allDone ? (
            <CheckCheck className="w-4 h-4 text-emerald-500 flex-shrink-0" />
          ) : isProcessing ? (
            <Loader2 className="w-4 h-4 text-blue-500 animate-spin flex-shrink-0" />
          ) : null}
          <div>
            <h3 className="text-sm font-semibold text-text-primary leading-none">
              Tasks
            </h3>
            <p className="text-[11px] text-text-tertiary mt-0.5">
              {allDone
                ? "All done"
                : isProcessing && currentExecutingTask
                ? "Running…"
                : isProcessing
                ? "Planning…"
                : convStatus === "completed"
                ? "Completed"
                : "Waiting"}
            </p>
          </div>
        </div>

        {total > 0 && (
          <span className="text-[11px] px-1.5 py-0.5 bg-brand-primary-light text-brand-primary rounded-full border border-brand-primary/20 tabular-nums flex-shrink-0">
            {doneCount}&thinsp;/&thinsp;{total}
          </span>
        )}
      </div>

      {/* ── Progress bar ── */}
      {total > 0 && (
        <div className="h-[2px] bg-gray-100 dark:bg-gray-800 flex-shrink-0">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-700 ease-out",
              allDone ? "bg-status-success" : "bg-brand-primary",
            )}
            style={{ width: `${progressPct}%` }}
          />
        </div>
      )}

      {/* ── Content ── */}
      {tasks.length === 0 ? (
        <EmptyState brainReasoning={brainReasoning} />
      ) : (
        <div className="flex-1 overflow-y-auto min-h-0">
          {/* Brain reasoning shown above list when actively thinking between tasks */}
          {brainReasoning && !currentExecutingTask && (
            <div className="mx-3 mt-3 flex items-start gap-2 px-2.5 py-2 rounded-lg bg-brand-primary-light border border-brand-primary/20">
              <BrainCircuit className="w-3.5 h-3.5 text-brand-primary mt-0.5 flex-shrink-0 animate-pulse" />
              <p className="text-[11px] text-text-secondary leading-relaxed line-clamp-3">
                {brainReasoning}
              </p>
            </div>
          )}

          <ol className="flex flex-col gap-0.5 p-3">
            {tasks.map((task, index) => {
              const isActive = task.status === "in_progress"
              return (
                <TaskRow
                  key={task.id}
                  task={task}
                  index={index}
                  isActive={isActive}
                  rowRef={isActive ? activeRef : undefined}
                />
              )
            })}
          </ol>

          {/* All done banner */}
          {allDone && (
            <div className="mx-3 mb-3 flex items-center gap-2 px-3 py-2.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60">
              <CheckCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
              <span className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">
                All {total} task{total !== 1 ? "s" : ""} completed
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default TaskPlanViewer
