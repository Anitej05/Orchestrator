'use client'

import { useConversationStore } from '@/lib/conversation-store'
import { cn } from '@/lib/utils'
import type { FlowEvent } from '@/lib/types'
import { Clock } from 'lucide-react'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtAgentName(id?: string): string {
  if (!id) return 'Agent'
  return id
    .replace(/_agent$/i, '')
    .replace(/_/g, ' ')
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ') + ' Agent'
}

function formatDuration(ms?: number): string | undefined {
  if (!ms || isNaN(ms) || ms <= 0) return undefined
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`
}

// Brain decision → plain English sentence
const BRAIN_LABELS: Record<string, string> = {
  plan:     'Planning your tasks',
  replan:   'Adjusting the plan',
  agent:    'Selecting the right agent',
  tool:     'Selecting a tool',
  python:   'Writing code to process the data',
  terminal: 'Running a command',
  finish:   'Preparing your response',
  parallel: 'Running tasks in parallel',
  skip:     'Already have what\'s needed',
}

// ── Dots ─────────────────────────────────────────────────────────────────────

function GreenDot() {
  return <div className="mt-[6px] w-3 h-3 rounded-full bg-emerald-500 flex-shrink-0" />
}

function AmberDot() {
  return (
    <div className="relative mt-[6px] flex-shrink-0 w-3 h-3">
      <div className="absolute inset-0 rounded-full bg-amber-400/50 animate-ping" />
      <div className="relative w-3 h-3 rounded-full bg-amber-400" />
    </div>
  )
}

function Connector({ done }: { done?: boolean }) {
  return (
    <div className={cn(
      'w-px flex-1 min-h-[20px] my-1 mx-auto transition-colors duration-500',
      done ? 'bg-emerald-400/40' : 'bg-border-color/25'
    )} />
  )
}

function BouncingDots() {
  return (
    <span className="inline-flex items-center gap-[3px]">
      <span className="w-[3px] h-[3px] rounded-full bg-amber-400 animate-bounce [animation-delay:0ms]" />
      <span className="w-[3px] h-[3px] rounded-full bg-amber-400 animate-bounce [animation-delay:130ms]" />
      <span className="w-[3px] h-[3px] rounded-full bg-amber-400 animate-bounce [animation-delay:260ms]" />
    </span>
  )
}

// ── Brain step — compact, muted ───────────────────────────────────────────────

function BrainRow({
  event,
  isLast,
  nextAgentName,
}: {
  event: FlowEvent
  isLast: boolean
  nextAgentName?: string
}) {
  // For the 'agent' decision, say which agent was selected (from the next event)
  const label = event.action_type === 'agent' && nextAgentName
    ? `Selected ${fmtAgentName(nextAgentName)}`
    : BRAIN_LABELS[event.action_type || ''] || 'Thinking'

  return (
    <div className="flex gap-4 items-start">
      <div className="flex flex-col items-center flex-shrink-0 w-3">
        <GreenDot />
        {!isLast && <Connector done />}
      </div>
      <div className={cn('flex-1 min-w-0', isLast ? 'pb-2' : 'pb-3')}>
        <p className="text-[13px] text-text-tertiary leading-snug">{label}</p>
      </div>
    </div>
  )
}

// ── Agent step — prominent, shows name + what it did ─────────────────────────

function AgentRow({ event, isLast }: { event: FlowEvent; isLast: boolean }) {
  const agentName = fmtAgentName(event.agentName)
  const running   = event.status === 'running'
  const done      = event.status === 'completed'
  const failed    = event.status === 'failed'
  const duration  = formatDuration(event.executionTime)

  // Clean up the result sublabel — strip any JSON-like prefixes
  const cleanResult = (() => {
    const raw = event.sublabel || ''
    if (!raw) return ''
    const m = raw.match(/["']task_summary["']\s*:\s*["']([^"']+)["']/i)
    if (m) return m[1]
    return raw.replace(/^result:\s*/i, '').replace(/^{[^:]+:\s*["']?/i, '').replace(/["']?}$/i, '').trim()
  })()

  return (
    <div className="flex gap-4 items-start">
      <div className="flex flex-col items-center flex-shrink-0 w-3">
        {running ? <AmberDot /> : <GreenDot />}
        {!isLast && <Connector done={done} />}
      </div>

      <div className={cn('flex-1 min-w-0', isLast ? 'pb-2' : 'pb-6')}>
        {/* Agent name + status */}
        <div className="flex items-start justify-between gap-2">
          <span className="text-[15px] font-semibold text-text-primary leading-snug">
            {agentName}
          </span>
          <div className="flex items-center gap-2 flex-shrink-0 mt-0.5">
            {running && <BouncingDots />}
            {done && duration && (
              <span className="flex items-center gap-1 text-[11px] text-text-tertiary tabular-nums">
                <Clock className="w-3 h-3" />{duration}
              </span>
            )}
          </div>
        </div>

        {/* What it was doing — task description */}
        {event.label && (
          <p className="mt-0.5 text-[13px] text-text-secondary leading-relaxed line-clamp-2">
            {event.label}
          </p>
        )}

        {/* Live progress while running */}
        {running && event.sublabel && (
          <div className="mt-2 pl-3 border-l-2 border-amber-400">
            <p className="text-[12px] text-amber-600 dark:text-amber-400 leading-relaxed">
              {event.sublabel}
            </p>
          </div>
        )}

        {/* Result when done — clean, no raw JSON, no errors shown */}
        {done && cleanResult && (
          <div className="mt-2 pl-3 border-l-2 border-emerald-400/60">
            <p className="text-[12px] text-text-tertiary leading-relaxed line-clamp-3">
              {cleanResult}
            </p>
          </div>
        )}

        {/* Failed — human-friendly only, no raw error */}
        {failed && (
          <div className="mt-2 pl-3 border-l-2 border-border-color/40">
            <p className="text-[12px] text-text-tertiary leading-relaxed">
              Could not complete this step — the orchestrator will retry or adjust
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Live "brain thinking" indicator ──────────────────────────────────────────

function ThinkingRow({ label }: { label: string }) {
  return (
    <div className="flex gap-4 items-start">
      <div className="flex flex-col items-center flex-shrink-0 w-3">
        <AmberDot />
      </div>
      <div className="flex-1 min-w-0 pb-2 flex items-center gap-2">
        <p className="text-[14px] text-text-secondary">{label}</p>
        <BouncingDots />
      </div>
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full py-16 px-6 text-center gap-3">
      <div className="w-10 h-10 rounded-full border-2 border-dashed border-border-color/30 flex items-center justify-center">
        <div className="w-2 h-2 rounded-full bg-border-color/30" />
      </div>
      <p className="text-sm font-semibold text-text-secondary">No activity yet</p>
      <p className="text-xs text-text-tertiary leading-relaxed max-w-[180px]">
        Steps the AI takes will appear here as it works
      </p>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function OrchestrationFlow() {
  const flowEvents  = useConversationStore((s) => s.flow_events as FlowEvent[] | undefined) || []
  const convStatus  = useConversationStore((s) => s.status)
  const isLoading   = useConversationStore((s) => (s as any).isLoading as boolean)

  const isProcessing = isLoading || convStatus === 'processing'
  const isCompleted  = convStatus === 'completed'

  if (!isProcessing && !isCompleted && flowEvents.length === 0) {
    return <EmptyState />
  }

  return (
    <div className="h-full overflow-y-auto px-5 py-5">

      {/* Nothing received yet */}
      {flowEvents.length === 0 && isProcessing && (
        <ThinkingRow label="Analysing your request" />
      )}

      {/* All captured steps */}
      {flowEvents.map((event, idx) => {
        const isLast = idx === flowEvents.length - 1

        if (event.type === 'brain') {
          // Look at the next event to name the agent when the brain just selected one
          const nextAgent = event.action_type === 'agent'
            ? flowEvents[idx + 1]?.agentName
            : undefined
          return (
            <BrainRow
              key={`${event.id}-${idx}`}
              event={event}
              isLast={isLast && !isProcessing}
              nextAgentName={nextAgent}
            />
          )
        }

        return (
          <AgentRow
            key={`${event.id}-${idx}`}
            event={event}
            isLast={isLast && !isProcessing}
          />
        )
      })}

      {/* Live indicator between events */}
      {isProcessing && flowEvents.length > 0 && (() => {
        const last = flowEvents[flowEvents.length - 1]
        let label = 'Working'
        if (last?.type === 'agent_call' && last.status === 'running')        label = 'Waiting for agent'
        else if (last?.type === 'agent_call' && last.status === 'completed') label = 'Reviewing result'
        else if (last?.type === 'brain')                                      label = 'Executing next step'
        return <ThinkingRow label={label} />
      })()}

      {/* Done */}
      {isCompleted && flowEvents.length > 0 && (
        <div className="flex items-center gap-3 mt-1">
          <div className="w-3 h-3 rounded-full bg-emerald-500 flex-shrink-0" />
          <p className="text-[14px] font-semibold text-emerald-600 dark:text-emerald-400">All done</p>
        </div>
      )}

    </div>
  )
}
