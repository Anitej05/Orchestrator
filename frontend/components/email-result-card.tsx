"use client"

import { useState } from "react"
import { Mail, ChevronDown, ChevronUp, User, Clock, Paperclip } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export interface EmailMessage {
  id?: string
  subject?: string
  sender?: string
  from?: string
  to?: string
  date?: string
  snippet?: string
  body?: string
  labels?: string[]
  has_attachments?: boolean
  thread_id?: string
}

interface EmailResultCardProps {
  messages: EmailMessage[]
  totalCount?: number
  query?: string
  className?: string
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return ""
  try {
    const d = new Date(dateStr)
    if (isNaN(d.getTime())) return dateStr
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: d.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    })
  } catch {
    return dateStr
  }
}

function extractName(senderStr?: string): string {
  if (!senderStr) return "Unknown"
  // "Name <email@example.com>" → "Name"
  const match = senderStr.match(/^([^<]+)</)
  if (match) return match[1].trim()
  return senderStr
}

function EmailCard({ email, index }: { email: EmailMessage; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const sender = email.sender || email.from || ""
  const senderName = extractName(sender)
  const hasBody = !!(email.body && email.body.trim())
  const displayDate = formatDate(email.date)

  return (
    <div className="ui-card rounded-lg border border-border-color bg-bg-card overflow-hidden">
      <button
        className="w-full text-left px-4 py-3 hover:bg-bg-hover transition-colors flex items-start gap-3"
        onClick={() => hasBody && setExpanded(!expanded)}
        disabled={!hasBody}
      >
        {/* Avatar */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-teal/20 flex items-center justify-center mt-0.5">
          <User className="w-4 h-4 text-brand-teal" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium text-text-primary truncate">
              {senderName}
            </span>
            <div className="flex items-center gap-2 flex-shrink-0">
              {email.has_attachments && (
                <Paperclip className="w-3 h-3 text-text-tertiary" />
              )}
              {displayDate && (
                <span className="text-xs text-text-tertiary whitespace-nowrap flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {displayDate}
                </span>
              )}
            </div>
          </div>

          <p className="text-sm font-medium text-text-primary mt-0.5 truncate">
            {email.subject || "(no subject)"}
          </p>

          {!expanded && email.snippet && (
            <p className="text-xs text-text-tertiary mt-1 line-clamp-1">
              {email.snippet}
            </p>
          )}

          {email.labels && email.labels.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {email.labels.slice(0, 3).map((label) => (
                <Badge key={label} variant="ui-pending" className="text-[10px] px-1.5 py-0">
                  {label.replace("CATEGORY_", "").toLowerCase()}
                </Badge>
              ))}
            </div>
          )}
        </div>

        {/* Expand indicator */}
        {hasBody && (
          <div className="flex-shrink-0 text-text-tertiary mt-1">
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </div>
        )}
      </button>

      {/* Expanded body */}
      {expanded && hasBody && (
        <div className="px-4 pb-4 pt-1 border-t border-border-color bg-bg-subtle">
          <pre className="text-xs text-text-secondary whitespace-pre-wrap font-sans leading-relaxed max-h-64 overflow-y-auto">
            {email.body}
          </pre>
        </div>
      )}
    </div>
  )
}

export function EmailResultCard({ messages, totalCount, query, className }: EmailResultCardProps) {
  const shown = messages.slice(0, 20)
  const extra = (totalCount ?? messages.length) - shown.length

  return (
    <div className={cn("w-full space-y-2", className)}>
      {/* Header */}
      <div className="flex items-center gap-2 px-1">
        <Mail className="w-4 h-4 text-brand-teal" />
        <span className="text-sm font-medium text-text-secondary">
          {totalCount != null ? `${totalCount} email${totalCount !== 1 ? "s" : ""}` : `${messages.length} email${messages.length !== 1 ? "s" : ""}`}
          {query ? ` for "${query}"` : ""}
        </span>
      </div>

      {/* Email cards */}
      <div className="space-y-1.5">
        {shown.map((email, i) => (
          <EmailCard key={email.id || `email-${i}`} email={email} index={i} />
        ))}
      </div>

      {extra > 0 && (
        <p className="text-xs text-text-tertiary text-center py-1">
          +{extra} more email{extra !== 1 ? "s" : ""}
        </p>
      )}
    </div>
  )
}
