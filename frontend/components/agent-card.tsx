"use client"

import { Badge } from "@/components/ui/badge"
import type { Agent } from "@/lib/types"

interface AgentCardProps {
  agent: Agent
}

export default function AgentCard({ agent }: AgentCardProps) {
  const descriptionText = agent.description
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/_(.*?)_/g, "$1")
    .replace(/`([^`]+)`/g, "$1")

  return (
    <div className="ui-card-hover p-6">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="ui-task-name">{agent.name}</h3>
          <div className="mt-1">
            <Badge variant={agent.status === "active" ? "ui-active" : "ui-pending"}>
              {agent.status}
            </Badge>
          </div>
        </div>
      </div>

      {/* Description */}
      <p className="ui-task-description mb-4 line-clamp-2">{descriptionText}</p>

      {/* Capabilities */}
      <div className="flex flex-wrap gap-1 mb-4">
        {agent.capabilities.slice(0, 3).map((cap, index) => (
          <Badge key={`${agent.id}-${cap}-${index}`} variant="ui-pending" className="ui-file-meta">
            {cap.replace(/_/g, " ")}
          </Badge>
        ))}
        {agent.capabilities.length > 3 && (
          <Badge variant="outline" className="ui-file-meta">
            +{agent.capabilities.length - 3}
          </Badge>
        )}
      </div>

    </div>
  )
}

