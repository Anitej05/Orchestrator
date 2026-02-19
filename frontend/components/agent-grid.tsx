"use client"

import { useState, useEffect } from "react"
import AgentCard from "./agent-card"
import type { Agent } from "@/lib/types"

interface AgentGridProps {
  agents: Agent[]
  searchQuery?: string
  selectedCapability?: string
}

export default function AgentGrid({ agents, searchQuery = "" }: AgentGridProps) {
  const [filteredAgents, setFilteredAgents] = useState<Agent[]>(agents)

  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredAgents(agents)
      return
    }

    const query = searchQuery.toLowerCase()
    const filtered = agents.filter(
      (agent) =>
        agent.name.toLowerCase().includes(query) ||
        agent.description.toLowerCase().includes(query) ||
        agent.capabilities.some((cap) => cap.toLowerCase().includes(query)),
    )
    setFilteredAgents(filtered)
  }, [agents, searchQuery])

  if (filteredAgents.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="ui-section-subtitle">No agents found</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {filteredAgents.map((agent) => (
        <AgentCard
          key={agent.id}
          agent={agent}
        />
      ))}
    </div>
  )
}

