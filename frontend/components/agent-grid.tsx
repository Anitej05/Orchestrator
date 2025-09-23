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
  const [registeredAgents, setRegisteredAgents] = useState<Set<string>>(new Set())
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

  const handleToggleRegistration = (agentId: string) => {
    setRegisteredAgents((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(agentId)) {
        newSet.delete(agentId)
      } else {
        newSet.add(agentId)
      }
      return newSet
    })
  }

  if (filteredAgents.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">No agents found</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {filteredAgents.map((agent) => (
        <AgentCard
          key={agent.id}
          agent={agent}
          isRegistered={registeredAgents.has(agent.id)}
          onToggleRegistration={() => handleToggleRegistration(agent.id)}
        />
      ))}
    </div>
  )
}
