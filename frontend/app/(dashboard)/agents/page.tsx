"use client"

import { useState, useEffect } from "react"
import AgentGrid from "@/components/agent-grid"
import { SidebarInset, useSidebar } from "@/components/ui/sidebar"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Search } from "lucide-react"
import { fetchAllAgents } from "@/lib/api-client"
import type { Agent } from "@/lib/types"

// Define categories based on agent capabilities
const categories = [
  "All",
  "Business & Sales",
  "Technical & Development",
  "Customer & Support",
  "Financial & Operations",
  "Content & Media",
]

function AgentsContent() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("All")
  const [agents, setAgents] = useState<Agent[]>([]) // Initial state is an empty array
  const [loading, setLoading] = useState(true)
  const { open } = useSidebar()

  // Fetch agents from backend on component mount
  useEffect(() => {
    const loadAgents = async () => {
      setLoading(true)
      try {
        const data = await fetchAllAgents()
        setAgents(data) // Directly set the agents from the API response
      } catch (err) {
        console.error('Failed to fetch agents from backend:', err)
        setAgents([]) // On error, ensure the agent list is empty
      } finally {
        setLoading(false)
      }
    }

    loadAgents()
  }, [])

  // Filter agents by category
  const getAgentsByCategory = (category: string) => {
    if (category === "All") return agents

    return agents.filter((agent) => {
      const caps = agent.capabilities.map((c) => c.toLowerCase())

      switch (category) {
        case "Business & Sales":
          return caps.some(
            (cap) =>
              cap.includes("lead") ||
              cap.includes("sales") ||
              cap.includes("marketing") ||
              cap.includes("email") ||
              cap.includes("market"),
          )
        case "Technical & Development":
          return caps.some(
            (cap) =>
              cap.includes("code") ||
              cap.includes("development") ||
              cap.includes("technical") ||
              cap.includes("autogen") ||
              cap.includes("langchain") ||
              cap.includes("crewai"),
          )
        case "Customer & Support":
          return caps.some(
            (cap) =>
              cap.includes("support") ||
              cap.includes("customer") ||
              cap.includes("translation") ||
              cap.includes("social") ||
              cap.includes("scheduling"),
          )
        case "Financial & Operations":
          return caps.some(
            (cap) =>
              cap.includes("financial") || cap.includes("payment") || cap.includes("analysis") || cap.includes("data"),
          )
        case "Content & Media":
          return caps.some(
            (cap) =>
              cap.includes("content") ||
              cap.includes("document") ||
              cap.includes("translation") ||
              cap.includes("social") ||
              cap.includes("media"),
          )
        default:
          return true
      }
    })
  }

  const filteredAgents = getAgentsByCategory(selectedCategory)

  return (
    <SidebarInset>
      <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
        {/* Main Content */}
        <main className="p-6">
          {/* Title Section */}
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-brand-teal">Agent Directory</h1>
            <p className="text-text-secondary mt-2">
              Browse and discover AI agents for your orchestration workflows.
            </p>
          </div>

          {/* Search and Category Filter */}
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary w-4 h-4" />
              <Input
                placeholder="Search agents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 ui-input"
              />
            </div>

            {/* Category Dropdown */}
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="w-full sm:w-[200px]">
                <SelectValue placeholder="Select category" />
              </SelectTrigger>
              <SelectContent>
                {categories.map((category) => (
                  <SelectItem key={category} value={category}>
                    {category}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Results Count */}
          <div className="mb-4 text-sm text-text-secondary">
            {loading ? "Loading agents..." : (
              <>
                {filteredAgents.length} agent{filteredAgents.length !== 1 ? "s" : ""}
                {selectedCategory !== "All" && ` in ${selectedCategory}`}
              </>
            )}
          </div>

          {/* Agent Grid */}
          <AgentGrid agents={filteredAgents} searchQuery={searchQuery} selectedCapability="All" />
        </main>
      </div>
    </SidebarInset>
  )
}

export default function AgentsPage() {
  return <AgentsContent />
}
