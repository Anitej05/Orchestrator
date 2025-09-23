"use client"

import { useState, useEffect } from "react"
import AppSidebar from "@/components/app-sidebar"
import AgentGrid from "@/components/agent-grid"
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
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

export default function AgentsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("All")
  const [agents, setAgents] = useState<Agent[]>([]) // Initial state is an empty array
  const [loading, setLoading] = useState(true)

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
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <div className="min-h-screen bg-gray-50">
          {/* Simple Header */}
          <div className="bg-white border-b px-6 py-4">
            <div className="flex items-center space-x-4">
              <SidebarTrigger />
              <h1 className="text-2xl font-bold text-gray-900">Agents</h1>
            </div>
          </div>

          {/* Main Content */}
          <main className="p-6">
            {/* Search and Category Filter */}
            <div className="flex flex-col sm:flex-row gap-4 mb-6">
              {/* Search */}
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search agents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
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
            <div className="mb-4 text-sm text-gray-600">
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
    </SidebarProvider>
  )
}