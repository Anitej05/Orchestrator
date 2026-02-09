"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useConversationStore } from "@/lib/conversation-store"
import { SidebarInset, SidebarTrigger, useSidebar } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Play, Trash2, Copy, Calendar, Clock, DollarSign, Zap } from "lucide-react"

interface Workflow {
  workflow_id: string
  workflow_name: string
  workflow_description: string
  created_at: string
  updated_at: string
  task_count: number
  estimated_cost: number
  is_public: boolean
}

function SavedWorkflowsContent() {
  const router = useRouter()
  const { open } = useSidebar()
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadWorkflows()
  }, [])

  const loadWorkflows = async () => {
    setLoading(true)
    setError(null)

    try {
      const { authFetch } = await import("@/lib/auth-fetch")
      const response = await authFetch("http://localhost:8000/api/workflows")

      if (!response.ok) {
        throw new Error("Failed to load workflows")
      }

      const data = await response.json()
      setWorkflows(data)
    } catch (err) {
      console.error("Failed to load workflows:", err)
      setError(err instanceof Error ? err.message : "Failed to load workflows")
    } finally {
      setLoading(false)
    }
  }

  const handleExecuteWorkflow = async (workflowId: string, e: React.MouseEvent) => {
    e.stopPropagation()

    try {
      const { authFetch } = await import("@/lib/auth-fetch")
      const { toast } = await import("sonner")

      toast.info("Loading workflow...")

      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}/create-conversation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error("Failed to create conversation:", errorText)
        throw new Error("Failed to load workflow")
      }

      const data = await response.json()
      const threadId = data.thread_id

      toast.success("Workflow loaded! Review the plan and click to execute.")

      const { loadConversation } = useConversationStore.getState().actions
      await loadConversation(threadId)

      window.history.replaceState({}, "", `/c/${threadId}`)
      router.push("/")
    } catch (err) {
      console.error("Failed to execute workflow:", err)
      const { toast } = await import("sonner")
      toast.error("Failed to load workflow")
    }
  }

  const handleDeleteWorkflow = async (workflowId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm("Are you sure you want to delete this workflow?")) return

    try {
      const { authFetch } = await import("@/lib/auth-fetch")
      const { toast } = await import("sonner")

      toast.info("Deleting workflow...")
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}`, {
        method: "DELETE"
      })

      if (!response.ok) {
        throw new Error("Failed to delete workflow")
      }

      toast.success("Workflow deleted successfully")
      loadWorkflows()
    } catch (err) {
      console.error("Failed to delete workflow:", err)
      const { toast } = await import("sonner")
      toast.error("Failed to delete workflow")
    }
  }

  const handleCloneWorkflow = async (workflowId: string, e: React.MouseEvent) => {
    e.stopPropagation()

    try {
      const { authFetch } = await import("@/lib/auth-fetch")
      const { toast } = await import("sonner")

      toast.info("Cloning workflow...")
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}/clone`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: "Copy of workflow" })
      })

      if (!response.ok) {
        throw new Error("Failed to clone workflow")
      }

      toast.success("Workflow cloned successfully")
      loadWorkflows()
    } catch (err) {
      console.error("Failed to clone workflow:", err)
      const { toast } = await import("sonner")
      toast.error("Failed to clone workflow")
    }
  }

  if (loading) {
    return (
      <SidebarInset>
        <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
          <main className="p-6">
            <div className="flex items-center justify-center h-64 text-text-secondary">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-3 border-border-color border-t-brand-teal mx-auto"></div>
                <p className="mt-4">Loading workflows...</p>
              </div>
            </div>
          </main>
        </div>
      </SidebarInset>
    )
  }

  if (error) {
    return (
      <SidebarInset>
        <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
          <main className="p-6">
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <p className="text-status-error">{error}</p>
                <Button onClick={loadWorkflows} className="mt-4">Retry</Button>
              </div>
            </div>
          </main>
        </div>
      </SidebarInset>
    )
  }

  return (
    <SidebarInset>
      <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
        <main className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-brand-teal">Saved Workflows</h1>
            <p className="text-text-secondary mt-2">Your reusable automation workflows. Execute, edit, or delete them anytime.</p>
          </div>

          {workflows.length === 0 ? (
            <Card className="ui-card">
              <CardContent className="flex flex-col items-center justify-center py-12 text-text-secondary">
                <Zap className="w-16 h-16 text-brand-teal mb-4" />
                <h3 className="text-xl font-semibold mb-2 text-text-primary">No workflows yet</h3>
                <p className="text-center mb-4">Start a conversation and click "Save as Workflow" to create your first reusable workflow.</p>
                <Button onClick={() => router.push("/")}>Start Conversation</Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {workflows.map((workflow) => (
                <Card
                  key={workflow.workflow_id}
                  className="ui-card flex flex-col hover:border-brand-teal hover:shadow-brand transition-all"
                >
                  <CardHeader
                    className="cursor-pointer"
                    onClick={() => router.push(`/saved-workflows/${workflow.workflow_id}`)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 space-y-1">
                        <CardTitle className="text-lg text-text-primary">{workflow.workflow_name || "Untitled Workflow"}</CardTitle>
                        <CardDescription className="line-clamp-2 text-text-secondary">
                          {workflow.workflow_description || "No description"}
                        </CardDescription>
                      </div>
                      {workflow.is_public && <Badge className="bg-status-active/15 text-status-active border border-status-active/30 ml-2">Public</Badge>}
                    </div>
                  </CardHeader>

                  <CardContent className="flex-1 flex flex-col justify-between space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center text-sm text-text-secondary">
                        <Clock className="w-4 h-4 mr-2 text-brand-teal" />
                        <span>{workflow.task_count} tasks</span>
                      </div>

                      {workflow.estimated_cost > 0 && (
                        <div className="flex items-center text-sm text-text-secondary">
                          <DollarSign className="w-4 h-4 mr-2 text-status-warning" />
                          <span>${workflow.estimated_cost.toFixed(4)}</span>
                        </div>
                      )}

                      <div className="flex items-center text-sm text-text-secondary">
                        <Calendar className="w-4 h-4 mr-2 text-text-secondary" />
                        <span>{new Date(workflow.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <Button size="sm" className="flex-1" onClick={(e) => handleExecuteWorkflow(workflow.workflow_id, e)}>
                        <Play className="w-4 h-4 mr-1" />
                        Run
                      </Button>

                      <Button size="sm" variant="outline" onClick={(e) => handleCloneWorkflow(workflow.workflow_id, e)}>
                        <Copy className="w-4 h-4" />
                      </Button>

                      <Button size="sm" variant="destructive" onClick={(e) => handleDeleteWorkflow(workflow.workflow_id, e)}>
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </main>
      </div>
    </SidebarInset>
  )
}

export default function SavedWorkflowsPage() {
  return (
    <>
      <SavedWorkflowsContent />
    </>
  )
}

