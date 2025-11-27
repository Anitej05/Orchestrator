// app/[thread_id]/page.tsx
"use client"

import { useParams, useRouter } from "next/navigation"
import { useEffect } from "react"
import { useConversationStore } from "@/lib/conversation-store"
import { SidebarProvider } from "@/components/ui/sidebar"
import Navbar from "@/components/navbar"
import AppSidebar from "@/components/app-sidebar"
import TaskBuilder from "@/components/task-builder"
import OrchestrationDetailsSidebar, { type OrchestrationDetailsSidebarRef } from "@/components/orchestration-details-sidebar"
import { SidebarInset, useSidebar } from "@/components/ui/sidebar"
import { type TaskAgentPair, type ProcessResponse } from "@/lib/types"
import { useToast } from "@/hooks/use-toast"
import { useWebSocketManager } from "@/hooks/use-websocket-conversation"
import { useUser } from "@clerk/nextjs"
import { useState, useRef } from "react"
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable"

interface ExecutionResult {
  taskId: string
  taskDescription: string
  agentName: string
  status: string
  output: string
  cost: number
  executionTime: number
}

interface ApiResponse {
  final_response: string | null
  message: string
  task_agent_pairs: TaskAgentPair[]
  thread_id: string
  pending_user_input: boolean
  question_for_user: string | null
}

function ConversationContent() {
  const { open } = useSidebar()
  const params = useParams()
  const router = useRouter()
  const threadId = params.thread_id as string
  const { user, isLoaded: clerkLoaded } = useUser()
  const [taskAgentPairs, setTaskAgentPairs] = useState<TaskAgentPair[]>([])
  const [selectedAgents, setSelectedAgents] = useState<Record<string, string>>({})
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionResults, setExecutionResults] = useState<ExecutionResult[]>([])
  const [apiResponseData, setApiResponseData] = useState<ApiResponse | null>(null)
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null)
  const [isRestoring, setIsRestoring] = useState(false)
  const { toast } = useToast()
  const sidebarRef = useRef<OrchestrationDetailsSidebarRef>(null)

  const conversationState = useConversationStore()
  const { resetConversation, loadConversation } = useConversationStore(state => state.actions)
  const isConversationLoading = useConversationStore((state: any) => state.isLoading)

  useWebSocketManager()

  // Load conversation from URL parameter
  useEffect(() => {
    if (!clerkLoaded || !threadId) return
    
    // If the store already has this conversation loaded, don't reload
    if (conversationState.thread_id === threadId) return
    
    console.log('Loading conversation from URL:', threadId)
    setIsRestoring(true)
    
    loadConversation(threadId)
      .catch((error: any) => {
        console.error('Failed to load conversation:', error)
        
        if (error?.message?.includes('403') || error?.message?.includes('permission')) {
          toast({
            title: "Access Denied",
            description: "You don't have permission to view this conversation",
            variant: "destructive",
          })
        } else if (error?.message?.includes('404')) {
          toast({
            title: "Not Found",
            description: "This conversation doesn't exist",
            variant: "destructive",
          })
        } else {
          toast({
            title: "Error",
            description: "Failed to load conversation",
            variant: "destructive",
          })
        }
        
        // Redirect to home on error
        router.push('/')
      })
      .finally(() => {
        setIsRestoring(false)
      })
  }, [threadId, clerkLoaded, conversationState.thread_id, loadConversation, router, toast])

  useEffect(() => {
    if (isRestoring) return
    if (conversationState.status === 'completed' && conversationState.final_response) {
      const result: ProcessResponse = {
        thread_id: conversationState.thread_id || '',
        message: "Completed",
        task_agent_pairs: conversationState.task_agent_pairs || [],
        final_response: conversationState.final_response,
        pending_user_input: false,
        question_for_user: null,
      }
      setApiResponseData(result as ApiResponse)
      setTaskAgentPairs(result.task_agent_pairs)
    } else if (conversationState.status === 'error') {
      const lastMessage = conversationState.messages[conversationState.messages.length - 1]
      const errorMessage = lastMessage?.content || "An unknown error occurred."
      toast({
        title: "Orchestration Error",
        description: errorMessage,
        variant: "destructive",
      })
    }
  }, [conversationState.status, conversationState.final_response, conversationState.thread_id, isRestoring, toast])

  const handleInteractiveWorkflowComplete = (result: ProcessResponse) => {
    setApiResponseData(result as ApiResponse)
    setTaskAgentPairs(result.task_agent_pairs)

    const initialSelections: Record<string, string> = {}
    result.task_agent_pairs.forEach((pair: TaskAgentPair) => {
      initialSelections[pair.task_name] = pair.primary.id
    })
    setSelectedAgents(initialSelections)

    if (result.task_agent_pairs.length > 0 && !result.pending_user_input) {
      setIsExecuting(true)
      toast({
        title: "Starting workflow execution",
        description: `Executing ${result.task_agent_pairs.length} tasks.`,
      })
    }
  }

  const handleViewCanvas = (canvasContent: string, canvasType: 'html' | 'markdown') => {
    sidebarRef.current?.viewCanvas(canvasContent, canvasType)
  }

  const handleConversationSelect = async (newThreadId: string) => {
    // Navigate to the new conversation URL
    router.push(`/${newThreadId}`)
  }

  const handleNewConversation = () => {
    // Clear localStorage and reset conversation state
    if (typeof window !== 'undefined') {
      localStorage.removeItem('thread_id')
    }
    resetConversation()
    
    // Navigate to home, which will start fresh
    router.push('/')
  }

  const handleOrchestrationComplete = (results: ExecutionResult[]) => {
    setExecutionResults(results)
    setIsExecuting(false)

    const totalCost = results.reduce((sum, result) => sum + result.cost, 0)

    toast({
      title: "Workflow executed successfully",
      description: `All ${results.length} tasks completed. Total cost: ${totalCost.toFixed(4)}`,
    })
  }

  const handleThreadIdUpdate = (newThreadId: string) => {
    setCurrentThreadId(newThreadId)
    
    // Update URL if thread ID changes (e.g., new conversation created)
    if (newThreadId && newThreadId !== threadId) {
      router.push(`/${newThreadId}`)
    }
  }

  const handleExecutionResultsUpdate = (results: ExecutionResult[]) => {
    setExecutionResults(results)
  }

  const handleAcceptPlan = async (modifiedPrompt?: string) => {
    try {
      if (modifiedPrompt && modifiedPrompt !== conversationState.original_prompt) {
        console.log('Modified prompt execution not yet implemented')
        toast({
          title: "Info",
          description: "Modified prompt execution will be available soon. Using original plan for now.",
        })
      }
      
      toast({
        title: "Executing Workflow",
        description: "Starting workflow execution...",
      })
      
      // For saved workflows with pre-seeded plans, send approval via WebSocket
      // This will skip re-planning and go straight to execution
      if (conversationState.status === 'planning_complete' && conversationState.thread_id) {
        const { continueConversation } = useConversationStore.getState().actions
        // Send "approve" as user_response to trigger execution of pre-seeded plan
        await continueConversation("approve", [], false, user?.id)
      }
      
    } catch (error) {
      console.error('Error accepting plan:', error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process plan. Please try again.",
        variant: "destructive"
      })
    }
  }

  const handleRejectPlan = () => {
    resetConversation()
    router.push('/')
    toast({
      title: "Workflow cancelled",
      description: "Workflow execution was cancelled. You can start a new conversation."
    })
  }

  return (
    <>
      <AppSidebar
        onConversationSelect={handleConversationSelect}
        onNewConversation={handleNewConversation}
        currentThreadId={conversationState.thread_id || undefined}
      />
      
      <SidebarInset className={!open ? "ml-16" : ""}>
        <div className="h-screen pt-[64px] bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-950 dark:to-gray-900 relative flex flex-col transition-all duration-300">
          <ResizablePanelGroup direction="horizontal" className="flex-1 overflow-hidden">
            <ResizablePanel defaultSize={45} minSize={35} maxSize={60}>
              <main className="h-full p-6">
                <TaskBuilder
                  onWorkflowComplete={handleInteractiveWorkflowComplete}
                  onOrchestrationComplete={handleOrchestrationComplete}
                  taskAgentPairs={taskAgentPairs}
                  selectedAgents={selectedAgents}
                  isExecuting={isExecuting}
                  apiResponseData={apiResponseData}
                  onThreadIdUpdate={handleThreadIdUpdate}
                  onExecutionResultsUpdate={handleExecutionResultsUpdate}
                  onViewCanvas={handleViewCanvas}
                  owner={user?.id}
                />
              </main>
            </ResizablePanel>

            <ResizableHandle withHandle />

            <ResizablePanel defaultSize={50} maxSize={65} minSize={35}>
              <OrchestrationDetailsSidebar
                ref={sidebarRef}
                executionResults={executionResults}
                threadId={currentThreadId || conversationState.thread_id || null}
                onThreadIdUpdate={handleThreadIdUpdate}
                onAcceptPlan={handleAcceptPlan}
                onRejectPlan={handleRejectPlan}
              />
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
      </SidebarInset>
    </>
  )
}

export default function ConversationPage() {
  return (
    <>
      <Navbar />
      <SidebarProvider defaultOpen={false}>
        <ConversationContent />
      </SidebarProvider>
    </>
  )
}
