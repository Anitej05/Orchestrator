// app/page.tsx
"use client"

import { useState } from "react"
import AppSidebar from "@/components/app-sidebar"
import TaskBuilder from "@/components/task-builder"
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Users } from "lucide-react"
import OrchestrationDetailsSidebar from "@/components/orchestration-details-sidebar"
import { type TaskAgentPair, type ProcessResponse } from "@/lib/types"
import { useToast } from "@/hooks/use-toast"
import { useConversation } from "@/hooks/use-conversation"
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

export default function Home() {
  const [taskAgentPairs, setTaskAgentPairs] = useState<TaskAgentPair[]>([])
  const [selectedAgents, setSelectedAgents] = useState<Record<string, string>>({})
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionResults, setExecutionResults] = useState<ExecutionResult[]>([])
  const [apiResponseData, setApiResponseData] = useState<ApiResponse | null>(null)
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null)
  const { toast } = useToast()

  const { state: conversationState, isLoading: isConversationLoading, startConversation, continueConversation, resetConversation, loadConversation } = useConversation({
    onComplete: (result) => {
      console.log('Conversation completed:', result);
      handleInteractiveWorkflowComplete(result);
    },
    onError: (error) => {
      console.error('Conversation error:', error);
      toast({
        title: "Orchestration Error",
        description: error,
        variant: "destructive",
      });
    }
  });

  const handleInteractiveWorkflowComplete = (result: ProcessResponse) => {
    setApiResponseData(result as ApiResponse);
    setTaskAgentPairs(result.task_agent_pairs);

    const initialSelections: Record<string, string> = {};
    result.task_agent_pairs.forEach((pair) => {
      initialSelections[pair.task_name] = pair.primary.id;
    });
    setSelectedAgents(initialSelections);

    if (result.task_agent_pairs.length > 0 && !result.pending_user_input) {
      setIsExecuting(true);
      toast({
        title: "Starting workflow execution",
        description: `Executing ${result.task_agent_pairs.length} tasks.`,
      });
    }
  };


  const handleConversationSelect = async (threadId: string) => {
    try {
      await loadConversation(threadId);
      
      // Update local state with loaded conversation data
      if (conversationState.task_agent_pairs && conversationState.task_agent_pairs.length > 0) {
        setTaskAgentPairs(conversationState.task_agent_pairs);
        
        // Update selected agents
        const initialSelections: Record<string, string> = {};
        conversationState.task_agent_pairs.forEach((pair) => {
          initialSelections[pair.task_name] = pair.primary.id;
        });
        setSelectedAgents(initialSelections);
      }
      
      toast({
        title: "Conversation loaded",
        description: `Loaded conversation ${threadId}`,
      });
    } catch (error) {
      toast({
        title: "Error loading conversation",
        description: "Failed to load the selected conversation",
        variant: "destructive",
      });
    }
  };

  const handleOrchestrationComplete = (results: ExecutionResult[]) => {
    setExecutionResults(results)
    setIsExecuting(false)

    const totalCost = results.reduce((sum, result) => sum + result.cost, 0)

    toast({
      title: "Workflow executed successfully",
      description: `All ${results.length} tasks completed. Total cost: ${totalCost.toFixed(4)}`,
    })
  }

  const handleThreadIdUpdate = (threadId: string) => {
    setCurrentThreadId(threadId)
  }

  const handleExecutionResultsUpdate = (results: ExecutionResult[]) => {
    setExecutionResults(results)
  }
  
  return (
    <SidebarProvider>
      <AppSidebar
        onConversationSelect={handleConversationSelect}
        currentThreadId={conversationState.thread_id}
      />
      <SidebarInset>
        <div className="h-screen bg-gray-50 relative flex flex-col">
          {/* Header */}
          <div className="flex-shrink-0 sticky top-0 z-10 bg-white border-b border-gray-200 px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <SidebarTrigger className="h-8 w-8" />
                <h1 className="text-xl font-semibold text-gray-900">Orchestrator</h1>
              </div>
              <Link href="/agents">
                <Button className="bg-blue-600 hover:bg-blue-700 shadow-sm">
                  <Users className="w-4 h-4 mr-2" />
                  See Agents
                </Button>
              </Link>
            </div>
          </div>

          {/* Main Content Area - Resizable */}
          <ResizablePanelGroup direction="horizontal" className="flex-1 overflow-hidden">
            <ResizablePanel defaultSize={70} minSize={50}>
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
              // Conversation props passed down so TaskBuilder/InteractiveChatInterface
              // can render the loaded conversation state.
              conversationState={conversationState}
              isConversationLoading={isConversationLoading}
              startConversation={startConversation}
              continueConversation={continueConversation}
              resetConversation={resetConversation}
              loadConversation={loadConversation}
                 />
              </main>
            </ResizablePanel>
            
            <ResizableHandle withHandle />

            <ResizablePanel defaultSize={30} maxSize={40} minSize={25}>
              <OrchestrationDetailsSidebar
                executionResults={executionResults}
                threadId={currentThreadId || conversationState.thread_id}
                taskAgentPairs={taskAgentPairs}
                messages={conversationState.messages}
                onThreadIdUpdate={handleThreadIdUpdate}
              />
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
