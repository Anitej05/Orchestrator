// app/page.tsx
"use client"

import { useState, useRef, useEffect } from "react"
import AppSidebar from "@/components/app-sidebar"
import TaskBuilder from "@/components/task-builder"
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Users } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import OrchestrationDetailsSidebar, { type OrchestrationDetailsSidebarRef } from "@/components/orchestration-details-sidebar"
import { type TaskAgentPair, type ProcessResponse } from "@/lib/types"
import { useToast } from "@/hooks/use-toast"
import { useConversationStore } from "@/lib/conversation-store"
import { useWebSocketManager } from "@/hooks/use-websocket-conversation"
import { OrchestrationProgress } from "@/components/orchestration-progress"
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable"
import Navbar from "@/components/navbar"

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
  const sidebarRef = useRef<OrchestrationDetailsSidebarRef>(null);

  // --- Zustand Store Integration ---
  // The entire conversation state is now managed by the Zustand store.
  const conversationState = useConversationStore();
  const { 
    resetConversation, 
    loadConversation 
  } = useConversationStore(state => state.actions);
  const isConversationLoading = useConversationStore((state: any) => state.isLoading);

  // Initialize the WebSocket manager. It will automatically connect and keep the Zustand store in sync with backend updates.
  useWebSocketManager();
  
  // Load conversation from localStorage on page load (persistence)
  useEffect(() => {
    const savedThreadId = localStorage.getItem('thread_id');
    if (savedThreadId && !conversationState.thread_id) {
      console.log('Restoring conversation from localStorage:', savedThreadId);
      loadConversation(savedThreadId);
    }
  }, []); // Run only once on mount
  
  useEffect(() => {
    if (conversationState.status === 'completed' && conversationState.final_response) {
      const result: ProcessResponse = {
        thread_id: conversationState.thread_id || '',
        message: "Completed",
        task_agent_pairs: conversationState.task_agent_pairs || [],
        final_response: conversationState.final_response,
        pending_user_input: false,
        question_for_user: null,
      };
      handleInteractiveWorkflowComplete(result);
    } else if (conversationState.status === 'error') {
      const lastMessage = conversationState.messages[conversationState.messages.length - 1];
      const errorMessage = lastMessage?.content || "An unknown error occurred.";
      toast({
        title: "Orchestration Error",
        description: errorMessage,
        variant: "destructive",
      });
    }
  }, [conversationState.status, conversationState.final_response, conversationState.thread_id]);

  const handleInteractiveWorkflowComplete = (result: ProcessResponse) => {
    setApiResponseData(result as ApiResponse);
    setTaskAgentPairs(result.task_agent_pairs);

    const initialSelections: Record<string, string> = {};
    result.task_agent_pairs.forEach((pair: TaskAgentPair) => {
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
  
  const handleViewCanvas = (canvasContent: string, canvasType: 'html' | 'markdown') => {
    sidebarRef.current?.viewCanvas(canvasContent, canvasType);
  };


  const handleConversationSelect = async (threadId: string) => {
    try {
      await loadConversation(threadId);
      
      // Update local state with loaded conversation data
      if (conversationState.task_agent_pairs && conversationState.task_agent_pairs.length > 0) {
        setTaskAgentPairs(conversationState.task_agent_pairs);
        
        // Update selected agents
        const initialSelections: Record<string, string> = {};
        conversationState.task_agent_pairs.forEach((pair: TaskAgentPair) => {
          initialSelections[pair.task_name] = pair.primary.id;
        });
        setSelectedAgents(initialSelections);
      } else {
        // If no task_agent_pairs in conversation state, try to get them from the API
        try {
          const response = await fetch(`http://localhost:8000/api/chat/status/${threadId}`);
          if (response.ok) {
            const statusData = await response.json();
            if (statusData.task_agent_pairs && statusData.task_agent_pairs.length > 0) {
              setTaskAgentPairs(statusData.task_agent_pairs);
              
              const initialSelections: Record<string, string> = {};
              statusData.task_agent_pairs.forEach((pair: any) => {
                initialSelections[pair.task_name] = pair.primary.id;
              });
              setSelectedAgents(initialSelections);
            }
          }
        } catch (err) {
          console.error("Error fetching conversation status:", err);
        }
      }
      
      toast({
        title: "Conversation loaded",
        description: `Loaded conversation ${threadId}`,
      });

      // Refresh the plan in the sidebar after loading the conversation
      if (sidebarRef.current) {
        sidebarRef.current.refreshPlan();
      }
    } catch (error) {
      toast({
        title: "Error loading conversation",
        description: "Failed to load the selected conversation",
        variant: "destructive",
      });
    }
  };
  
  const handleNewConversation = () => {
    resetConversation();
    setTaskAgentPairs([]);
    setSelectedAgents({});
    setExecutionResults([]);
    setApiResponseData(null);
    setCurrentThreadId(null);
    
    toast({
      title: "New conversation started",
      description: "Ready to start a new orchestration",
    });
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

  // This useEffect was causing issues with sidebar content display
  // The sidebar should update naturally through its existing mechanisms
  // useEffect(() => {
  //   // Check if the last message is an assistant message (AI response)
  //   if (conversationState.messages.length > 0) {
  //     const lastMessage = conversationState.messages[conversationState.messages.length - 1];
  //     if (lastMessage.type === 'assistant') {
  //       // Small delay to ensure the backend has time to update the plan file
  //       const timer = setTimeout(() => {
  //         if (sidebarRef.current) {
  //           console.log("Calling refreshPlan from page.tsx");
  //           sidebarRef.current.refreshPlan();
  //         }
  //       }, 1500); // 1.5 second delay to ensure plan file is updated
  //       
  //       // Cleanup function to clear the timeout if the effect runs again before timeout completes
  //       return () => clearTimeout(timer);
  //     }
  //   }
  // }, [conversationState.messages]);
  
  return (
    <SidebarProvider>
      <div className="flex h-screen w-full overflow-hidden">
        <AppSidebar
          onConversationSelect={handleConversationSelect}
          onNewConversation={handleNewConversation}
          currentThreadId={conversationState.thread_id || undefined}
        />
        <SidebarInset className="flex-1 flex flex-col overflow-hidden">
          <Navbar />
          <div className="flex-1 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-950 dark:to-gray-900 overflow-hidden">
            {/* Main Content Area - Resizable */}
            <ResizablePanelGroup direction="horizontal" className="h-full">
              <ResizablePanel defaultSize={70} minSize={50}>
                <main className="h-full p-6 overflow-auto">
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
                  />
                </main>
              </ResizablePanel>
              
              <ResizableHandle withHandle />

              <ResizablePanel defaultSize={30} maxSize={50} minSize={20}>
                <OrchestrationDetailsSidebar
                  ref={sidebarRef}
                  executionResults={executionResults}
                  threadId={currentThreadId || conversationState.thread_id || null}
                  onThreadIdUpdate={handleThreadIdUpdate}
                />
              </ResizablePanel>
            </ResizablePanelGroup>
          </div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}
