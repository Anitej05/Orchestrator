// app/page.tsx
"use client"

import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import AppSidebar from "@/components/app-sidebar"
import TaskBuilder from "@/components/task-builder"
import { SidebarProvider, SidebarInset, SidebarTrigger, useSidebar } from "@/components/ui/sidebar"
import OrchestrationDetailsSidebar, { type OrchestrationDetailsSidebarRef } from "@/components/orchestration-details-sidebar"
import { type TaskAgentPair, type ProcessResponse } from "@/lib/types"
import { useToast } from "@/hooks/use-toast"
import { useConversationStore } from "@/lib/conversation-store"
import { useWebSocketManager } from "@/hooks/use-websocket-conversation"
import { useUser } from "@clerk/nextjs"
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

function HomeContent() {
  const { open } = useSidebar()
  const router = useRouter()
  const { user, isLoaded: clerkLoaded } = useUser()
  const [taskAgentPairs, setTaskAgentPairs] = useState<TaskAgentPair[]>([])
  const [selectedAgents, setSelectedAgents] = useState<Record<string, string>>({})
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionResults, setExecutionResults] = useState<ExecutionResult[]>([])
  const [apiResponseData, setApiResponseData] = useState<ApiResponse | null>(null)
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null)
  const [isRestoring, setIsRestoring] = useState(false)
  const [isResetting, setIsResetting] = useState(false) // Track intentional resets
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

  // Redirect to conversation URL when thread_id is created (new conversation)
  useEffect(() => {
    if (!clerkLoaded || isResetting) return;
    
    // Only navigate if we have a VALID thread_id and we're on home page
    if (conversationState.thread_id && 
        typeof window !== 'undefined' && 
        window.location.pathname === '/' &&
        !isRestoring) {
      console.log('New thread_id detected, navigating to:', conversationState.thread_id);
      router.push(`/${conversationState.thread_id}`);
    }
  }, [conversationState.thread_id, clerkLoaded, router, isRestoring, isResetting]);

  // Check localStorage on initial mount only (not on every state change)
  useEffect(() => {
    if (!clerkLoaded) return;
    
    const savedThreadId = typeof window !== 'undefined' ? localStorage.getItem('thread_id') : null;
    // Only redirect if we have a saved thread AND we're on home page AND conversation state is empty
    if (savedThreadId && 
        !conversationState.thread_id && 
        typeof window !== 'undefined' && 
        window.location.pathname === '/') {
      console.log('Initial load: Redirecting to saved conversation:', savedThreadId);
      router.push(`/${savedThreadId}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clerkLoaded]); // Only run on Clerk load, not on state changes

  useEffect(() => {
    if (isRestoring) return; // Avoid side-effects while restoring existing convo
    if (conversationState.status === 'completed' && conversationState.final_response) {
      const result: ProcessResponse = {
        thread_id: conversationState.thread_id || '',
        message: "Completed",
        task_agent_pairs: conversationState.task_agent_pairs || [],
        final_response: conversationState.final_response,
        pending_user_input: false,
        question_for_user: null,
      };
      // Just update state presentation; don't auto-trigger execution
      setApiResponseData(result as ApiResponse);
      setTaskAgentPairs(result.task_agent_pairs);
    } else if (conversationState.status === 'error') {
      const lastMessage = conversationState.messages[conversationState.messages.length - 1];
      const errorMessage = lastMessage?.content || "An unknown error occurred.";
      toast({
        title: "Orchestration Error",
        description: errorMessage,
        variant: "destructive",
      });
    }
  }, [conversationState.status, conversationState.final_response, conversationState.thread_id, isRestoring]);

  // Handle URL parameters for auto-executing saved workflows or pre-seeded threads
  useEffect(() => {
    if (typeof window === 'undefined' || !clerkLoaded) return;
    
    const params = new URLSearchParams(window.location.search);
    const threadId = params.get('threadId');
    const promptParam = params.get('prompt');
    const executeNow = params.get('executeNow');
    
    // Priority 1: Pre-seeded workflow thread (from workflow execute endpoint)
    if (threadId) {
      // Clear the URL parameters
      window.history.replaceState({}, '', window.location.pathname);
      
      console.log('Loading pre-seeded workflow thread:', threadId);
      // Load this thread - it already has the plan pre-seeded
      // The plan will automatically show in the sidebar with approval buttons
      loadConversation(threadId).catch((error) => {
        console.error('Failed to load workflow:', error);
        toast({
          title: "Error",
          description: "Failed to load workflow. Please try again.",
          variant: "destructive"
        });
      });
      return;
    }
    
    // Priority 2: Auto-execute with prompt (legacy)
    if (promptParam && executeNow === 'true') {
      // Clear the URL parameters
      window.history.replaceState({}, '', window.location.pathname);
      
      // Trigger the chat with the saved workflow prompt
      console.log('Auto-executing saved workflow with prompt:', promptParam);
      
      // Find the TaskBuilder component and set the input
      // This will be handled by passing the prompt to TaskBuilder
      const event = new CustomEvent('autoExecuteWorkflow', { 
        detail: { prompt: promptParam } 
      });
      window.dispatchEvent(event);
    }
  }, [clerkLoaded]);

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
    // Navigate to the conversation URL
    router.push(`/${threadId}`);
  };

  const handleNewConversation = () => {
    console.log('Starting new conversation - clearing all state');
    setIsResetting(true);
    
    // Clear localStorage to prevent auto-restoration
    if (typeof window !== 'undefined') {
      localStorage.removeItem('thread_id')
    }
    
    // Reset all conversation state
    resetConversation();
    setTaskAgentPairs([]);
    setSelectedAgents({});
    setExecutionResults([]);
    setApiResponseData(null);
    setCurrentThreadId(null);
    
    // If we're on a conversation page, navigate to home
    if (typeof window !== 'undefined' && window.location.pathname !== '/') {
      router.push('/');
    }
    
    // Clear the resetting flag after state has settled
    setTimeout(() => {
      setIsResetting(false);
    }, 100);

    toast({
      title: "New conversation started",
      description: "Ready to start a new orchestration",
    });
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

  const handleThreadIdUpdate = (threadId: string) => {
    setCurrentThreadId(threadId)
    
    // Navigate to the new conversation URL when a thread is created
    if (threadId && typeof window !== 'undefined' && window.location.pathname === '/') {
      router.push(`/${threadId}`)
    }
  }

  const handleExecutionResultsUpdate = (results: ExecutionResult[]) => {
    setExecutionResults(results)
  }

  const handleAcceptPlan = async (modifiedPrompt?: string) => {
    try {
      // If prompt was modified, we need to re-plan (not implemented in initial version)
      // For now, just accept and execute the pre-seeded plan
      if (modifiedPrompt && modifiedPrompt !== conversationState.original_prompt) {
        // TODO: Implement re-planning with modified prompt
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
    toast({
      title: "Workflow cancelled",
      description: "Workflow execution was cancelled. You can start a new conversation."
    })
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
    <>
      <AppSidebar
        onConversationSelect={handleConversationSelect}
        onNewConversation={handleNewConversation}
        currentThreadId={conversationState.thread_id || undefined}
      />
      
      <SidebarInset className={!open ? "ml-16" : ""}>
        <div className="h-screen pt-[64px] bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-950 dark:to-gray-900 relative flex flex-col transition-all duration-300">
          {/* Main Content Area - Resizable */}
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

export default function Home() {
  return (
    <>
      <Navbar />
      <SidebarProvider defaultOpen={false}>
        <HomeContent />
      </SidebarProvider>
    </>
  )
}
