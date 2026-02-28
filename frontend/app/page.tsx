// app/page.tsx
"use client"

import { useState, useRef, useEffect, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import dynamic from "next/dynamic"
import { SidebarInset, useSidebar } from "@/components/ui/sidebar"
import OrchestrationDetailsSidebar, { type OrchestrationDetailsSidebarRef } from "@/components/orchestration-details-sidebar"
import { type TaskAgentPair, type ProcessResponse } from "@/lib/types"
import { useToast } from "@/hooks/use-toast"
import { useConversationStore } from "@/lib/conversation-store"
import { useWebSocketManager } from "@/hooks/use-websocket-conversation"
import { useNewConversation } from "@/hooks/use-new-conversation"
import { useUser } from "@clerk/nextjs"
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

const InteractiveChatInterface = dynamic(
  () => import("@/components/interactive-chat-interface").then((mod) => mod.InteractiveChatInterface),
  { ssr: false }
);

function HomeContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { user, isLoaded: clerkLoaded } = useUser()
  const { open } = useSidebar()
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
  const { startConversation, continueConversation } = useConversationStore((state: any) => state.actions);

  // Initialize the WebSocket manager. It will automatically connect and keep the Zustand store in sync with backend updates.
  useWebSocketManager();

  // Save thread_id to localStorage and navigate when conversation is created
  useEffect(() => {
    if (!clerkLoaded || isResetting) return;
    
    // Save thread_id to localStorage for page reload persistence
    if (conversationState.thread_id && typeof window !== 'undefined') {
      localStorage.setItem('thread_id', conversationState.thread_id);
      console.log('Saved thread_id to localStorage:', conversationState.thread_id);
    }
    
    // Clear localStorage if thread_id is cleared
    if (!conversationState.thread_id && typeof window !== 'undefined') {
      localStorage.removeItem('thread_id');
      console.log('Cleared thread_id from localStorage');
    }
  }, [conversationState.thread_id, clerkLoaded, router, isRestoring, isResetting]);

  // On initial load, if there's a saved thread_id in localStorage and we're on home page,
  // load that conversation into the store but DON'T redirect (stay on home page)
  useEffect(() => {
    if (!clerkLoaded) return;
    
    // If user navigates directly to home, check if we should clear state
    if (typeof window !== 'undefined' && window.location.pathname === '/') {
      // If there's no query parameters and we have a thread_id in state but NOT in localStorage
      const params = new URLSearchParams(window.location.search);
      const hasQueryParams = params.has('threadId') || params.has('prompt');
      const savedThreadId = localStorage.getItem('thread_id');
      
      if (!hasQueryParams && conversationState.thread_id && !savedThreadId) {
        // User manually navigated to home - clear the state
        console.log('User navigated to home manually - clearing state');
        setIsResetting(true);
        useConversationStore.setState({
          metadata: {},
          plan: [],
          task_agent_pairs: [],
          messages: [],
          final_response: undefined,
          thread_id: undefined,
          status: 'idle',
          canvas_content: undefined,
          has_canvas: false,
          task_statuses: {},
          current_executing_task: null,
        });
        resetConversation();
        setTaskAgentPairs([]);
        setSelectedAgents({});
        setExecutionResults([]);
        setApiResponseData(null);
        setCurrentThreadId(null);
        setTimeout(() => setIsResetting(false), 200);
        return;
      }
    }
    
    const savedThreadId = typeof window !== 'undefined' ? localStorage.getItem('thread_id') : null;
    // Only load if we have a saved thread AND conversation state is empty AND we're on home page
    if (savedThreadId && 
        !conversationState.thread_id && 
        typeof window !== 'undefined' && 
        window.location.pathname === '/') {
      console.log('Initial load: Loading saved conversation into state:', savedThreadId);
      // Load the conversation but stay on home page
      setIsRestoring(true);
      loadConversation(savedThreadId)
        .catch(err => console.error('Failed to restore conversation:', err))
        .finally(() => setIsRestoring(false));
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
      
      // Generate execution results from task_agent_pairs (previously done by WorkflowOrchestration)
      if (result.task_agent_pairs.length > 0) {
        const results: ExecutionResult[] = result.task_agent_pairs.map((pair) => ({
          taskId: pair.task_name,
          taskDescription: pair.task_name.replace(/_/g, " ").replace(/\b\w/g, (l: string) => l.toUpperCase()),
          agentName: pair.primary?.name || "Unknown Agent",
          status: "success",
          output: conversationState.final_response || `Successfully completed: ${pair.task_name.replace(/_/g, " ")}`,
          cost: pair.primary?.price_per_call_usd || 0,
          executionTime: Math.floor((Math.random() * 5 + 3) * 10) / 10,
        }));
        
        setExecutionResults(results);
        setIsExecuting(false);
        
        const totalCost = results.reduce((sum, result) => sum + result.cost, 0);
        toast({
          title: "Workflow executed successfully",
          description: `All ${results.length} tasks completed. Total cost: $${totalCost.toFixed(4)}`,
        });
      }
      
      // Update thread ID (previously done by WorkflowOrchestration's handleThreadIdUpdate)
      if (conversationState.thread_id) {
        setCurrentThreadId(conversationState.thread_id);
      }
    } else if (conversationState.status === 'error') {
      const lastMessage = conversationState.messages[conversationState.messages.length - 1];
      const errorMessage = lastMessage?.content || "An unknown error occurred.";
      toast({
        title: "Orchestration Error",
        description: errorMessage,
        variant: "destructive",
      });
    }
  }, [conversationState.status, conversationState.final_response, conversationState.thread_id, conversationState.task_agent_pairs, isRestoring, toast]);

  // Handle URL parameters for auto-executing saved workflows
  useEffect(() => {
    if (!clerkLoaded) return;
    
    const promptParam = searchParams.get('prompt');
    const executeNow = searchParams.get('executeNow');
    const threadId = searchParams.get('threadId');
    
    // If threadId is in query params, redirect to proper route
    if (threadId) {
      router.replace(`/c/${threadId}`);
      return;
    }
    
    // Auto-execute with prompt (for saved workflows)
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
  }, [clerkLoaded, searchParams, router]);

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

  const handleViewCanvas = (
    canvasContent: string,
    canvasType: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json'
  ) => {
    sidebarRef.current?.viewCanvas(canvasContent, canvasType);
  };


  const handleConversationSelect = async (threadId: string) => {
    // Don't navigate - just load the conversation into the store
    // This gives us ChatGPT-like behavior: update content without full page reload
    console.log('Loading conversation:', threadId);
    setIsRestoring(true);
    
    try {
      const { loadConversation: loadConv } = useConversationStore.getState().actions;
      await loadConv(threadId);
      
      // Update URL without navigation (for sharing/bookmarking)
      window.history.replaceState({}, '', `/c/${threadId}`);
      
    } catch (error) {
      console.error('Failed to load conversation:', error);
      toast({
        title: "Error",
        description: "Failed to load conversation. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsRestoring(false);
    }
  };

  const { startNewConversation } = useNewConversation()

  const handleNewConversation = startNewConversation

  // Callback handlers moved to useEffect above that listens to conversationState.status === 'completed'
  const handleThreadIdUpdate = (threadId: string) => {
    setCurrentThreadId(threadId)
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

  return (
    <SidebarInset className="h-screen overflow-hidden">
      <div className="flex-1 bg-bg-page relative flex flex-col overflow-hidden w-full h-full max-w-full">
        {/* Main Content Area - Resizable */}
        <ResizablePanelGroup direction="horizontal" className="flex-1 overflow-hidden w-full h-full max-w-full">
            <ResizablePanel defaultSize={70} minSize={45} maxSize={75} className="overflow-hidden w-full min-w-0">
              <main className="h-full p-0">
                <div className="h-full flex flex-col">
                  <InteractiveChatInterface
                    onWorkflowComplete={handleInteractiveWorkflowComplete}
                    onError={(error) => {
                      toast({
                        title: "Orchestration Error",
                        description: error,
                        variant: "destructive",
                      });
                    }}
                    state={conversationState}
                    isLoading={isConversationLoading}
                    startConversation={startConversation}
                    continueConversation={continueConversation}
                    resetConversation={resetConversation}
                    onViewCanvas={handleViewCanvas}
                    owner={clerkLoaded && user?.id ? user.id : undefined}
                    onAcceptPlan={handleAcceptPlan}
                  />
                </div>
              </main>
            </ResizablePanel>

            <ResizableHandle withHandle />

            <ResizablePanel defaultSize={50} maxSize={65} minSize={35} className="overflow-hidden w-full min-w-0">
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
  )
}

export default function Home() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HomeContent />
    </Suspense>
  )
}
