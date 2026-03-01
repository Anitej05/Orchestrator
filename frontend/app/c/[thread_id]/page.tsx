'use client'

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { useConversationStore } from "@/lib/conversation-store"
import { useWebSocketManager } from "@/hooks/use-websocket-conversation"
import { useToast } from "@/hooks/use-toast"
import { useUser } from "@clerk/nextjs"
import dynamic from "next/dynamic"
import { SidebarInset } from "@/components/ui/sidebar"
import OrchestrationDetailsSidebar from "@/components/orchestration-details-sidebar"
import { convertTasksToExecutionResults, type ExecutionResult } from "@/lib/execution-utils"
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable"

const InteractiveChatInterface = dynamic(
  () => import("@/components/interactive-chat-interface").then((mod) => mod.InteractiveChatInterface),
  { ssr: false }
);

const PAGE_BG = "min-h-screen bg-bg-page dark:bg-background text-text-primary"

export default function ConversationPage() {
  const params = useParams()
  const router = useRouter()
  const threadId = params?.thread_id as string
  const { toast } = useToast()
  const { user, isLoaded: clerkLoaded } = useUser()
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState(false)
  const [executionResults, setExecutionResults] = useState<ExecutionResult[]>([])
  const [isExecuting, setIsExecuting] = useState(false)
  
  const conversationState = useConversationStore()
  const { loadConversation, startConversation, continueConversation, resetConversation } = useConversationStore(state => state.actions)
  const isConversationLoading = useConversationStore((state: any) => state.isLoading)
  
  // Initialize WebSocket manager
  useWebSocketManager()

  // Load conversation on mount or when threadId changes
  useEffect(() => {
    if (!threadId) {
      console.log('⚠️ No threadId, redirecting to home')
      router.replace('/')
      return
    }

    // Wait for Clerk to load before attempting to fetch conversation
    if (!clerkLoaded) {
      console.log('⏳ Waiting for Clerk to load...')
      return
    }

    // Always reload if threadId changed, even if we have a conversation loaded
    // (in case user navigates between different conversations on the same page)
    const shouldLoad = !conversationState.thread_id || conversationState.thread_id !== threadId;
    
    if (!shouldLoad) {
      console.log('✓ Conversation already loaded:', threadId)
      setIsLoading(false)
      return
    }

    console.log('📥 Loading conversation from history:', threadId)
    console.log('   Current state thread_id:', conversationState.thread_id)
    console.log('   Will load:', threadId !== conversationState.thread_id)
    setIsLoading(true)
    
    loadConversation(threadId)
      .then(() => {
        console.log('✅ Conversation loaded successfully:', threadId)
        setIsLoading(false)
      })
      .catch((error) => {
        console.error('❌ Failed to load conversation:', threadId, error)
        setLoadError(true)
        toast({
          title: "Error",
          description: "Failed to load conversation. Redirecting to home...",
          variant: "destructive"
        })
        setTimeout(() => router.replace('/'), 2000)
      })
  }, [threadId, clerkLoaded])

  // Compute executionResults from task_agent_pairs whenever they change
  useEffect(() => {
    if (conversationState.task_agent_pairs && conversationState.task_agent_pairs.length > 0) {
      const results = convertTasksToExecutionResults(
        conversationState.task_agent_pairs,
        conversationState.final_response
      );
      setExecutionResults(results);
    } else {
      setExecutionResults([]);
    }
  }, [conversationState.task_agent_pairs, conversationState.final_response])

  // Debug effect - log state changes
  useEffect(() => {
    console.log('🔍 ConversationPage DEBUG:', {
      threadId,
      conversationThreadId: conversationState.thread_id,
      messagesCount: conversationState.messages?.length || 0,
      messagesPreview: conversationState.messages?.slice(0, 2).map(m => ({ type: m.type, contentLength: m.content?.length })),
      isLoading,
      isConversationLoading,
      hasContent: !!conversationState.messages?.length
    });
  }, [conversationState.thread_id, conversationState.messages, isLoading, isConversationLoading, threadId])

  // Handle canvas viewing
  const handleViewCanvas = (
    canvasContent: string,
    canvasType: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json'
  ) => {
    console.log('View canvas:', canvasType);
  };

  // Handle plan approval
  const handleAcceptPlan = async () => {
    try {
      await continueConversation("approve", [], false, user?.id)
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

  // Show loading state
  if (isLoading) {
    return (
      <div className={`${PAGE_BG} flex items-center justify-center`}>
        <div className="flex flex-col items-center space-y-3">
          <div className="h-12 w-12 rounded-full border-3 border-border-color border-t-brand-teal animate-spin" />
          <p className="text-text-secondary text-sm">Loading conversation…</p>
        </div>
      </div>
    )
  }

  // Show error state
  if (loadError) {
    return (
      <div className={`${PAGE_BG} flex items-center justify-center`}>
        <div className="flex flex-col items-center space-y-3">
          <p className="text-text-primary text-lg">Failed to load conversation</p>
          <p className="text-text-secondary text-sm">Redirecting to home...</p>
        </div>
      </div>
    )
  }

  // Render the same chat interface as home page
  return (
    <SidebarInset className="h-screen overflow-hidden">
      <div className="flex-1 bg-bg-page relative flex flex-col overflow-hidden w-full h-full max-w-full">
        <ResizablePanelGroup direction="horizontal" className="flex-1 overflow-hidden w-full h-full max-w-full">
          <ResizablePanel defaultSize={70} minSize={45} maxSize={75} className="overflow-hidden w-full min-w-0">
            <main className="h-full p-0">
              <div className="h-full flex flex-col">
                <InteractiveChatInterface
                  onError={(error) => {
                    toast({
                      title: "Error",
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
              executionResults={executionResults}
              threadId={conversationState.thread_id} 
            />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </SidebarInset>
  )
}
