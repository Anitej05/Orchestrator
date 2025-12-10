// components/task-builder.tsx
"use client"

import { useToast } from "@/hooks/use-toast"
import { useEffect } from "react"
import type { ProcessResponse, TaskAgentPair } from "@/lib/types"
import { useConversationStore } from "@/lib/conversation-store"
import WorkflowOrchestration from "./workflow-orchestration"
import dynamic from "next/dynamic"
import { useUser } from "@clerk/nextjs"
 

const InteractiveChatInterface = dynamic(
  () => import("@/components/interactive-chat-interface").then((mod) => mod.InteractiveChatInterface),
  { ssr: false }
);

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

interface TaskBuilderProps {
    onWorkflowComplete: (result: ProcessResponse) => void;
    onOrchestrationComplete: (results: ExecutionResult[]) => void;
    taskAgentPairs: TaskAgentPair[];
    selectedAgents: Record<string, string>;
    isExecuting: boolean;
    apiResponseData: ApiResponse | null;
    onThreadIdUpdate?: (threadId: string) => void;
    onExecutionResultsUpdate?: (results: ExecutionResult[]) => void;
    onViewCanvas?: (canvasContent: string, canvasType: 'html' | 'markdown') => void;
    owner?: string;
    onAcceptPlan?: (modifiedPrompt?: string) => Promise<void>;
}

export default function TaskBuilder({
    onWorkflowComplete,
    onOrchestrationComplete,
    taskAgentPairs,
    selectedAgents,
    isExecuting,
    apiResponseData,
    onThreadIdUpdate,
    onExecutionResultsUpdate,
    onViewCanvas,
    owner,
    onAcceptPlan
}: TaskBuilderProps) {
  const { toast } = useToast()
  const { user } = useUser()
  
  // Get conversation state from Zustand store
  const conversationState = useConversationStore();
  const isConversationLoading = useConversationStore((state: any) => state.isLoading);
  const { startConversation, continueConversation, resetConversation } = useConversationStore((state: any) => state.actions);

  useEffect(() => {
    // Debug helper: log conversation messages
    if (!conversationState) {
      console.debug('TaskBuilder: no conversationState from store');
      return;
    }

    if (!Array.isArray(conversationState.messages)) {
      console.warn('TaskBuilder: conversationState.messages is not an array', conversationState.messages);
      return;
    }

    console.debug('TaskBuilder: received conversation messages length=', conversationState.messages.length);
    if (conversationState.messages.length > 0) {
      console.debug('TaskBuilder: sample messages', conversationState.messages.slice(0, 5));
    }
  }, [conversationState]);

  return (
    <div className="h-full flex flex-col bg-gray-50/50 dark:bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-200 dark:border-gray-800 shadow-sm">
      <InteractiveChatInterface
        onWorkflowComplete={onWorkflowComplete}
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
        onViewCanvas={onViewCanvas}
        owner={owner}
        onAcceptPlan={onAcceptPlan}
      />

      {/* Hidden component to handle execution logic */}
      <div className="hidden">
        {isExecuting && apiResponseData && (
            <WorkflowOrchestration
              isExecuting={isExecuting}
              isDryRun={false}
              taskAgentPairs={taskAgentPairs}
              selectedAgents={selectedAgents}
              onComplete={onOrchestrationComplete}
              onThreadIdUpdate={onThreadIdUpdate}
              onExecutionResultsUpdate={onExecutionResultsUpdate}
            />
        )}
      </div>
    </div>
  )
}
