// components/task-builder.tsx
"use client"

import { useToast } from "@/hooks/use-toast"
import { useEffect } from "react"
import type { ProcessResponse, TaskAgentPair, ConversationState } from "@/lib/types"
import WorkflowOrchestration from "./workflow-orchestration"
import dynamic from "next/dynamic"
 

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
  // Conversation props (lifted from parent)
  conversationState: ConversationState;
  isConversationLoading: boolean;
  startConversation: (input: string, files?: File[]) => Promise<void>;
  continueConversation: (input: string) => Promise<void>;
  resetConversation: () => void;
  loadConversation: (threadId: string) => Promise<void>;
}

export default function TaskBuilder({
    onWorkflowComplete,
    onOrchestrationComplete,
    taskAgentPairs,
    selectedAgents,
    isExecuting,
    apiResponseData,
    onThreadIdUpdate,
    onExecutionResultsUpdate
    ,
    conversationState,
    isConversationLoading,
    startConversation,
    continueConversation,
    resetConversation,
    loadConversation
}: TaskBuilderProps) {
  const { toast } = useToast()

  useEffect(() => {
    // Debug helper: log conversation messages when they arrive so we can confirm
    // the parent `useConversation` state is being passed correctly.
    // Leave as console.debug to avoid spamming production logs; remove later.
    if (!conversationState) {
      console.debug('TaskBuilder: no conversationState prop received');
      return;
    }

    if (!Array.isArray(conversationState.messages)) {
      console.warn('TaskBuilder: conversationState.messages is not an array', conversationState.messages);
      return;
    }

    console.debug('TaskBuilder: received conversation messages length=', conversationState.messages.length);
    if (conversationState.messages.length > 0) {
      // Log only the first few messages to avoid huge logs
      console.debug('TaskBuilder: sample messages', conversationState.messages.slice(0, 5));
    }
  }, [conversationState]);

  return (
    <div className="h-full flex flex-col bg-white rounded-lg border shadow-sm">
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
