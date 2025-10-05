// components/task-builder.tsx
"use client"

import { useToast } from "@/hooks/use-toast"
import { useConversation } from "@/hooks/use-conversation"
import WorkflowOrchestration from "./workflow-orchestration"
import dynamic from "next/dynamic"
import type { ProcessResponse, TaskAgentPair, ConversationState } from "@/lib/types"

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
}

export default function TaskBuilder({
    onWorkflowComplete,
    onOrchestrationComplete,
    taskAgentPairs,
    selectedAgents,
    isExecuting,
    apiResponseData
}: TaskBuilderProps) {
  const { toast } = useToast()

  const { state: conversationState, isLoading: isConversationLoading, startConversation, continueConversation, resetConversation } = useConversation({
    onComplete: onWorkflowComplete,
    onError: (error) => {
      toast({
        title: "Orchestration Error",
        description: error,
        variant: "destructive",
      });
    }
  });

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
            />
        )}
      </div>
    </div>
  )
}
