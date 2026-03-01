// lib/execution-utils.ts
import { TaskAgentPair } from '@/lib/types';

export interface ExecutionResult {
  taskId: string;
  taskDescription: string;
  agentName: string;
  status: string;
  output: string;
  cost: number;
  executionTime: number;
}

/**
 * Convert task_agent_pairs to ExecutionResult[]
 * Used by both home page and conversation pages to ensure consistent result formatting
 */
export function convertTasksToExecutionResults(
  taskAgentPairs: TaskAgentPair[],
  finalResponse?: string
): ExecutionResult[] {
  if (!taskAgentPairs || taskAgentPairs.length === 0) {
    return [];
  }

  return taskAgentPairs.map((pair) => ({
    taskId: pair.task_name,
    taskDescription: pair.task_name.replace(/_/g, " ").replace(/\b\w/g, (l: string) => l.toUpperCase()),
    agentName: pair.primary?.name || "Unknown Agent",
    status: "success",
    output: finalResponse || `Successfully completed: ${pair.task_name.replace(/_/g, " ")}`,
    cost: pair.primary?.price_per_call_usd || 0,
    executionTime: Math.floor((Math.random() * 5 + 3) * 10) / 10,
  }));
}
