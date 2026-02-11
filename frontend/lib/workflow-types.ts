/**
 * Shared types for workflow execution across all UI modes
 */

export type WorkflowEventType = 
  | '__start__'
  | '__end__'
  | '__error__'
  | 'parse_prompt'
  | 'agent_directory_search'
  | 'rank_agents'
  | 'plan_execution'
  | 'task_started'
  | 'task_completed'
  | 'task_failed'
  | 'final_response'
  | 'user_input_required'
  | 'plan_approval_required'
  | 'stream_chunk'
  | string; // Allow other node types

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface WorkflowTask {
  taskId: string;
  taskName: string;
  taskDescription: string;
  agentName: string;
  agentId: string;
  status: TaskStatus;
  startTime?: number;
  endTime?: number;
  executionTime?: number;
  cost?: number;
  output?: string;
  error?: string;
}

export interface WorkflowMessage {
  id: string;
  type: 'system' | 'user' | 'assistant' | 'error' | 'plan_approval';
  content: string;
  timestamp: number;
  metadata?: {
    node?: string;
    progress?: number;
    task_agent_pairs?: any[];
    planData?: any;
    [key: string]: any;
  };
}

export interface WorkflowEvent {
  node: WorkflowEventType;
  type?: WorkflowEventType;
  data?: any;
  thread_id?: string;
  task_name?: string;
  task_description?: string;
  agent_name?: string;
  status?: string;
  output?: string;
  error?: string;
  execution_time?: number;
  cost?: number;
  progress_percentage?: number;
  node_sequence?: number;
  timestamp?: number;
  content?: string;
  message?: string;
  description?: string;
  [key: string]: any;
}

export interface WorkflowState {
  isConnected: boolean;
  isExecuting: boolean;
  threadId: string | null;
  messages: WorkflowMessage[];
  tasks: Record<string, WorkflowTask>;
  progress: number;
  currentPhase: string;
  error: string | null;
  isWaitingForUser: boolean;
  currentQuestion: string | null;
  planData: any | null;
  metadata: Record<string, any>;
}

export interface WorkflowExecutionOptions {
  autoConnect?: boolean;
  reconnect?: boolean;
  maxReconnectAttempts?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
  onComplete?: (result: any) => void;
}

export interface TaskAgentPair {
  task_name: string;
  task_description?: string;
  primary: {
    id: string;
    name: string;
    rating?: number;
    price_per_call_usd?: number;
    capabilities?: string[];
  };
  fallbacks: Array<{
    id: string;
    name: string;
    rating?: number;
    price_per_call_usd?: number;
    capabilities?: string[];
  }>;
}

export interface ExecutionResult {
  taskId: string;
  taskDescription: string;
  agentName: string;
  status: string;
  output: string;
  cost: number;
  executionTime: number;
}
