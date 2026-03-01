// lib/types.ts

// Keep existing type definitions
export interface AgentEndpoint {
  endpoint: string;
  http_method: "GET" | "POST" | "PUT" | "DELETE";
  description?: string;
}

export interface Agent {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  status: "active" | "inactive";
  rating?: number;
  rating_count?: number;
  owner_id?: string;
  price_per_call_usd?: number;
  public_key_pem?: string;
  endpoints: AgentEndpoint[];
}

export interface TaskAgentPair {
  task_name: string
  task_description: string
  primary: Agent & { score?: number }
  fallbacks: (Agent & { score?: number })[]
}

export interface ProcessResponse {
  message: string;
  thread_id: string;
  task_agent_pairs: TaskAgentPair[];
  final_response: string | null;
  pending_user_input: boolean;
  question_for_user: string | null;
}

export interface ConversationStatus {
  thread_id: string;
  status: 'pending_user_input' | 'completed' | 'processing';
  question_for_user: string | null;
  final_response: string | null;
  task_agent_pairs: TaskAgentPair[];
}

export interface TaskStatus {
  status: 'pending' | 'running' | 'completed' | 'failed';
  taskName: string;
  taskId?: string;
  taskDescription?: string;
  activityDescription?: string;  // Short, user-friendly one-liner for real-time display
  resultSummary?: string;  // Brief outcome summary for completed tasks
  agentName?: string;
  startedAt?: Date;
  completedAt?: Date;
  startTime?: number;  // Unix timestamp
  executionTime?: number;
  cost?: number;
  result?: any;
  error?: string;
  is_dialogue?: boolean;
}

export interface TodoItem {
  id?: string;
  task_id?: string;
  description: string;
  status: string;
  priority?: number;
  payload?: Record<string, any>;
  result?: any;
  error?: string;
  created_at?: number | string;
  updated_at?: number | string;
}

// NEW: Action History Entry - Shows actual agent/tool execution
export interface ActionHistoryEntry {
  iteration: number;
  action_type: string;
  resource_id: string;       // Agent/tool name (e.g., "gmail_agent", "send_email")
  instruction: string;
  success: boolean;
  result_summary: string;
  execution_time_ms: number;
  error?: string;
}

export interface ConversationState {
  thread_id?: string;
  status: 'pending_user_input' | 'completed' | 'processing' | 'idle' | 'error' | 'planning_complete';
  messages: Message[];
  isWaitingForUser: boolean;
  currentQuestion?: string;
  task_agent_pairs?: TaskAgentPair[];
  final_response?: string;
  metadata?: any;
  uploaded_files?: FileObject[];
  plan?: any[];
  // Original prompt for pre-seeded workflows
  original_prompt?: string;
  // Real-time task execution tracking
  task_statuses?: Record<string, TaskStatus>;
  current_executing_task?: string | null;
  todo_list?: TodoItem[];
  // NEW: Brain thinking state for real-time AI reasoning display
  brain_reasoning?: string;
  // NEW: Orchestration state from OMNI-DISPATCHER
  action_history?: ActionHistoryEntry[];  // Complete execution log with agent/tool info
  // Canvas feature fields
  canvas_content?: string;  // Legacy: HTML/markdown string
  canvas_data?: Record<string, any>;  // Preferred: Structured data
  canvas_type?: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json';
  has_canvas?: boolean;
  canvas_title?: string;
  canvas_metadata?: Record<string, any>;
  canvas_requires_confirmation?: boolean;
  canvas_confirmation_message?: string;
  browser_view?: string;
  plan_view?: string;
  current_view?: 'browser' | 'plan';
  // Plan approval fields
  approval_required?: boolean;
  estimated_cost?: number;
  task_count?: number;
  task_plan?: any[];
  // Action approval fields (Omni-Dispatcher)
  pending_action_approval?: boolean;
  pending_action?: any;
  // Canvas confirmation fields
  pending_confirmation?: boolean;
  pending_confirmation_task?: {
    task_name: string;
    agent_name: string;
    canvas_display: any;
  };
  // Flag to track if this is a newly started conversation (vs loading old one)
  isNewConversation?: boolean;
}

export interface Attachment {
  name: string;
  type: string; // e.g., 'image/png', 'application/pdf'
  content: string; // For images, this will be a data URL (base64)
}

// Updated Message interface with the new `attachments` field
export interface BrowsingTraceStep {
  step_number: number;
  action: string;
  description: string;
  status: 'success' | 'error' | 'pending';
  duration?: number;
  timestamp: string;
}

export interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
  metadata?: {
    task_agent_pairs?: TaskAgentPair[];
    progress?: number;
  };
  // Canvas information for this specific message
  canvas_content?: string;
  canvas_type?: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json';
  has_canvas?: boolean;
  // Browser automation fields
  is_browser_task?: boolean;
  browser_in_progress?: boolean;
  browsing_trace?: BrowsingTraceStep[];
  screenshot_files?: FileObject[];
  show_trace?: boolean;  // UI state for collapsible trace
}

export type FileObject = {
  file_name: string;
  file_path: string;
  file_type: string;
};
