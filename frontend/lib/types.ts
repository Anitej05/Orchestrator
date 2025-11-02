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

export interface ConversationState {
  thread_id?: string;
  status: 'pending_user_input' | 'completed' | 'processing' | 'idle' | 'error';
  messages: Message[];
  isWaitingForUser: boolean;
  currentQuestion?: string;
  task_agent_pairs?: TaskAgentPair[];
  final_response?: string;
  metadata?: any;
  uploaded_files?: FileObject[];
  plan?: any[];
  // Canvas feature fields
  canvas_content?: string;
  canvas_type?: 'html' | 'markdown';
  has_canvas?: boolean;
  // Plan approval fields
  approval_required?: boolean;
  estimated_cost?: number;
  task_count?: number;
  task_plan?: any[];
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
  canvas_type?: 'html' | 'markdown';
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
