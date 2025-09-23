// lib/api-client.ts
import type { Agent } from "./types"

// API base URL
const API_BASE_URL = 'http://localhost:8000';

// Type definitions aligned with backend API
export interface TaskAgentPair {
  task_name: string
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
  thread_id: string;
  status: 'pending_user_input' | 'completed' | 'processing' | 'idle';
  messages: Message[];
  isWaitingForUser: boolean;
  currentQuestion?: string;
}

export interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    task_agent_pairs?: TaskAgentPair[];
    progress?: number;
  };
}

// Agent Management Functions
export async function fetchAllAgents(): Promise<Agent[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/agents/all`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const agents = await response.json();
    return agents;
  } catch (error) {
    console.error('Error fetching agents from backend:', error);
    throw error;
  }
}

export async function fetchFilteredAgents(options: {
  maxPrice?: number;
  minRating?: number;
  status?: 'active' | 'inactive';
} = {}): Promise<Agent[]> {
  try {
    const params = new URLSearchParams();
    
    if (options.maxPrice !== undefined) {
      params.append('max_price', options.maxPrice.toString());
    }
    if (options.minRating !== undefined) {
      params.append('min_rating', options.minRating.toString());
    }
    if (options.status) {
      params.append('status_filter', options.status);
    }
    
    const url = `${API_BASE_URL}/api/agents/all${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const agents = await response.json();
    return agents;
  } catch (error) {
    console.error('Error fetching filtered agents from backend:', error);
    throw error;
  }
}

export async function searchAgents(options: {
  capabilities: string[];
  maxPrice?: number;
  minRating?: number;
  similarityThreshold?: number;
}): Promise<Agent[]> {
  try {
    const params = new URLSearchParams();
    
    options.capabilities.forEach(cap => {
      params.append('capabilities', cap);
    });
    
    if (options.maxPrice !== undefined) {
      params.append('max_price', options.maxPrice.toString());
    }
    if (options.minRating !== undefined) {
      params.append('min_rating', options.minRating.toString());
    }
    if (options.similarityThreshold !== undefined) {
      params.append('similarity_threshold', options.similarityThreshold.toString());
    }
    
    const response = await fetch(`${API_BASE_URL}/api/agents/search?${params.toString()}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const agents = await response.json();
    return agents;
  } catch (error) {
    console.error('Error searching agents:', error);
    throw error;
  }
}

export async function rateAgent(agentId: string, rating: number): Promise<Agent> {
  try {
    if (rating < 0 || rating > 5) {
      throw new Error('Rating must be between 0 and 5');
    }

    const response = await fetch(`${API_BASE_URL}/api/agents/${encodeURIComponent(agentId)}/rate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ rating }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const updatedAgent = await response.json();
    return updatedAgent;
  } catch (error) {
    console.error('Error rating agent:', error);
    throw error;
  }
}

export async function rateAgentByName(agentName: string, rating: number): Promise<Agent> {
  try {
    if (rating < 0 || rating > 5) {
      throw new Error('Rating must be between 0 and 5');
    }

    const response = await fetch(`${API_BASE_URL}/api/agents/by-name/${encodeURIComponent(agentName)}/rate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ rating }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const updatedAgent = await response.json();
    return updatedAgent;
  } catch (error) {
    console.error('Error rating agent by name:', error);
    throw error;
  }
}

export async function fetchAgentById(agentId: string): Promise<Agent> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/agents/${encodeURIComponent(agentId)}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const agent = await response.json();
    return agent;
  } catch (error) {
    console.error('Error fetching agent by ID:', error);
    throw error;
  }
}

export async function registerAgent(agentData: Agent): Promise<Agent> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/agents/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(agentData),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const registeredAgent = await response.json();
    return registeredAgent;
  } catch (error) {
    console.error('Error registering agent:', error);
    throw error;
  }
}

// Interactive Conversation Functions - Aligned with Backend API
export async function startConversation(prompt: string, thread_id?: string): Promise<ProcessResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, thread_id }) // Pass thread_id if it exists
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    return {
      message: data.message,
      thread_id: data.thread_id,
      task_agent_pairs: data.task_agent_pairs || [],
      final_response: data.final_response,
      pending_user_input: data.pending_user_input,
      question_for_user: data.question_for_user
    };
  } catch (error) {
    console.error('Error starting conversation:', error);
    throw error;
  }
}

export async function continueConversation(response: string, threadId: string): Promise<ProcessResponse> {
  try {
    const apiResponse = await fetch(`${API_BASE_URL}/api/chat/continue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ response, thread_id: threadId })
    });
    
    if (!apiResponse.ok) {
      throw new Error(`HTTP ${apiResponse.status}: ${apiResponse.statusText}`);
    }
    
    const data = await apiResponse.json();
    
    return {
      message: data.message,
      thread_id: data.thread_id,
      task_agent_pairs: data.task_agent_pairs || [],
      final_response: data.final_response,
      pending_user_input: data.pending_user_input,
      question_for_user: data.question_for_user
    };
  } catch (error) {
    console.error('Error continuing conversation:', error);
    throw error;
  }
}

export async function getConversationStatus(threadId: string): Promise<ConversationStatus> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/status/${threadId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    return {
      thread_id: data.thread_id,
      status: data.status,
      question_for_user: data.question_for_user,
      final_response: data.final_response,
      task_agent_pairs: data.task_agent_pairs || []
    };
  } catch (error) {
    console.error('Error getting conversation status:', error);
    throw error;
  }
}

export async function clearConversation(threadId: string): Promise<{ message: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/${threadId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error clearing conversation:', error);
    throw error;
  }
}

// Legacy function for backward compatibility
export async function processPrompt(request: { prompt: string }): Promise<{
  message: string;
  thread_id: string;
  task_agent_pairs: TaskAgentPair[];
  final_response?: string | null;
}> {
  try {
    const response = await startConversation(request.prompt);
    return {
      message: response.message,
      thread_id: response.thread_id,
      task_agent_pairs: response.task_agent_pairs,
      final_response: response.final_response
    };
  } catch (error) {
    console.error('Error processing prompt:', error);
    throw error;
  }
}

export async function fetchPlanFile(threadId: string): Promise<string> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/plan/${threadId}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        console.log(`No plan file found for threadId: ${threadId}`);
        return ""; // Return empty string if not found, so the UI doesn't break.
      }
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.content;
  } catch (error) {
    console.error('Error fetching plan file from backend:', error);
    throw error;
  }
}

export async function healthCheck(): Promise<{ status: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
}

// Static data exports for components
export const frameworks = ["CrewAI", "AutoGen", "LangGraph", "LangChain", "Custom"]

export const capabilities = [
  "find_travel_agent",
  "find_hotel_booking_agent", 
  "summarize_documents",
  "write_python_code",
  "research_sales_leads",
  "draft_marketing_emails",
  "Lead Generation",
  "Email Drafting", 
  "Translation",
  "Scheduling",
  "Payments",
  "Data Analysis",
  "Content Creation",
  "Research",
  "Customer Support",
  "Social Media Management",
]