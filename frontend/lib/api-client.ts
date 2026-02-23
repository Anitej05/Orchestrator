// lib/api-client.ts
import type { Agent, ProcessResponse, ConversationStatus, ConversationState, Message } from "./types"
import { authFetch } from './auth-fetch';
import { API_BASE_URL } from './config';

// Agent Management Functions
export async function fetchAllAgents(): Promise<Agent[]> {
  try {
    const response = await authFetch(`${API_BASE_URL}/api/agents/all`);
    
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
    const response = await authFetch(url);
    
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
    
    const response = await authFetch(`${API_BASE_URL}/api/agents/search?${params.toString()}`);
    
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

export async function rateAgentByName(agentName: string, rating: number): Promise<Agent> {
  try {
    if (rating < 0 || rating > 5) {
      throw new Error('Rating must be between 0 and 5');
    }

    const response = await authFetch(`${API_BASE_URL}/api/agents/by-name/${encodeURIComponent(agentName)}/rate`, {
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

export async function registerAgent(agentData: Agent): Promise<Agent> {
  try {
    const response = await authFetch(`${API_BASE_URL}/api/agents/register`, {
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
export async function startConversation(prompt: string, thread_id?: string, uploadedFiles?: any[]): Promise<ProcessResponse> {
  try {
    const response = await authFetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, thread_id, files: uploadedFiles }) // Pass thread_id and files if it exists
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

export async function continueConversation(response: string, threadId: string, uploadedFiles?: any[]): Promise<ProcessResponse> {
  try {
    const apiResponse = await authFetch(`${API_BASE_URL}/api/chat/continue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ response, thread_id: threadId, files: uploadedFiles })
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
    const response = await authFetch(`${API_BASE_URL}/api/chat/status/${threadId}`);
    
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

export async function uploadFiles(files: File[]): Promise<any[]> {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  try {
    const response = await authFetch(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading files:', error);
    throw error;
  }
}

// Legacy function for backward compatibility
export async function processPrompt(request: { prompt: string }): Promise<{
  message: string;
  thread_id: string;
  task_agent_pairs: any[];
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
