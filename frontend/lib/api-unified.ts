// lib/api-unified.ts - Single consolidated API client
import type { Agent } from "./types"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Unified interfaces
export interface ProcessRequest {
  prompt: string
}

export interface TaskAgentPair {
  task_name: string
  primary: Agent & { score?: number }
  fallbacks: (Agent & { score?: number })[]
}

export interface ProcessResponse {
  message: string
  thread_id: string
  task_agent_pairs: TaskAgentPair[]
  final_response?: string
}

export interface EndpointDetail {
  endpoint: string
  http_method: string
  description?: string
}

// Single API client with all functionality
class UnifiedAPIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // Agent fetching methods
  async getAllAgents(): Promise<Agent[]> {
    const response = await this.fetch('/api/agents/all');
    return response.json();
  }

  async getFilteredAgents(options: {
    maxPrice?: number;
    minRating?: number;
    status?: 'active' | 'inactive';
  } = {}): Promise<Agent[]> {
    const params = new URLSearchParams();
    Object.entries(options).forEach(([key, value]) => {
      if (value !== undefined) {
        params.append(key === 'status' ? 'status_filter' : key, value.toString());
      }
    });
    
    const url = `/api/agents/all${params.toString() ? `?${params}` : ''}`;
    const response = await this.fetch(url);
    return response.json();
  }

  async searchAgents(params: {
    capabilities: string[];
    max_price?: number;
    min_rating?: number;
    similarity_threshold?: number;
  }): Promise<Agent[]> {
    const searchParams = new URLSearchParams();
    params.capabilities.forEach(cap => searchParams.append('capabilities', cap));
    
    Object.entries(params).forEach(([key, value]) => {
      if (key !== 'capabilities' && value !== undefined) {
        searchParams.append(key, value.toString());
      }
    });

    const response = await this.fetch(`/agents/search?${searchParams}`);
    return response.json();
  }

  async getAgent(agentId: string): Promise<Agent> {
    const response = await this.fetch(`/agents/${encodeURIComponent(agentId)}`);
    return response.json();
  }

  // Agent management methods
  async registerAgent(agentData: Agent): Promise<Agent> {
    const response = await this.fetch('/agents/register', {
      method: 'POST',
      body: JSON.stringify(agentData),
    });
    return response.json();
  }

  async rateAgent(agentId: string, rating: number): Promise<Agent> {
    if (rating < 0 || rating > 5) {
      throw new Error('Rating must be between 0 and 5');
    }

    const response = await this.fetch(`/agents/${encodeURIComponent(agentId)}/rate`, {
      method: 'POST',
      body: JSON.stringify(rating),
    });
    return response.json();
  }

  // Orchestration methods
  async processPrompt(request: ProcessRequest): Promise<ProcessResponse> {
    const response = await this.fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
    return response.json();
  }

  // Utility methods
  async healthCheck(): Promise<{ status: string }> {
    const response = await this.fetch('/api/health');
    return response.json();
  }

  // Private fetch wrapper with error handling
  private async fetch(endpoint: string, options: RequestInit = {}): Promise<Response> {
    const url = `${this.baseURL}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    return response;
  }
}

// Export single instance
export const api = new UnifiedAPIClient();

// Export types for backwards compatibility
export type { Agent };
