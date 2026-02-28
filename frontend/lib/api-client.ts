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
