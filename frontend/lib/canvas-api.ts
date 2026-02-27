// lib/canvas-api.ts
// Canvas REST API helpers for user-triggered actions

import { authFetch } from './auth-fetch';
import { API_BASE_URL } from './config';

/**
 * Fetch all available canvas templates from backend
 */
export async function fetchCanvasTemplates(category?: string, agent?: string) {
  const params = new URLSearchParams();
  if (category) params.append('category', category);
  if (agent) params.append('agent', agent);
  
  const response = await authFetch(
    `${API_BASE_URL}/api/canvas/templates${params.toString() ? '?' + params.toString() : ''}`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to fetch canvas templates: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Get the full canvas registry state for a thread
 * Useful for loading saved conversations with canvas data
 */
export async function fetchCanvasState(threadId: string) {
  const response = await authFetch(
    `${API_BASE_URL}/api/canvas/${threadId}`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to fetch canvas state: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Focus a specific canvas (user-triggered action)
 */
export async function focusCanvas(threadId: string, canvasId: string) {
  const response = await authFetch(
    `${API_BASE_URL}/api/canvas/focus`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        thread_id: threadId,
        canvas_id: canvasId
      })
    }
  );
  
  if (!response.ok) {
    throw new Error(`Failed to focus canvas: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Dismiss a canvas (hide without deleting)
 */
export async function dismissCanvas(threadId: string, canvasId: string) {
  const response = await authFetch(
    `${API_BASE_URL}/api/canvas/dismiss`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        thread_id: threadId,
        canvas_id: canvasId
      })
    }
  );
  
  if (!response.ok) {
    throw new Error(`Failed to dismiss canvas: ${response.statusText}`);
  }
  
  return await response.json();
}
