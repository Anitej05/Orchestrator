/**
 * Frontend Configuration
 * 
 * Single source of truth for all runtime configuration.
 * All API and WebSocket URLs should derive from this file.
 */

// Get API base URL from environment variable or default to localhost for development
export const API_BASE_URL = 
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Derive WebSocket URL from API URL
// http://... -> ws://..., https://... -> wss://...
export const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

/**
 * Helper to construct full API URLs
 * @param path - API path (with or without leading slash)
 * @returns Full API URL
 * 
 * @example
 * getApiUrl('/conversations') // 'http://localhost:8000/conversations'
 * getApiUrl('conversations')  // 'http://localhost:8000/conversations'
 */
export function getApiUrl(path: string): string {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
}

/**
 * Helper to construct full WebSocket URLs
 * @param path - WebSocket path (with or without leading slash)
 * @returns Full WebSocket URL
 * 
 * @example
 * getWsUrl('/ws/chat') // 'ws://localhost:8000/ws/chat'
 * getWsUrl('ws/chat')  // 'ws://localhost:8000/ws/chat'
 */
export function getWsUrl(path: string): string {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${WS_BASE_URL}${cleanPath}`;
}

// Export individual URLs for common use cases
export const CHAT_WS_URL = getWsUrl('/ws/chat');
export const WORKFLOW_WS_URL = getWsUrl('/ws/workflow');
export const ORCHESTRATOR_WS_URL = getWsUrl('/ws/orchestrator');
