import { useEffect, useRef, useState, useCallback } from 'react';
import type { TaskStatus } from '@/lib/types';

interface TaskResult {
  task_name: string;
  result: any;
  timestamp: string;
}

interface UseTaskExecutionWebSocketOptions {
  threadId?: string;
  autoConnect?: boolean;
  reconnect?: boolean;
  maxReconnectAttempts?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onTaskStart?: (taskName: string, agentName: string) => void;
  onTaskComplete?: (taskName: string, result: any, executionTime: number) => void;
  onTaskFail?: (taskName: string, error: string) => void;
  onFinalResponse?: (data: any) => void;
  onError?: (error: string) => void;
  onMessage?: (data: any) => void;
}

interface UseTaskExecutionWebSocketReturn {
  taskStatuses: Record<string, TaskStatus>;
  taskResults: TaskResult[];
  isConnected: boolean;
  connect: (threadId: string) => void;
  disconnect: () => void;
  clearTasks: () => void;
}

/**
 * Unified WebSocket hook for task execution monitoring.
 * Handles task lifecycle events with auto-reconnect and proper TaskStatus typing.
 */
export function useTaskExecutionWebSocket({
  threadId: initialThreadId,
  autoConnect = false,
  reconnect = true,
  maxReconnectAttempts = 5,
  onConnect,
  onDisconnect,
  onTaskStart,
  onTaskComplete,
  onTaskFail,
  onFinalResponse,
  onError,
  onMessage,
}: UseTaskExecutionWebSocketOptions = {}): UseTaskExecutionWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const currentThreadId = useRef<string | undefined>(initialThreadId);

  const [taskStatuses, setTaskStatuses] = useState<Record<string, TaskStatus>>({});
  const [taskResults, setTaskResults] = useState<TaskResult[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  const WS_BASE_URL = process.env.NEXT_PUBLIC_API_URL
    ? process.env.NEXT_PUBLIC_API_URL.replace(/^http/, 'ws')
    : 'ws://localhost:8000';

  const connect = useCallback((threadId: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('[TaskExecutionWS] Already connected');
      return;
    }

    currentThreadId.current = threadId;
    const url = `${WS_BASE_URL}/ws/chat?thread_id=${threadId}`;

    console.log('[TaskExecutionWS] Connecting to:', url);

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[TaskExecutionWS] Connected successfully');
      setIsConnected(true);
      reconnectAttempts.current = 0;
      if (onConnect) onConnect();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.debug('[TaskExecutionWS] Message received:', data.type);

        // Generic message callback — called for every message
        if (onMessage) onMessage(data);

        switch (data.type) {
          case 'task_started':
            setTaskStatuses(prev => ({
              ...prev,
              [data.task_name]: {
                status: 'running',
                taskName: data.task_name, // ✅ Fixed: Include required field
                agentName: data.agent_name,
                startTime: Date.now(),
              }
            }));
            if (onTaskStart) onTaskStart(data.task_name, data.agent_name);
            break;

          case 'task_completed':
            const executionTime = data.execution_time || 0;
            setTaskStatuses(prev => ({
              ...prev,
              [data.task_name]: {
                status: 'completed',
                taskName: data.task_name, // ✅ Fixed: Include required field
                agentName: data.agent_name,
                executionTime,
                startTime: prev[data.task_name]?.startTime || Date.now(),
              }
            }));
            
            setTaskResults(prev => [...prev, {
              task_name: data.task_name,
              result: data.result,
              timestamp: new Date().toISOString()
            }]);
            
            if (onTaskComplete) onTaskComplete(data.task_name, data.result, executionTime);
            break;

          case 'task_failed':
            setTaskStatuses(prev => ({
              ...prev,
              [data.task_name]: {
                status: 'failed',
                taskName: data.task_name, // ✅ Fixed: Include required field
                agentName: data.agent_name,
                error: data.error,
                startTime: prev[data.task_name]?.startTime,
              }
            }));
            if (onTaskFail) onTaskFail(data.task_name, data.error);
            break;

          case 'final_response':
            if (onFinalResponse) onFinalResponse(data);
            break;

          case 'error':
            if (onError) onError(data.message || 'Execution error');
            break;

          default:
            console.debug('[TaskExecutionWS] Unhandled message type:', data.type);
        }
      } catch (err) {
        console.error('[TaskExecutionWS] Failed to parse message:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('[TaskExecutionWS] WebSocket error:', error);
      setIsConnected(false);
    };

    ws.onclose = (event) => {
      console.log('[TaskExecutionWS] Disconnected:', event.code, event.reason);
      setIsConnected(false);
      wsRef.current = null;
      if (onDisconnect) onDisconnect();

      // Auto-reconnect logic with exponential backoff
      if (reconnect && reconnectAttempts.current < maxReconnectAttempts && !event.wasClean) {
        reconnectAttempts.current++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
        console.log(`[TaskExecutionWS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
        reconnectTimer.current = setTimeout(() => {
          if (currentThreadId.current) {
            connect(currentThreadId.current);
          }
        }, delay);
      }
    };
  }, [WS_BASE_URL, reconnect, maxReconnectAttempts, onConnect, onDisconnect, onTaskStart, onTaskComplete, onTaskFail, onFinalResponse, onError]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'User initiated disconnect');
      wsRef.current = null;
    }
    setIsConnected(false);
    currentThreadId.current = undefined;
  }, []);

  const clearTasks = useCallback(() => {
    setTaskStatuses({});
    setTaskResults([]);
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect && initialThreadId) {
      connect(initialThreadId);
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, initialThreadId]); // Only run on mount/unmount

  return {
    taskStatuses,
    taskResults,
    isConnected,
    connect,
    disconnect,
    clearTasks,
  };
}
