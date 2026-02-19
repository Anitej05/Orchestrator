/**
 * Shared hook for workflow execution across all UI modes
 * Handles WebSocket connection, event processing, and state management
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { WS_BASE_URL } from '@/lib/config';
import type {
  WorkflowState,
  WorkflowEvent,
  WorkflowMessage,
  WorkflowTask,
  WorkflowExecutionOptions,
} from '@/lib/workflow-types';

const initialState: WorkflowState = {
  isConnected: false,
  isExecuting: false,
  threadId: null,
  messages: [],
  tasks: {},
  progress: 0,
  currentPhase: '',
  error: null,
  isWaitingForUser: false,
  currentQuestion: null,
  planData: null,
  metadata: {},
};

export function useWorkflowExecution(options: WorkflowExecutionOptions = {}) {
  const {
    autoConnect = false,
    reconnect = true,
    maxReconnectAttempts = 3,
    onConnect,
    onDisconnect,
    onError,
    onComplete,
  } = options;

  const [state, setState] = useState<WorkflowState>(initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);

  // Generate unique message ID
  const generateMessageId = () => `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  // Add message to state
  const addMessage = useCallback((
    type: WorkflowMessage['type'],
    content: string,
    metadata?: WorkflowMessage['metadata']
  ) => {
    const message: WorkflowMessage = {
      id: generateMessageId(),
      type,
      content,
      timestamp: Date.now(),
      metadata,
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, message],
    }));

    return message;
  }, []);

  // Update task status
  const updateTask = useCallback((taskId: string, updates: Partial<WorkflowTask>) => {
    setState(prev => ({
      ...prev,
      tasks: {
        ...prev.tasks,
        [taskId]: {
          ...prev.tasks[taskId],
          ...updates,
        } as WorkflowTask,
      },
    }));
  }, []);

  // Process incoming WebSocket event
  const processEvent = useCallback((event: WorkflowEvent) => {
    const node = event.node || event.type;
    const threadId = event.thread_id || event.data?.thread_id;

    console.log('[WorkflowExecution] Event:', node, event);

    // Update thread ID if present
    if (threadId && !state.threadId) {
      setState(prev => ({ ...prev, threadId }));
    }

    // Update progress if present
    if (event.progress_percentage !== undefined) {
      setState(prev => ({ ...prev, progress: event.progress_percentage }));
    }

    switch (node) {
      case '__start__':
        setState(prev => ({
          ...prev,
          isExecuting: true,
          currentPhase: 'started',
          error: null,
        }));
        addMessage('system', '🚀 Workflow execution started');
        break;

      case 'parse_prompt':
        setState(prev => ({ ...prev, currentPhase: 'parsing' }));
        addMessage('system', '📝 Analyzing your request...');
        break;

      case 'agent_directory_search':
        setState(prev => ({ ...prev, currentPhase: 'searching' }));
        addMessage('system', '🔍 Finding the best agents for your tasks...');
        break;

      case 'rank_agents':
        setState(prev => ({ ...prev, currentPhase: 'ranking' }));
        addMessage('system', '⚖️ Ranking agents by capability and cost...');
        break;

      case 'plan_execution':
        setState(prev => ({ ...prev, currentPhase: 'planning' }));
        if (event.data?.task_agent_pairs) {
          setState(prev => ({
            ...prev,
            planData: event.data,
            metadata: { ...prev.metadata, task_agent_pairs: event.data.task_agent_pairs },
          }));
          addMessage('assistant', '📋 Execution plan ready', {
            task_agent_pairs: event.data.task_agent_pairs,
            planData: event.data,
          });
        }
        break;

      case 'plan_approval_required':
        setState(prev => ({
          ...prev,
          isWaitingForUser: true,
          currentQuestion: 'Please review and approve the execution plan',
        }));
        addMessage('system', '⏸️ Awaiting plan approval');
        break;

      case 'user_input_required':
        setState(prev => ({
          ...prev,
          isWaitingForUser: true,
          currentQuestion: event.message || event.content || 'Input required',
        }));
        addMessage('system', `❓ ${event.message || event.content || 'Please provide input'}`);
        break;

      case 'task_started':
        if (event.task_name) {
          const taskId = event.task_name;
          updateTask(taskId, {
            taskId,
            taskName: event.task_name,
            taskDescription: event.task_description || '',
            agentName: event.agent_name || 'Unknown Agent',
            agentId: event.agent_id || '',
            status: 'running',
            startTime: Date.now(),
          });
          addMessage('assistant', `🚀 Starting: ${event.task_name} with ${event.agent_name}`);
        }
        break;

      case 'task_completed':
        if (event.task_name) {
          const taskId = event.task_name;
          updateTask(taskId, {
            status: 'completed',
            endTime: Date.now(),
            executionTime: event.execution_time,
            cost: event.cost,
            output: event.output,
          });
          addMessage('assistant', `✅ Completed: ${event.task_name} (${event.execution_time || 0}ms)`);
        }
        break;

      case 'task_failed':
        if (event.task_name) {
          const taskId = event.task_name;
          updateTask(taskId, {
            status: 'failed',
            endTime: Date.now(),
            error: event.error,
          });
          addMessage('assistant', `❌ Failed: ${event.task_name} - ${event.error || 'Unknown error'}`);
        }
        break;

      case '__end__':
      case 'final_response':
        setState(prev => ({
          ...prev,
          isExecuting: false,
          currentPhase: 'completed',
          progress: 100,
        }));
        if (event.content || event.message) {
          addMessage('assistant', event.content || event.message || '✨ Workflow completed successfully');
        } else {
          addMessage('system', '✨ Workflow completed successfully');
        }
        if (onComplete) {
          onComplete(event);
        }
        break;

      case '__error__':
        const errorMsg = event.error || event.message || 'An error occurred';
        setState(prev => ({
          ...prev,
          isExecuting: false,
          error: errorMsg,
          currentPhase: 'error',
        }));
        addMessage('error', `⚠️ Error: ${errorMsg}`);
        if (onError) {
          onError(new Error(errorMsg));
        }
        break;

      case 'stream_chunk':
        // Handle streaming text chunks
        if (event.content) {
          setState(prev => ({
            ...prev,
            metadata: {
              ...prev.metadata,
              streamingContent: (prev.metadata.streamingContent || '') + event.content,
            },
          }));
        }
        break;

      default:
        // Handle any other node types with description
        if (event.description || event.content) {
          addMessage('system', event.description || event.content || `Processing ${node}...`);
        }
    }
  }, [state.threadId, addMessage, updateTask, onComplete, onError]);

  // WebSocket connection management
  const connect = useCallback((threadId?: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[WorkflowExecution] Already connected');
      return;
    }

    const url = threadId 
      ? `${WS_BASE_URL}/ws/chat?thread_id=${threadId}`
      : `${WS_BASE_URL}/ws/chat`;

    console.log('[WorkflowExecution] Connecting to:', url);

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WorkflowExecution] Connected');
      setState(prev => ({ ...prev, isConnected: true, error: null }));
      reconnectAttempts.current = 0;
      if (onConnect) onConnect();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        processEvent(data);
      } catch (err) {
        console.error('[WorkflowExecution] Failed to parse message:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('[WorkflowExecution] WebSocket error:', error);
      setState(prev => ({ ...prev, error: 'Connection error' }));
    };

    ws.onclose = (event) => {
      console.log('[WorkflowExecution] Disconnected:', event.code, event.reason);
      setState(prev => ({ ...prev, isConnected: false }));
      wsRef.current = null;
      if (onDisconnect) onDisconnect();

      // Auto-reconnect logic
      if (reconnect && reconnectAttempts.current < maxReconnectAttempts && !event.wasClean) {
        reconnectAttempts.current++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
        console.log(`[WorkflowExecution] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
        reconnectTimer.current = setTimeout(() => connect(threadId), delay);
      }
    };
  }, [reconnect, maxReconnectAttempts, onConnect, onDisconnect, processEvent]);

  // Disconnect
  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState(prev => ({ ...prev, isConnected: false }));
  }, []);

  // Send message
  const sendMessage = useCallback((message: any) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('[WorkflowExecution] Cannot send message: not connected');
      return false;
    }

    try {
      const payload = typeof message === 'string' ? { prompt: message } : message;
      wsRef.current.send(JSON.stringify(payload));
      return true;
    } catch (err) {
      console.error('[WorkflowExecution] Failed to send message:', err);
      return false;
    }
  }, []);

  // Start execution with prompt
  const startExecution = useCallback((prompt: string, files: any[] = [], dryRun: boolean = false) => {
    if (!state.isConnected) {
      connect();
      // Wait for connection then send
      setTimeout(() => {
        sendMessage({ prompt, files, dry_run: dryRun });
      }, 1000);
    } else {
      sendMessage({ prompt, files, dry_run: dryRun });
    }
    
    setState(prev => ({ ...prev, isExecuting: true }));
    addMessage('user', prompt);
  }, [state.isConnected, connect, sendMessage, addMessage]);

  // Continue conversation with user response
  const continueExecution = useCallback((userResponse: string) => {
    if (!state.isConnected) {
      console.error('[WorkflowExecution] Cannot continue: not connected');
      return;
    }

    sendMessage({ user_response: userResponse, thread_id: state.threadId });
    setState(prev => ({ ...prev, isWaitingForUser: false, currentQuestion: null }));
    addMessage('user', userResponse);
  }, [state.isConnected, state.threadId, sendMessage, addMessage]);

  // Approve plan
  const approvePlan = useCallback(() => {
    if (!state.isConnected) {
      console.error('[WorkflowExecution] Cannot approve: not connected');
      return;
    }

    sendMessage({ action: 'approve_plan', thread_id: state.threadId });
    setState(prev => ({ ...prev, isWaitingForUser: false, currentQuestion: null }));
    addMessage('user', '✓ Plan approved');
  }, [state.isConnected, state.threadId, sendMessage, addMessage]);

  // Reset state
  const reset = useCallback(() => {
    disconnect();
    setState(initialState);
  }, [disconnect]);

  // Auto-connect on mount if requested
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect]);

  return {
    // State
    state,
    
    // Connection
    connect,
    disconnect,
    isConnected: state.isConnected,
    
    // Execution
    startExecution,
    continueExecution,
    approvePlan,
    sendMessage,
    
    // State management
    addMessage,
    updateTask,
    reset,
    
    // Computed values
    messages: state.messages,
    tasks: Object.values(state.tasks),
    tasksMap: state.tasks,
    isExecuting: state.isExecuting,
    threadId: state.threadId,
    progress: state.progress,
    currentPhase: state.currentPhase,
    error: state.error,
    isWaitingForUser: state.isWaitingForUser,
    currentQuestion: state.currentQuestion,
    planData: state.planData,
    metadata: state.metadata,
  };
}
