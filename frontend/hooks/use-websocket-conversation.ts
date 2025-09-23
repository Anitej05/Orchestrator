import { useEffect, useRef, useState, useCallback } from 'react';
import { type Message } from '@/lib/api-client';

interface WebSocketMessage {
  type: 'progress' | 'completion' | 'error' | 'user_input_required';
  progress_percentage?: number;
  requires_user_input?: boolean;
  question_for_user?: string;
  final_response?: string;
  task_agent_pairs?: any[];
  message?: string;
  thread_id?: string;
  error?: string;
}

interface UseWebSocketConversationProps {
  onMessageReceived?: (data: WebSocketMessage) => void;
  onError?: (error: string) => void;
  url?: string;
}

export function useWebSocketConversation({ 
  onMessageReceived, 
  onError,
  url = 'ws://localhost:8000/ws/chat'
}: UseWebSocketConversationProps = {}) {
  const ws = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [waitingForUser, setWaitingForUser] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState<string>();
  const [progress, setProgress] = useState(0);
  const [currentThreadId, setCurrentThreadId] = useState<string>();

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;

    try {
      ws.current = new WebSocket(url);
      
      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      ws.current.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          console.log('WebSocket message received:', data);
          
          onMessageReceived?.(data);
          
          if (data.type === 'progress') {
            // Handle progress updates
            setProgress(data.progress_percentage || 0);
            console.log('Progress:', data.progress_percentage);
          } else if (data.type === 'completion') {
            if (data.requires_user_input) {
              // Orchestrator is asking for user input
              setWaitingForUser(true);
              setCurrentQuestion(data.question_for_user);
              
              const systemMessage: Message = {
                id: Date.now().toString(),
                type: 'system',
                content: data.question_for_user || 'Please provide additional information',
                timestamp: new Date()
              };
              
              setMessages(prev => [...prev, systemMessage]);
            } else {
              // Conversation completed
              setWaitingForUser(false);
              setCurrentQuestion(undefined);
              
              const assistantMessage: Message = {
                id: Date.now().toString(),
                type: 'assistant',
                content: data.final_response || data.message || 'Task completed',
                timestamp: new Date(),
                metadata: { task_agent_pairs: data.task_agent_pairs }
              };
              
              setMessages(prev => [...prev, assistantMessage]);
            }
          } else if (data.type === 'error') {
            const errorMessage = data.error || data.message || 'An error occurred';
            onError?.(errorMessage);
            
            const errorSystemMessage: Message = {
              id: Date.now().toString(),
              type: 'system',
              content: `Error: ${errorMessage}`,
              timestamp: new Date()
            };
            
            setMessages(prev => [...prev, errorSystemMessage]);
          }
        } catch (parseError) {
          console.error('Failed to parse WebSocket message:', parseError);
          onError?.('Failed to parse server message');
        }
      };

      ws.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event);
        setIsConnected(false);
        setWaitingForUser(false);
        setCurrentQuestion(undefined);
        
        // Attempt to reconnect after a delay if not closed intentionally
        if (!event.wasClean) {
          setTimeout(() => {
            console.log('Attempting to reconnect...');
            connect();
          }, 3000);
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.('WebSocket connection error');
        setIsConnected(false);
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      onError?.('Failed to establish WebSocket connection');
    }
  }, [url, onMessageReceived, onError]);

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close(1000, 'User disconnected');
      ws.current = null;
    }
  }, []);

  const sendMessage = useCallback((prompt: string, threadId?: string) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      onError?.('WebSocket is not connected');
      return;
    }

    const message = {
      type: threadId ? 'continue' : 'start',
      prompt: threadId ? undefined : prompt,
      response: threadId ? prompt : undefined,
      thread_id: threadId || `ws-${Date.now()}`
    };

    try {
      ws.current.send(JSON.stringify(message));
      
      if (!threadId) {
        setCurrentThreadId(message.thread_id);
      }

      // Add user message to UI
      const userMessage: Message = {
        id: Date.now().toString(),
        type: 'user',
        content: prompt,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, userMessage]);
      setProgress(0);
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
      onError?.('Failed to send message');
    }
  }, [onError]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setWaitingForUser(false);
    setCurrentQuestion(undefined);
    setProgress(0);
    setCurrentThreadId(undefined);
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    messages,
    isConnected,
    waitingForUser,
    currentQuestion,
    progress,
    currentThreadId,
    sendMessage,
    connect,
    disconnect,
    clearMessages
  };
}
