import { useState, useCallback, useEffect } from 'react';
import {
  startConversation as apiStartConversation,
  continueConversation as apiContinueConversation,
  uploadFiles as apiUploadFiles,
} from '@/lib/api-client';
import type { ProcessResponse, ConversationState, Message, Attachment, FileObject } from '@/lib/types';

// Helper function to read file as data URL
const readFileAsDataURL = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

interface UseConversationProps {
  onComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
}

export function useConversation({ onComplete, onError }: UseConversationProps = {}) {
  const [state, setState] = useState<ConversationState>(() => {
    const savedThreadId = typeof window !== 'undefined' ? localStorage.getItem('thread_id') : null;
    return {
      thread_id: savedThreadId || '',
      status: 'idle',
      messages: [],
      isWaitingForUser: false,
      currentQuestion: '',
    };
  });
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (state.thread_id) {
      localStorage.setItem('thread_id', state.thread_id);
    } else {
      localStorage.removeItem('thread_id');
    }
  }, [state.thread_id]);

  // Automatically load conversation when component mounts if there's a saved thread_id
  useEffect(() => {
    const savedThreadId = typeof window !== 'undefined' ? localStorage.getItem('thread_id') : null;
    if (savedThreadId) {
      loadConversation(savedThreadId);
    }
  }, []);

  const handleApiResponse = useCallback((response: ProcessResponse) => {
    const {
      thread_id,
      pending_user_input,
      question_for_user,
      final_response,
      message,
      task_agent_pairs,
    } = response;

    setState(prevState => {
      const newMessages: Message[] = [...prevState.messages];
      if (pending_user_input) {
        newMessages.push({
          id: Date.now().toString(),
          type: 'system',
          content: question_for_user || 'Input required',
          timestamp: new Date(),
        });
      } else {
        newMessages.push({
          id: Date.now().toString(),
          type: 'assistant',
          content: final_response || message,
          timestamp: new Date(),
          metadata: { task_agent_pairs },
        });
      }

      return {
        ...prevState,
        thread_id,
        status: pending_user_input ? 'pending_user_input' : 'completed',
        messages: newMessages,
        isWaitingForUser: pending_user_input,
        currentQuestion: question_for_user || undefined,
      };
    });

    if (!pending_user_input) {
      onComplete?.(response);
    }
  }, [onComplete]);

  const startConversation = useCallback(async (input: string, files: File[] = []) => {
    setIsLoading(true);
    try {
      let uploadedFiles: FileObject[] = [];
      if (files.length > 0) {
        uploadedFiles = await apiUploadFiles(files);
      }

      const attachments: Attachment[] = await Promise.all(
        files.map(async (file) => {
          let content = '';
          if (file.type.startsWith('image/')) {
            content = await readFileAsDataURL(file);
          }
          return { name: file.name, type: file.type, content };
        })
      );

      const userMessage: Message = {
        id: Date.now().toString(),
        type: 'user',
        content: input,
        timestamp: new Date(),
        attachments: attachments.length > 0 ? attachments : undefined,
      };
      
      setState(prevState => ({
        ...prevState,
        messages: [...prevState.messages, userMessage],
        status: 'processing',
      }));

      const response = await apiStartConversation(input, state.thread_id, uploadedFiles);
      handleApiResponse(response);
    } catch (error: any) {
      const errorMessage = error.message || 'An unknown error occurred';
      setState(prevState => ({ ...prevState, status: 'error' }));
      onError?.(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [handleApiResponse, onError]);
  
  const continueConversation = useCallback(async (input: string) => {
    if (!state.thread_id) {
      onError?.('Cannot continue conversation without a thread ID.');
      return;
    }
    setIsLoading(true);
    try {
      const userMessage: Message = {
        id: Date.now().toString(),
        type: 'user',
        content: input,
        timestamp: new Date(),
      };
      setState(prevState => ({
        ...prevState,
        messages: [...prevState.messages, userMessage],
        status: 'processing',
        isWaitingForUser: false,
        currentQuestion: undefined,
      }));

      const response = await apiContinueConversation(input, state.thread_id);
      handleApiResponse(response);
    } catch (error: any) {
      const errorMessage = error.message || 'An unknown error occurred';
      setState(prevState => ({ ...prevState, status: 'error' }));
      onError?.(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [state.thread_id, handleApiResponse, onError]);

  const resetConversation = useCallback(() => {
    // Create a new thread_id by generating a fresh UUID
    setState({
      thread_id: '',
      status: 'idle',
      messages: [],
      isWaitingForUser: false,
      currentQuestion: '',
    });
  }, []);

  const loadConversation = useCallback(async (thread_id: string) => {
    setIsLoading(true);
    try {
      // Load conversation history from API
      const response = await fetch(`http://localhost:8000/api/conversations/${thread_id}`);
      if (!response.ok) {
        throw new Error(`Failed to load conversation: ${response.statusText}`);
      }
      const historyData = await response.json();

      // Process the messages from the API response
      const messages: Message[] = historyData.map((msgData: any, index: number) => {
        // Determine the message type based on the LangChain message type
        let messageType: 'user' | 'assistant' | 'system' = 'system';
        if (typeof msgData.type === 'string') {
          if (msgData.type.toLowerCase().includes('human') || msgData.type.toLowerCase() === 'user') {
            messageType = 'user';
          } else if (msgData.type.toLowerCase().includes('ai') || msgData.type.toLowerCase() === 'assistant') {
            messageType = 'assistant';
          } else {
            messageType = 'system';
          }
        } else if (msgData.type === 'human') {
          messageType = 'user';
        } else if (msgData.type === 'ai') {
          messageType = 'assistant';
        }

        return {
          id: msgData.id || (Date.now() + index).toString(),
          type: messageType,
          content: msgData.content || msgData.data?.content || '',
          timestamp: msgData.timestamp ? new Date(msgData.timestamp) : new Date(),
          metadata: msgData.metadata || msgData.data?.metadata || {},
        };
      });

      setState({
        thread_id,
        status: 'completed',
        messages,
        isWaitingForUser: false,
        currentQuestion: '',
      });
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to load conversation';
      onError?.(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [onError]);

  return {
    state,
    isLoading,
    startConversation,
    continueConversation,
    resetConversation,
    loadConversation,
  };
}
