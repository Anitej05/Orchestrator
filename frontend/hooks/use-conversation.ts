// In hooks/use-conversation.ts
import { useState, useCallback } from 'react';
import {
  startConversation as apiStartConversation,
  continueConversation as apiContinueConversation,
  type ProcessResponse,
  type ConversationState,
  type Message,
} from '@/lib/api-client';

interface UseConversationProps {
  onComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
}

export function useConversation({ onComplete, onError }: UseConversationProps = {}) {
  const [state, setState] = useState<ConversationState>({
    thread_id: '',
    status: 'idle',
    messages: [],
    isWaitingForUser: false,
    currentQuestion: '',
  });
  const [isLoading, setIsLoading] = useState(false);

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

  const startConversation = useCallback(async (input: string) => {
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
      }));

      const response = await apiStartConversation(input);
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
    setState({
      thread_id: '',
      status: 'idle',
      messages: [],
      isWaitingForUser: false,
      currentQuestion: '',
    });
  }, []);

  return {
    state,
    isLoading,
    startConversation,
    continueConversation,
    resetConversation,
  };
}
