import { useState, useCallback, useEffect, useRef } from 'react';
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

  // Keep refs to the latest onError/onComplete to avoid recreating callbacks
  // when parent components pass inline functions (which changes their identity each render).
  const _onErrorRef = useRef(onError);
  const _onCompleteRef = useRef(onComplete);

  // Keep refs up to date
  useEffect(() => { _onErrorRef.current = onError }, [onError]);
  useEffect(() => { _onCompleteRef.current = onComplete }, [onComplete]);

  useEffect(() => {
    if (state.thread_id) {
      localStorage.setItem('thread_id', state.thread_id);
    } else {
      localStorage.removeItem('thread_id');
    }
  }, [state.thread_id]);

  // Initial conversation auto-load moved below (after loadConversation declaration)

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
        task_agent_pairs: task_agent_pairs || prevState.task_agent_pairs,
        final_response: final_response || prevState.final_response,
      };
    });

    if (!pending_user_input) {
      try {
        _onCompleteRef.current?.(response);
      } catch (e) {
        // swallow
      }
    }
  }, []);
  // Intentionally no external deps; onComplete is accessed via ref above.

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
      try { _onErrorRef.current?.(errorMessage); } catch (e) {}
    } finally {
      setIsLoading(false);
    }
  }, [handleApiResponse, state.thread_id]);
  
  const continueConversation = useCallback(async (input: string) => {
    if (!state.thread_id) {
      try { _onErrorRef.current?.('Cannot continue conversation without a thread ID.'); } catch (e) {}
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
      try { _onErrorRef.current?.(errorMessage); } catch (e) {}
    } finally {
      setIsLoading(false);
    }
  }, [state.thread_id, handleApiResponse]);

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
    console.debug('[useConversation] loadConversation called for', thread_id);
    try {
      // Load conversation history from API
      const response = await fetch(`http://localhost:8000/api/conversations/${thread_id}`);
      console.debug('[useConversation] fetch complete', response.status, response.statusText);
      if (!response.ok) {
        const errText = await response.text().catch(() => '<no-body>');
        console.error('[useConversation] fetch failed', response.status, response.statusText, errText);
        throw new Error(`Failed to load conversation: ${response.statusText}`);
      }

      // Parse the new response format from backend
      const conversationData = await response.json();
      console.debug('[useConversation] received conversation data', conversationData);

      if (!conversationData || typeof conversationData !== 'object') {
        console.error('[useConversation] invalid conversation data structure');
        throw new Error('Invalid conversation data structure');
      }

      const rawMessages = Array.isArray(conversationData.messages) ? conversationData.messages : [];
      console.debug('[useConversation] found messages array with length:', rawMessages.length);

      // Process the messages from the API response
      const messages: Message[] = rawMessages.map((msgData: any, index: number) => {
        const messageType = msgData.type || 'system';
        const id = msgData.id || `${Date.now()}-${index}`;
        const content = msgData.content || '';
        
        // Handle timestamp conversion - it might be a number (Unix timestamp) or a string
        let timestamp = new Date();
        if (msgData.timestamp) {
          if (typeof msgData.timestamp === 'number') {
            // Unix timestamp - convert to milliseconds if needed
            timestamp = new Date(msgData.timestamp * 1000);
          } else if (typeof msgData.timestamp === 'string') {
            // String timestamp
            timestamp = new Date(msgData.timestamp);
          } else {
            // Already a Date object or something else
            timestamp = new Date(msgData.timestamp);
          }
        } else {
          timestamp = new Date();
        }
        
        const metadata = msgData.metadata || {};
        const attachments = msgData.attachments || undefined;

        return {
          id,
          type: messageType,
          content,
          timestamp,
          metadata,
          ...(attachments ? { attachments } : {})
        };
      });

      console.debug('[useConversation] mapped messages length', messages.length);
      
      // Update state with all the loaded data
      setState({
        thread_id: conversationData.thread_id || thread_id,
        status: 'completed',
        messages,
        isWaitingForUser: false,
        currentQuestion: '',
        task_agent_pairs: conversationData.task_agent_pairs || [],
        final_response: conversationData.final_response || undefined,
        metadata: conversationData.metadata || {},
        uploaded_files: conversationData.uploaded_files || []
      });
      
      console.debug('[useConversation] state updated on loadConversation', {
        thread_id: conversationData.thread_id || thread_id,
        messagesCount: messages.length,
        taskAgentPairsCount: conversationData.task_agent_pairs?.length || 0,
        hasFinalResponse: Boolean(conversationData.final_response)
      });
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to load conversation';
      console.error('[useConversation] loadConversation error', error);
      try { _onErrorRef.current?.(errorMessage); } catch (e) {}
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Prevent double-loading in React StrictMode during development by ensuring the
  // mount-load runs only once per full mount lifecycle.
  const _didMountRef = useRef(false);
  // Keep a ref to the loadConversation function so we can call it from an effect
  // with an empty dependency array. This avoids re-running the effect when the
  // callback identity changes (HMR/dev) and is safe because loadConversation is stable.
  const loadConversationRef = useRef(loadConversation);
  useEffect(() => { loadConversationRef.current = loadConversation }, [loadConversation]);

  // Automatically load conversation when component mounts if there's a saved thread_id
  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (_didMountRef.current) return; // already ran
    _didMountRef.current = true;

    const tryLoad = async () => {
      const savedThreadId = localStorage.getItem('thread_id');
      if (!savedThreadId) return;
      try {
        // Attempt to load the saved conversation on initial page load
        await loadConversationRef.current(savedThreadId);
      } catch (err) {
        // Swallow errors here - we don't want a failed load to crash the UI.
        // Keep the thread_id in localStorage so user can retry or continue later.
        console.warn('Failed to load saved conversation on mount', err);
      }
    };

    tryLoad();
  }, []);

  return {
    state,
    isLoading,
    startConversation,
    continueConversation,
    resetConversation,
    loadConversation,
  };
}
