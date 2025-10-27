import { useEffect, useRef, useState, useCallback } from 'react';
import { useConversationStore } from '@/lib/conversation-store';
import type { ConversationState, Message } from '@/lib/types';

// This interface defines the structure of messages coming from the WebSocket
interface WebSocketEventData {
  node: string; // e.g., "__start__", "parse_prompt", "__end__", "__error__"
  thread_id: string;
  data?: any;
  message?: string;
  error?: string;
  // ... other potential fields
}

interface UseWebSocketManagerProps {
  url?: string;
}

/**
 * Manages the WebSocket connection and updates the central Zustand store.
 * This hook does not manage its own state for messages, etc. It is a pure
 * connection and data-flow manager.
 */
export function useWebSocketManager({
  url = 'ws://localhost:8000/ws/chat',
}: UseWebSocketManagerProps = {}) {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const { _setConversationState } = useConversationStore((state: any) => state.actions);

  const connect = useCallback(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected.');
      return;
    }

    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);

        // Expose WebSocket to window for conversation store to use
        (window as any).__websocket = ws.current;

        // The WebSocket will wait for the first message from the client
        // which will be sent when startConversation or continueConversation is called
        console.log('WebSocket ready to receive messages');
      };

      ws.current.onmessage = (event) => {
        try {
          const eventData: WebSocketEventData = JSON.parse(event.data);
          console.log('WebSocket message received:', eventData);

          // Handle orchestration stage updates with animations
          if (eventData.node === '__start__') {
            // Preserve existing messages when updating state
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              status: 'processing',
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'initializing',
                stageMessage: 'Starting agent orchestration...'
              }
            });
          }
          else if (eventData.node === 'parse_prompt') {
            // Preserve existing messages when updating state
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'parsing',
                stageMessage: 'Analyzing your request...',
                progress: 10
              }
            });
          }
          else if (eventData.node === 'agent_directory_search') {
            // Preserve existing messages when updating state
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'searching',
                stageMessage: 'Searching agent directory...',
                progress: 25
              }
            });
          }
          else if (eventData.node === 'rank_agents') {
            // Preserve existing messages when updating.
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'ranking',
                stageMessage: 'Ranking agents for your tasks...',
                progress: 40
              }
            });
          }
          else if (eventData.node === 'plan_execution') {
            // Preserve existing messages when updating state
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'planning',
                stageMessage: 'Creating execution plan...',
                progress: 55
              }
            });
          }
          else if (eventData.node === 'validate_plan_for_execution') {
            // Preserve existing messages when updating state
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'validating',
                stageMessage: 'Validating execution plan...',
                progress: 70
              }
            });
          }
          else if (eventData.node === 'execute_batch') {
            // Preserve existing messages when updating state
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'executing',
                stageMessage: 'Executing tasks...',
                progress: 85
              }
            });
          }
          else if (eventData.node === 'aggregate_responses') {
            // Preserve existing messages when updating state
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'aggregating',
                stageMessage: 'Aggregating results...',
                progress: 95
              }
            });
          }
          // The '__end__' node now contains the final, complete state.
          // We use this as the single source of truth to update our store.
          else if (eventData.node === '__end__' && eventData.data) {
            console.log('Received __end__ event, processing final state...');
            // The backend sends the complete state in the data field
            const finalState: ConversationState = eventData.data;
            console.log('Final state:', {
              hasMessages: !!finalState.messages,
              messagesCount: finalState.messages?.length || 0,
              hasFinalResponse: !!finalState.final_response,
              finalResponseLength: finalState.final_response?.length || 0,
              hasCanvas: finalState.has_canvas
            });

            // Use backend messages as the single source of truth
            // The backend has the complete, authoritative message history
            const backendMessages = (finalState.messages || []).filter(msg => {
              // Keep all non-assistant messages
              if (msg.type !== 'assistant') return true;
              // Keep assistant messages that have content
              return msg.content && msg.content.trim() !== '';
            });

            // Don't merge with frontend messages - backend is the source of truth
            // This prevents duplicates from optimistic UI updates
            let finalMessages = backendMessages;

            // Filter out any empty assistant messages again to prevent empty bubbles
            finalMessages = finalMessages.filter(msg => {
              // Keep all non-assistant messages
              if (msg.type !== 'assistant') return true;
              // Keep assistant messages that have content
              return msg.content && msg.content.trim() !== '';
            });

            // Check if the final response contains HTML that should be in canvas
            const isHtmlContent = finalState.final_response && (
              finalState.final_response.includes('<!DOCTYPE html>') ||
              finalState.final_response.includes('<html') ||
              finalState.final_response.includes('<button') ||
              finalState.final_response.includes('<script>')
            );

            console.log('CANVAS DEBUG: Frontend canvas detection:', {
              hasCanvas: finalState.has_canvas,
              canvasType: finalState.canvas_type,
              canvasContentLength: finalState.canvas_content?.length,
              finalResponseLength: finalState.final_response?.length,
              isHtmlContent: isHtmlContent
            });

            // Filter out HTML content from messages (it should be in canvas, not chat)
            finalMessages = finalMessages.filter(msg => {
              if (msg.type === 'assistant' && msg.content) {
                const content = msg.content;
                const hasHtml = (
                  content.includes('<!DOCTYPE html>') ||
                  content.includes('<html') ||
                  content.includes('<button') ||
                  content.includes('<script>')
                );
                if (hasHtml) {
                  console.log('Filtering HTML content from message:', content.substring(0, 100));
                }
                return !hasHtml;
              }
              return true;
            });

            // If canvas exists, attach canvas metadata to the last assistant message
            if (finalState.has_canvas && finalState.canvas_content && finalMessages.length > 0) {
              const lastMessage = finalMessages[finalMessages.length - 1];
              if (lastMessage.type === 'assistant') {
                lastMessage.canvas_content = finalState.canvas_content;
                lastMessage.canvas_type = finalState.canvas_type;
                lastMessage.has_canvas = true;
              }
            }

            // Check if we have any assistant messages in finalMessages
            const hasAssistantMessage = finalMessages.some(msg => msg.type === 'assistant');

            // If we don't have any assistant messages and we have a final_response, add it
            if (!hasAssistantMessage && finalState.final_response && finalState.final_response.trim() !== '') {
              console.log('No assistant message found in backend messages, adding final_response');
              const assistantMessage: Message = {
                id: Date.now().toString(),
                type: 'assistant',
                content: finalState.final_response,
                timestamp: new Date(),
                // Attach canvas metadata if present
                canvas_content: finalState.canvas_content,
                canvas_type: finalState.canvas_type,
                has_canvas: finalState.has_canvas
              };
              finalMessages = [...finalMessages, assistantMessage];
            } else {
              console.log('Using messages from backend, not adding final_response separately');
            }

            // Additional filtering to ensure no HTML content appears in chat messages
            finalMessages = finalMessages.map(msg => {
              if (msg.type === 'assistant' && msg.content) {
                // Check if the message content contains HTML that should be in canvas
                const hasHtmlTags = (
                  msg.content.includes('<!DOCTYPE html>') ||
                  msg.content.includes('<html') ||
                  msg.content.includes('<button') ||
                  msg.content.includes('<script>') ||
                  msg.content.includes('<div') ||
                  msg.content.includes('<span') ||
                  msg.content.includes('<p>') ||
                  msg.content.includes('<h1>') ||
                  msg.content.includes('<h2>') ||
                  msg.content.includes('<h3>') ||
                  msg.content.includes('<style>') ||
                  msg.content.includes('<head>')
                );

                if (hasHtmlTags) {
                  // If this message contains HTML, replace it with a clean explanation
                  return {
                    ...msg,
                    content: "I've created an interactive visualization for your request. You can view it in the Canvas tab.",
                  };
                }
              }
              return msg;
            });

            // Preserve existing data that might be missing from finalState
            const currentState = useConversationStore.getState();

            _setConversationState({
              ...finalState,
              messages: finalMessages,
              status: 'completed',
              metadata: {
                ...currentState.metadata,
                ...finalState.metadata,
                currentStage: 'completed',
                stageMessage: 'Orchestration completed successfully!',
                progress: 100
              },
              // Ensure we're properly updating all fields by preserving existing data when finalState doesn't have it
              task_agent_pairs: finalState.task_agent_pairs || currentState.task_agent_pairs || [],
              plan: finalState.plan || currentState.plan || [],
              uploaded_files: finalState.uploaded_files || currentState.uploaded_files || [],
              // Handle canvas data
              canvas_content: finalState.canvas_content !== undefined ? finalState.canvas_content : currentState.canvas_content,
              canvas_type: finalState.canvas_type !== undefined ? finalState.canvas_type : currentState.canvas_type,
              has_canvas: finalState.has_canvas !== undefined ? finalState.has_canvas : currentState.has_canvas
            });
            // Explicitly set isLoading to false in the store
            console.log('Setting isLoading to false after __end__ event');
            useConversationStore.setState({ isLoading: false });
            console.log('Final state updated, isLoading:', useConversationStore.getState().isLoading);
          }
          // Handle intermediate states or errors
          else if (eventData.node === '__user_input_required__') {
            // Add the question as a system message to appear below the user's message
            const questionMessage: Message = {
              id: Date.now().toString(),
              type: 'system',
              content: eventData.data?.question_for_user || 'Additional information required.',
              timestamp: new Date()
            };

            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: [...currentMessages, questionMessage],
              isWaitingForUser: true,
              currentQuestion: eventData.data?.question_for_user,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'waiting_for_user',
                stageMessage: 'Waiting for your response...',
                progress: 50
              }
            });
            // Set isLoading to false so user can input their response
            useConversationStore.setState({ isLoading: false });
          }
          else if (eventData.node === '__error__') {
            const errorMessage = eventData.error || 'An unknown WebSocket error occurred';
            const errorSystemMessage: Message = {
              id: Date.now().toString(),
              type: 'system',
              content: `Error: ${errorMessage}`,
              timestamp: new Date()
            };
            // Add error message to the existing messages in the store
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: [...currentMessages, errorSystemMessage],
              status: 'error',
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'error',
                stageMessage: 'An error occurred during orchestration',
                progress: 0
              }
            });
            // Set isLoading to false when there's an error
            useConversationStore.setState({ isLoading: false });
          }

        } catch (parseError) {
          console.error('Failed to parse WebSocket message:', parseError);
        }
      };

      ws.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        // If the store is still in a processing state, mark it as an error
        if (useConversationStore.getState().status === 'processing') {
          console.error('WebSocket disconnected while processing, marking as error');
          _setConversationState({ status: 'error' });
          // Set isLoading to false when connection closes during processing
          useConversationStore.setState({ isLoading: false });
        }
        // Optional: Implement a reconnect strategy
        // For now, let's try to reconnect automatically
        setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...');
          connect();
        }, 2000);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        if (useConversationStore.getState().status === 'processing') {
          _setConversationState({ status: 'error' });
          // Set isLoading to false when there's a connection error during processing
          useConversationStore.setState({ isLoading: false });
        }
      };

    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }, [url, _setConversationState]);

  const disconnect = useCallback(() => {
    if (ws.current) {
      // Only close if the connection is still open
      if (ws.current.readyState === WebSocket.OPEN) {
        ws.current.close(1000, 'User initiated disconnect');
      }
      ws.current = null;
      (window as any).__websocket = null;
    }
  }, []);

  // Effect to manage the connection lifecycle
  useEffect(() => {
    // Connect on mount
    connect();
    // Disconnect on unmount
    return () => {
      // Add a small delay before disconnecting to allow final messages to be sent
      setTimeout(() => {
        disconnect();
      }, 1000);
    };
  }, [connect, disconnect]);

  // This hook now only returns the connection status.
  // All state is accessed via `useConversationStore`.
  return { isConnected };
}
