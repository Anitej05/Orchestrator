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
  error_type?: string;
  error_category?: string;
  status?: string;
  timestamp?: number;
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
      console.debug('WebSocket already connected.');
      return;
    }

    try {
      console.debug(`Initiating WebSocket connection to ${url}`);
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.debug('WebSocket connected successfully');
        setIsConnected(true);

        // Expose WebSocket to window for conversation store to use
        (window as any).__websocket = ws.current;

        // The WebSocket will wait for the first message from the client
        // which will be sent when startConversation or continueConversation is called
        console.debug('WebSocket ready to receive messages');
      };

      ws.current.onmessage = (event) => {
        try {
          const eventData: WebSocketEventData = JSON.parse(event.data);
          console.debug('WebSocket message received:', {
            node: eventData.node,
            thread_id: eventData.thread_id,
            hasData: !!eventData.data,
            dataKeys: eventData.data ? Object.keys(eventData.data).slice(0, 10) : []
          });

          // Handle orchestration stage updates with animations
          if (eventData.node === '__start__') {
            // Preserve existing messages when updating state
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              thread_id: eventData.thread_id,
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
          else if (eventData.node === '__end__') {
            try {
              console.debug('=== RECEIVED __END__ EVENT ===');
              console.debug('Event data:', eventData);
              console.debug('Has data field:', !!eventData.data);
              
              if (!eventData.data) {
                console.error('__end__ event received but no data field!');
                // Set isLoading to false even if there's no data
                useConversationStore.setState({ isLoading: false, status: 'completed' });
                return;
              }
              
              console.debug('Received __end__ event, processing final state...');
              // The backend sends the complete state in the data field
              const finalState: ConversationState = eventData.data;
            console.debug('Final state:', {
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

            console.debug('CANVAS DEBUG: Frontend canvas detection:', {
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
                  console.debug('Filtering HTML content from message:', content.substring(0, 100));
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
              console.debug('No assistant message found in backend messages, adding final_response');
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
              console.debug('Using messages from backend, not adding final_response separately');
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
            console.debug('Setting isLoading to false after __end__ event');
            useConversationStore.setState({ isLoading: false });
            console.debug('Final state updated, isLoading:', useConversationStore.getState().isLoading);
            } catch (endError) {
              console.error('Error processing __end__ event:', endError);
              // Always set isLoading to false even if there's an error
              useConversationStore.setState({ isLoading: false, status: 'error' });
            }
          }
          // Handle intermediate states or errors
          else if (eventData.node === '__user_input_required__') {
            // Check if this is an approval request
            const isApprovalRequest = eventData.data?.approval_required === true;
            
            console.debug('User input required:', {
              isApprovalRequest,
              approval_required: eventData.data?.approval_required,
              estimated_cost: eventData.data?.estimated_cost,
              task_count: eventData.data?.task_count,
              question: eventData.data?.question_for_user
            });
            
            if (isApprovalRequest) {
              // This is a plan approval request - set approval state WITHOUT adding a system message
              // The approval modal will handle the UI
              const currentState = useConversationStore.getState();
              _setConversationState({
                isWaitingForUser: true,
                currentQuestion: eventData.data?.question_for_user,
                approval_required: true,
                estimated_cost: eventData.data?.estimated_cost || 0,
                task_count: eventData.data?.task_count || 0,
                task_plan: eventData.data?.task_plan || currentState.task_plan || [],
                task_agent_pairs: eventData.data?.task_agent_pairs || currentState.task_agent_pairs || [],
                metadata: {
                  ...currentState.metadata,
                  currentStage: 'approval_required',
                  stageMessage: 'Waiting for plan approval...',
                  progress: 50
                }
              });
              console.debug('Set approval state in store (no system message added):', {
                approval_required: true,
                estimated_cost: eventData.data?.estimated_cost || 0,
                task_count: eventData.data?.task_count || 0
              });
            } else {
              // Regular user input required - add as system message
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
                approval_required: false,
                metadata: {
                  ...useConversationStore.getState().metadata,
                  currentStage: 'waiting_for_user',
                  stageMessage: 'Waiting for your response...',
                  progress: 50
                }
              });
            }
            // Set isLoading to false so user can respond
            useConversationStore.setState({ isLoading: false });
          }
          else if (eventData.node === '__error__') {
            const errorType = eventData.error_type || 'UnknownError';
            const errorCategory = eventData.error_category || 'unknown';
            const errorMessage = eventData.error || eventData.message || 'An unknown error occurred';
            
            // Log error for debugging
            console.error('WebSocket error received:', {
              thread_id: eventData.thread_id,
              type: errorType,
              category: errorCategory,
              message: errorMessage
            });
            
            // Create user-friendly error message based on category
            let displayMessage = errorMessage;
            switch (errorCategory) {
              case 'validation':
                displayMessage = `Input validation error: ${errorMessage}`;
                break;
              case 'authorization':
                displayMessage = `Permission denied: ${errorMessage}`;
                break;
              case 'timeout':
                displayMessage = `Request timed out. Your request took too long. Please try with a simpler request.`;
                break;
              case 'database':
                displayMessage = `Database error: Unable to process your request at this time. Please try again later.`;
                break;
              case 'resource_not_found':
                displayMessage = `A required resource was not found: ${errorMessage}`;
                break;
              default:
                displayMessage = `Error: ${errorMessage}`;
            }
            
            const errorSystemMessage: Message = {
              id: Date.now().toString(),
              type: 'system',
              content: displayMessage,
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
                stageMessage: `Error (${errorType}): An error occurred during orchestration`,
                progress: 0,
                lastError: {
                  type: errorType,
                  category: errorCategory,
                  message: errorMessage,
                  timestamp: new Date()
                }
              }
            });
            // Set isLoading to false when there's an error
            useConversationStore.setState({ isLoading: false });
          }
          else if (eventData.node === 'load_history') {
            // History loading event - just log it, messages are added separately
            console.debug('History loading event received');
          }
          else if (eventData.node === 'analyze_request') {
            // Request analysis event
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'analyzing',
                stageMessage: 'Analyzing your request...',
                progress: 15
              }
            });
          }
          else if (eventData.node === 'evaluate_agent_response') {
            // Agent response evaluation
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'evaluating',
                stageMessage: 'Evaluating agent response...',
                progress: 85
              }
            });
          }
          else if (eventData.node === 'ask_user') {
            // User input request (handled similarly to ask_user_input)
            const questionMessage: Message = {
              id: `ask_${Date.now()}`,
              type: 'system',
              content: eventData.data?.question_for_user || 'Please provide additional information',
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
            useConversationStore.setState({ isLoading: false });
          }
          else if (eventData.node === 'save_history') {
            // History saving event - just log it
            console.debug('History saving event received');
          }
          else {
            // Catch-all for any remaining unhandled events
            console.debug(`Unhandled WebSocket event: ${eventData.node}`, eventData);
          }

        } catch (parseError) {
          // Enhanced error handling for parse failures
          const errorMessage = parseError instanceof Error ? parseError.message : String(parseError);
          console.error('Failed to parse WebSocket message:', {
            error: errorMessage,
            rawData: event.data ? event.data.substring(0, 200) : 'empty',
            isJSON: typeof event.data === 'string'
          });
          
          // Only add error message if parsing error (not invalid data from server)
          if (event.data && event.data.length > 0) {
            const errorSystemMessage: Message = {
              id: Date.now().toString(),
              type: 'system',
              content: `Communication Error: Failed to process server response. This may indicate a server issue.`,
              timestamp: new Date()
            };
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: [...currentMessages, errorSystemMessage],
              status: 'error'
            });
          }
          
          // Set isLoading to false on parse error
          useConversationStore.setState({ isLoading: false, status: 'error' });
        }
      };

      ws.current.onclose = (event) => {
        const closeMessage = `WebSocket closed with code ${event.code}${event.reason ? `: ${event.reason}` : ''}`;
        console.log(closeMessage, { code: event.code, reason: event.reason, cleanClose: event.wasClean });
        setIsConnected(false);
        
        // If the store is still in a processing state, mark it as an error
        if (useConversationStore.getState().status === 'processing') {
          console.error('WebSocket disconnected while processing, marking as error');
          
          let disconnectReason = 'Connection closed unexpectedly.';
          // Categorize disconnect reason based on close code
          if (event.code === 1000) {
            disconnectReason = 'Connection closed normally.';
          } else if (event.code === 1001) {
            disconnectReason = 'Server is shutting down. Please try again later.';
          } else if (event.code === 1002 || event.code === 1003) {
            disconnectReason = 'Protocol error. Please refresh the page and try again.';
          } else if (event.code === 1006) {
            disconnectReason = 'Connection lost. Please check your network and try again.';
          } else if (event.code === 1011) {
            disconnectReason = 'Server error. Please try again later.';
          }
          
          // Add disconnection message to chat
          const disconnectMessage: Message = {
            id: `disconnect_${Date.now()}`,
            type: 'system',
            content: `Connection Error: ${disconnectReason}`,
            timestamp: new Date()
          };
          const currentMessages = useConversationStore.getState().messages;
          _setConversationState({ 
            status: 'error',
            messages: [...currentMessages, disconnectMessage]
          });
          useConversationStore.setState({ isLoading: false });
        }
        
        // Attempt automatic reconnection with backoff
        if (!event.wasClean && event.code !== 1000) {
          console.log('Attempting automatic reconnection in 2 seconds...');
          setTimeout(() => {
            if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
              connect();
            }
          }, 2000);
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket connection error:', error);
        setIsConnected(false);
        
        if (useConversationStore.getState().status === 'processing') {
          // Add error message to chat
          const errorMessage: Message = {
            id: `error_${Date.now()}`,
            type: 'system',
            content: 'Connection Error: An error occurred with the WebSocket connection. Please try refreshing the page.',
            timestamp: new Date()
          };
          const currentMessages = useConversationStore.getState().messages;
          _setConversationState({ 
            status: 'error',
            messages: [...currentMessages, errorMessage]
          });
          useConversationStore.setState({ isLoading: false });
        }
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error('Failed to initialize WebSocket:', {
        url,
        error: errorMessage,
        type: error instanceof Error ? error.constructor.name : typeof error
      });
      
      // Update store with connection error
      setIsConnected(false);
      _setConversationState({
        status: 'error',
        metadata: {
          ...useConversationStore.getState().metadata,
          currentStage: 'connection_failed',
          stageMessage: 'Failed to establish WebSocket connection'
        }
      });
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
  return { isConnected };
}
