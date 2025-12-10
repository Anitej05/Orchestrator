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
  // Task status tracking fields
  task_name?: string;
  task_description?: string;
  agent_name?: string;
  execution_time?: number;
  cost?: number;
  result?: any;
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
      console.log('🔌 Initiating WebSocket connection to', url);
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log('✅ WebSocket connected successfully to', url);
        setIsConnected(true);

        // Expose WebSocket to window for conversation store to use
        (window as any).__websocket = ws.current;

        // The WebSocket will wait for the first message from the client
        // which will be sent when startConversation or continueConversation is called
        console.debug('WebSocket ready to receive messages');
      };

      ws.current.onerror = (error) => {
        console.warn('❌ WebSocket connection error:', error);
        console.warn('Failed to connect to:', url);
        console.warn('Make sure the backend is running on http://localhost:8000');
        setIsConnected(false);
      };

      ws.current.onclose = (event) => {
        console.warn('⚠️ WebSocket connection closed:', {
          code: event.code,
          reason: event.reason || 'No reason provided',
          wasClean: event.wasClean
        });
        setIsConnected(false);
        (window as any).__websocket = null;
        
        // Always attempt to reconnect after a delay (for new conversations)
        // Only skip reconnect if this is a user-initiated disconnect (code 1000 with clean close)
        const isUserDisconnect = event.wasClean && event.code === 1000 && event.reason === 'User initiated disconnect';
        
        if (!isUserDisconnect) {
          console.log('🔄 Will attempt to reconnect in 2 seconds...');
          setTimeout(() => {
            console.log('🔄 Attempting to reconnect WebSocket...');
            connect();
          }, 2000);
        }
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
          // Use stage information from backend if available, otherwise fall back to node-based mapping
          const currentMessages = useConversationStore.getState().messages;
          const currentState = useConversationStore.getState();
          
          // Extract stage information from event data (sent by backend)
          const backendStage = eventData.data?.current_stage;
          const backendMessage = eventData.data?.stage_message;
          const backendProgress = eventData.data?.progress_percentage;
          
          if (eventData.node === '__start__') {
            _setConversationState({
              thread_id: eventData.thread_id,
              status: 'processing',
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: backendStage || 'initializing',
                stageMessage: backendMessage || 'Starting agent orchestration...',
                progress: backendProgress || 0
              }
            });
          }
          else if (eventData.node === 'analyze_request') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'analyzing',
                stageMessage: backendMessage || 'Analyzing your request...',
                progress: backendProgress || 10
              }
            });
          }
          else if (eventData.node === 'parse_prompt') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'parsing',
                stageMessage: backendMessage || 'Breaking down your request into tasks...',
                progress: backendProgress || 20
              }
            });
          }
          else if (eventData.node === 'agent_directory_search') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'searching',
                stageMessage: backendMessage || 'Searching for capable agents (REST & MCP)...',
                progress: backendProgress || 35
              }
            });
          }
          else if (eventData.node === 'rank_agents') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'ranking',
                stageMessage: backendMessage || 'Ranking and selecting best agents...',
                progress: backendProgress || 50
              }
            });
          }
          else if (eventData.node === 'plan_execution') {
            const updates: any = {
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'planning',
                stageMessage: backendMessage || 'Creating execution plan...',
                progress: backendProgress || 60
              }
            };
            
            // Handle node-specific data
            if (eventData.data?.task_plan) {
              updates.plan = eventData.data.task_plan;
              updates.task_plan = eventData.data.task_plan;
            }
            if (eventData.data?.task_agent_pairs) {
              updates.task_agent_pairs = eventData.data.task_agent_pairs;
            }
            
            _setConversationState(updates);
            console.log('Plan created with', updates.plan?.length || 0, 'batches');
          }
          else if (eventData.node === 'validate_plan_for_execution') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'validating',
                stageMessage: backendMessage || 'Validating execution plan...',
                progress: backendProgress || 70
              }
            });
          }
          else if (eventData.node === 'execute_batch') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'executing',
                stageMessage: backendMessage || 'Executing tasks with agents...',
                progress: backendProgress || 80
              }
            });
          }
          else if (eventData.node === 'load_history') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'loading',
                stageMessage: backendMessage || 'Loading conversation history...',
                progress: backendProgress || 5
              }
            });
          }
          else if (eventData.node === 'preprocess_files') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'analyzing',
                stageMessage: backendMessage || 'Processing uploaded files...',
                progress: backendProgress || 15
              }
            });
          }
          else if (eventData.node === 'task_started') {
            // Real-time task execution tracking - task started
            const taskName = eventData.data?.task_name || eventData.task_name;
            const agentName = eventData.data?.agent_name || eventData.agent_name;
            
            console.log('📨 WebSocket task_started event received:', { taskName, agentName, eventData });
            
            if (taskName) {
              const currentState = useConversationStore.getState();
              const updatedTaskStatuses = {
                ...currentState.task_statuses,
                [taskName]: {
                  status: 'running' as const,
                  taskName,
                  agentName,
                  taskDescription: eventData.data?.task_description || eventData.task_description,
                  startedAt: new Date(),
                }
              };
              
              _setConversationState({
                task_statuses: updatedTaskStatuses,
                current_executing_task: taskName,
              });
              
              console.log('✅ Updated task_statuses in store:', updatedTaskStatuses);
              console.debug('Task started:', { taskName, agentName });
            }
          }
          else if (eventData.node === 'task_completed') {
            // Real-time task execution tracking - task completed
            const taskName = eventData.data?.task_name || eventData.task_name;
            const executionTime = eventData.data?.execution_time || eventData.execution_time;
            const agentName = eventData.data?.agent_name || eventData.agent_name;
            const result = eventData.data?.result || eventData.result;
            const cost = eventData.data?.cost || eventData.cost;
            
            console.log('📨 WebSocket task_completed event received:', { taskName, executionTime, agentName });
            
            if (taskName) {
              const currentState = useConversationStore.getState();
              const existingStatus = currentState.task_statuses?.[taskName];
              
              const updatedTaskStatuses = {
                ...currentState.task_statuses,
                [taskName]: {
                  ...existingStatus,
                  status: 'completed' as const,
                  taskName,
                  agentName,
                  completedAt: new Date(),
                  executionTime,
                  cost,
                  result,
                }
              };
              
              _setConversationState({
                task_statuses: updatedTaskStatuses,
                current_executing_task: null,
              });
              
              console.log('✅ Updated task_statuses in store:', updatedTaskStatuses);
              console.debug('Task completed:', { taskName, executionTime, agentName });
            }
          }
          else if (eventData.node === 'task_failed') {
            // Real-time task execution tracking - task failed
            const taskName = eventData.data?.task_name || eventData.task_name;
            const executionTime = eventData.data?.execution_time || eventData.execution_time;
            const error = eventData.data?.error || eventData.error;
            const agentName = eventData.data?.agent_name || eventData.agent_name;
            
            if (taskName) {
              const currentState = useConversationStore.getState();
              const existingStatus = currentState.task_statuses?.[taskName];
              
              const updatedTaskStatuses = {
                ...currentState.task_statuses,
                [taskName]: {
                  ...existingStatus,
                  status: 'failed' as const,
                  taskName,
                  agentName,
                  completedAt: new Date(),
                  executionTime,
                  error,
                }
              };
              
              _setConversationState({
                task_statuses: updatedTaskStatuses,
                current_executing_task: null,
              });
              
              console.warn('Task failed:', { taskName, error, executionTime });
            }
          }
          else if (eventData.node === 'aggregate_responses' || eventData.node === 'generate_final_response') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'aggregating',
                stageMessage: backendMessage || 'Generating final response...',
                progress: backendProgress || 95
              }
            });
          }
          else if (eventData.node === 'evaluate_agent_response') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'evaluating',
                stageMessage: backendMessage || 'Evaluating agent responses...',
                progress: backendProgress || 90
              }
            });
          }
          else if (eventData.node === 'workflow_complete') {
            // Final completion event - set progress to 100%
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'completed',
                stageMessage: 'Workflow completed successfully',
                progress: 100
              }
            });
          }
          else if (eventData.node === 'ask_user' || eventData.node === '__user_input_required__') {
            const currentMessages = useConversationStore.getState().messages;
            const question = eventData.data?.question_for_user || eventData.data?.question || 'Please provide additional information';
            
            // Check if this is an approval request (has needs_approval or approval_required flag)
            const isApprovalRequest = eventData.data?.needs_approval === true || eventData.data?.approval_required === true;
            
            if (isApprovalRequest) {
              // This is a plan approval request - set approval state WITHOUT adding message
              const currentState = useConversationStore.getState();
              
              _setConversationState({
                // Don't add the approval message to chat - user can see the plan in the Plan tab
                isWaitingForUser: true,
                currentQuestion: question,
                approval_required: true,
                estimated_cost: eventData.data?.estimated_cost || 0,
                task_count: eventData.data?.task_count || 0,
                task_plan: eventData.data?.task_plan || currentState.task_plan || [],
                task_agent_pairs: eventData.data?.task_agent_pairs || currentState.task_agent_pairs || [],
                isLoading: false,
                metadata: {
                  ...currentState.metadata,
                  currentStage: 'validating',
                  stageMessage: 'Waiting for plan approval...',
                  progress: 50
                }
              });
            } else {
              // Regular user input required - add as system message
              const questionMessage: Message = {
                id: Date.now().toString(),
                type: 'system',
                content: question,
                timestamp: new Date()
              };

              _setConversationState({
                messages: [...currentMessages, questionMessage],
                status: 'waiting_for_user',
                isWaitingForUser: true,
                isLoading: false,  // IMPORTANT: Stop loading so user can respond
                currentQuestion: question,
                metadata: {
                  ...useConversationStore.getState().metadata,
                  currentStage: 'waiting',
                  stageMessage: 'Waiting for your input...',
                  progress: 100
                }
              });
            }
          }
          else if (eventData.node === 'save_history') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'saving',
                stageMessage: backendMessage || 'Saving conversation...',
                progress: backendProgress || 98
              }
            });
          }
          else if (eventData.node === 'connect_mcp_agent' || eventData.node === 'mcp_tool_call') {
            _setConversationState({
              messages: currentMessages,
              metadata: {
                ...currentState.metadata,
                currentStage: 'connecting_mcp',
                stageMessage: backendMessage || 'Connecting to MCP servers...',
                progress: backendProgress || currentState.metadata?.progress || 75
              }
            });
          }
          else if (eventData.node === '__live_canvas__') {
            // Live canvas update during browser execution
            console.log('Live canvas update received:', eventData.data);
            const currentState = useConversationStore.getState();
            _setConversationState({
              has_canvas: eventData.data.has_canvas,
              canvas_type: eventData.data.canvas_type,
              canvas_content: eventData.data.canvas_content,
              browser_view: eventData.data.browser_view,
              plan_view: eventData.data.plan_view,
              current_view: eventData.data.current_view,
              metadata: {
                ...currentState.metadata,
                currentStage: 'executing',
                stageMessage: `Executing browser task... (Step ${eventData.data.screenshot_count || 0})`,
                progress: 85,
                browserScreenshotCount: eventData.data.screenshot_count
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
                console.warn('__end__ event received but no data field!');
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

            console.debug('=== FINAL RESPONSE DECISION ===', {
              hasAssistantMessage,
              hasFinalResponse: !!finalState.final_response,
              finalResponsePreview: finalState.final_response?.substring(0, 100),
              backendMessagesCount: backendMessages.length,
              filteredMessagesCount: finalMessages.length,
              assistantMessagesInBackend: backendMessages.filter(m => m.type === 'assistant').length,
              assistantMessagesInFiltered: finalMessages.filter(m => m.type === 'assistant').length
            });

            // ALWAYS add final_response if it exists and is different from the last assistant message
            // This ensures the orchestrator's final response is always displayed
            if (finalState.final_response && finalState.final_response.trim() !== '') {
              const lastAssistantMessage = [...finalMessages].reverse().find(msg => msg.type === 'assistant');
              const isDifferent = !lastAssistantMessage || lastAssistantMessage.content !== finalState.final_response;
              
              if (!hasAssistantMessage || isDifferent) {
                console.debug('Adding final_response as assistant message');
                const assistantMessage: Message = {
                  id: `final-${Date.now()}`,
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
                console.debug('Final response already in messages, skipping');
              }
            } else {
              console.debug('No final_response to add');
            }

            // Additional filtering to ensure no HTML content appears in chat messages
            // BUT: Only filter if the message is MOSTLY HTML (not just mentions HTML tags in text)
            finalMessages = finalMessages.map(msg => {
              if (msg.type === 'assistant' && msg.content) {
                // Check if the message content is PRIMARILY HTML (starts with HTML tags)
                const content = msg.content.trim();
                const startsWithHtml = (
                  content.startsWith('<!DOCTYPE html>') ||
                  content.startsWith('<html') ||
                  content.startsWith('<div') ||
                  content.startsWith('<button') ||
                  content.startsWith('<script>')
                );

                // Only replace if it's actual HTML code, not just text mentioning HTML
                if (startsWithHtml && finalState.has_canvas) {
                  console.debug('Replacing HTML code message with canvas reference');
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

            // Debug: Log canvas data
            console.log('🎨 CANVAS DEBUG - Final State:', {
              has_canvas: finalState.has_canvas,
              canvas_type: finalState.canvas_type,
              canvas_data: (finalState as any).canvas_data,
              canvas_content: finalState.canvas_content,
              canvas_title: (finalState as any).canvas_title,
              pending_confirmation: (finalState as any).pending_confirmation,
              canvas_requires_confirmation: (finalState as any).canvas_requires_confirmation,
              pending_confirmation_task: (finalState as any).pending_confirmation_task
            });
            
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
              canvas_data: (finalState as any).canvas_data !== undefined ? (finalState as any).canvas_data : (currentState as any).canvas_data,
              canvas_type: finalState.canvas_type !== undefined ? finalState.canvas_type : currentState.canvas_type,
              canvas_title: (finalState as any).canvas_title !== undefined ? (finalState as any).canvas_title : (currentState as any).canvas_title,
              has_canvas: finalState.has_canvas !== undefined ? finalState.has_canvas : currentState.has_canvas,
              browser_view: (finalState as any).browser_view !== undefined ? (finalState as any).browser_view : (currentState as any).browser_view,
              plan_view: (finalState as any).plan_view !== undefined ? (finalState as any).plan_view : (currentState as any).plan_view,
              current_view: (finalState as any).current_view !== undefined ? (finalState as any).current_view : (currentState as any).current_view,
              // Canvas confirmation fields
              pending_confirmation: (finalState as any).pending_confirmation !== undefined ? (finalState as any).pending_confirmation : (currentState as any).pending_confirmation,
              canvas_requires_confirmation: (finalState as any).canvas_requires_confirmation !== undefined ? (finalState as any).canvas_requires_confirmation : (currentState as any).canvas_requires_confirmation,
              pending_confirmation_task: (finalState as any).pending_confirmation_task !== undefined ? (finalState as any).pending_confirmation_task : (currentState as any).pending_confirmation_task,
              canvas_confirmation_message: (finalState as any).canvas_confirmation_message !== undefined ? (finalState as any).canvas_confirmation_message : (currentState as any).canvas_confirmation_message
            });
            // Explicitly set isLoading to false in the store
            console.debug('Setting isLoading to false after __end__ event');
            useConversationStore.setState({ isLoading: false });
            console.debug('Final state updated, isLoading:', useConversationStore.getState().isLoading);
            } catch (endError) {
              console.warn('Error processing __end__ event:', endError);
              // Always set isLoading to false even if there's an error
              useConversationStore.setState({ isLoading: false, status: 'idle' });
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
            console.warn('WebSocket error received:', {
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
            
            // Use the orchestrator's message as-is for better UX
            const errorSystemMessage: Message = {
              id: Date.now().toString(),
              type: 'assistant', // Use 'assistant' type so it looks like a normal response
              content: displayMessage,
              timestamp: new Date()
            };
            
            // Add error message to the existing messages in the store
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({
              messages: [...currentMessages, errorSystemMessage],
              status: 'idle', // Keep status as 'idle' so conversation can continue
              metadata: {
                ...useConversationStore.getState().metadata,
                currentStage: 'ready',
                stageMessage: 'Ready for next request',
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
          else {
            // Catch-all for any remaining unhandled events - still update progress if provided
            if (backendStage || backendMessage || backendProgress) {
              _setConversationState({
                messages: currentMessages,
                metadata: {
                  ...currentState.metadata,
                  currentStage: backendStage || currentState.metadata?.currentStage || 'processing',
                  stageMessage: backendMessage || `Processing ${eventData.node.replace(/_/g, ' ')}...`,
                  progress: backendProgress || currentState.metadata?.progress || 50
                }
              });
            }
            console.debug(`Unhandled WebSocket event: ${eventData.node}`, eventData);
          }

        } catch (parseError) {
          // Enhanced error handling for parse failures
          const errorMessage = parseError instanceof Error ? parseError.message : String(parseError);
          console.warn('Failed to parse WebSocket message:', {
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
              status: 'idle'
            });
          }
          
          // Set isLoading to false on parse error
          useConversationStore.setState({ isLoading: false, status: 'idle' });
        }
      };

      ws.current.onclose = (event) => {
        const closeMessage = `WebSocket closed with code ${event.code}${event.reason ? `: ${event.reason}` : ''}`;
        console.log(closeMessage, { code: event.code, reason: event.reason, cleanClose: event.wasClean });
        setIsConnected(false);
        
        // If the store is still in a processing state, handle gracefully
        if (useConversationStore.getState().status === 'processing') {
          // Use console.warn instead of console.error to avoid Next.js error overlay
          console.warn('WebSocket disconnected while processing');
          
          // Only show error message for abnormal closures (not code 1000)
          if (event.code !== 1000) {
            let disconnectReason = 'Connection closed unexpectedly.';
            // Categorize disconnect reason based on close code
            if (event.code === 1001) {
              disconnectReason = 'Server is shutting down. Please try again later.';
            } else if (event.code === 1002 || event.code === 1003) {
              disconnectReason = 'Protocol error. Please refresh the page and try again.';
            } else if (event.code === 1006) {
              disconnectReason = 'Connection lost. Please check your network and try again.';
            } else if (event.code === 1011) {
              disconnectReason = 'Server error. Please try again later.';
            }
            
            // Add disconnection message to chat only for abnormal closures
            const disconnectMessage: Message = {
              id: `disconnect_${Date.now()}`,
              type: 'system',
              content: `Connection Error: ${disconnectReason}`,
              timestamp: new Date()
            };
            const currentMessages = useConversationStore.getState().messages;
            _setConversationState({ 
              status: 'idle', // Keep as idle so conversation can continue
              messages: [...currentMessages, disconnectMessage]
            });
          } else {
            // Normal closure (code 1000) - just reset state without error message
            _setConversationState({ status: 'idle' });
          }
          
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
        console.warn('WebSocket connection error:', error);
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
            status: 'idle',
            messages: [...currentMessages, errorMessage]
          });
          useConversationStore.setState({ isLoading: false });
        }
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.warn('Failed to initialize WebSocket:', {
        url,
        error: errorMessage,
        type: error instanceof Error ? error.constructor.name : typeof error
      });
      
      // Update store with connection error
      setIsConnected(false);
      _setConversationState({
        status: 'idle',
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
    
    // Listen for reconnection requests
    const handleReconnect = () => {
      console.log('🔄 Reconnection requested');
      if (ws.current && (ws.current.readyState === WebSocket.CLOSED || ws.current.readyState === WebSocket.CLOSING)) {
        console.log('🔄 Reconnecting WebSocket...');
        connect();
      }
    };
    
    window.addEventListener('reconnect-websocket', handleReconnect);
    
    // Cleanup on unmount
    return () => {
      window.removeEventListener('reconnect-websocket', handleReconnect);
      // DON'T disconnect on component unmount - keep the connection alive
      // This prevents closing the connection before the __end__ event arrives
      // The connection will be reused if the component remounts
      // It will only close when the page is actually navigated away (browser handles this)
    };
  }, [connect, disconnect]);

  // This hook now only returns the connection status.
  return { isConnected };
}
