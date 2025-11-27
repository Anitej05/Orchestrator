import { create } from 'zustand';
import {
  ConversationState,
  Message,
  FileObject,
  Attachment,
} from '@/lib/types';
import {
  uploadFiles as apiUploadFiles,
} from '@/lib/api-client';

// Helper to read a file as a Data URL for image previews
const readFileAsDataURL = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

// Helper to create deterministic message IDs (matches backend logic)
const createMessageId = (content: string, type: string, timestampMs: number): string => {
  // Convert milliseconds to seconds to match backend (Python's time.time())
  const timestampSeconds = Math.floor(timestampMs / 1000);
  // Use MD5-like hash to match backend exactly
  // For browser compatibility, we'll use a simple hash that produces consistent results
  const str = `${type}:${content}:${timestampSeconds}`;
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(16).padStart(16, '0').substring(0, 16);
};

interface ConversationStore extends ConversationState {
  isLoading: boolean;
  actions: {
    startConversation: (input: string, files?: File[], planningMode?: boolean, owner?: string) => Promise<void>;
    continueConversation: (input: string, files?: File[], planningMode?: boolean, owner?: string) => Promise<void>;
    loadConversation: (threadId: string) => Promise<void>;
    resetConversation: () => void;
    // Internal action to set the full state from an API response
    _setConversationState: (newState: Partial<ConversationStore>) => void;
  };
}

export const useConversationStore = create<ConversationStore>((set: any, get: any) => ({
  thread_id: undefined,
  status: 'idle',
  messages: [],
  isWaitingForUser: false,
  currentQuestion: undefined,
  task_agent_pairs: [],
  final_response: undefined,
  metadata: {},
  uploaded_files: [],
  plan: [],
  // Canvas feature fields
  canvas_content: undefined,
  canvas_type: undefined,
  has_canvas: false,
  browser_view: undefined,
  plan_view: undefined,
  current_view: 'browser',
  // Plan approval fields
  approval_required: false,
  estimated_cost: 0,
  task_count: 0,
  // Real-time task tracking
  task_statuses: {},
  current_executing_task: null,
  isLoading: false,

  actions: {
    startConversation: async (input: string, files: File[] = [], planningMode: boolean = false, owner?: string) => {
      // Clear previous conversation state when starting a new conversation
      console.debug(`Starting conversation with planning mode: ${planningMode}`);
      set({ 
        isLoading: true, 
        status: 'processing',
        thread_id: undefined,
        messages: [],
        task_agent_pairs: [],
        final_response: undefined,
        metadata: {},
        plan: [],
        canvas_content: undefined,
        canvas_type: undefined,
        has_canvas: false,
        browser_view: undefined,
        plan_view: undefined,
        current_view: 'browser',
      });
      
      try {
        let uploadedFiles: FileObject[] = [];
        if (files.length > 0) {
          uploadedFiles = await apiUploadFiles(files);
        }

        const attachments: Attachment[] = await Promise.all(
          files.map(async (file) => ({
            name: file.name,
            type: file.type,
            content: file.type.startsWith('image/') ? await readFileAsDataURL(file) : '',
          }))
        );

        const timestamp = Date.now();
        const messageId = createMessageId(input, 'user', timestamp);
          console.debug(`Frontend creating user message (continue): id=${messageId}, timestamp=${timestamp}, content=${input.substring(0, 50)}`);
        const userMessage: Message = {
          id: messageId,
          type: 'user',
          content: input,
          timestamp: new Date(timestamp),
          attachments: attachments.length > 0 ? attachments : undefined,
        };

        set({
          messages: [userMessage],
          uploaded_files: uploadedFiles,
        });

        // Send message to WebSocket with connection wait and enhanced error handling
        const sendMessageToWebSocket = async () => {
          const maxAttempts = 50; // Increase attempts
          const delayMs = 300; // Reduce delay
          const maxWaitTime = (maxAttempts * delayMs); // ~15 seconds total
          
          // Check if WebSocket is closed and trigger reconnection
          const ws = (window as any).__websocket;
          if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
            console.log('🔄 WebSocket is closed, triggering reconnection...');
            // Trigger reconnection by dispatching custom event
            window.dispatchEvent(new CustomEvent('reconnect-websocket'));
            // Wait a bit for reconnection to start
            await new Promise(resolve => setTimeout(resolve, 500));
          }
          
          for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
              const ws = (window as any).__websocket;
              
              // Check if WebSocket exists and is open
              if (!ws) {
                if (attempt === 0) {
                  console.log('⏳ WebSocket not yet initialized, waiting for connection...');
                } else if (attempt % 10 === 0) {
                  console.log(`⏳ Still waiting for WebSocket... (attempt ${attempt + 1}/${maxAttempts})`);
                }
              } else if (ws.readyState === WebSocket.OPEN) {
                // Successfully connected, send the message
                try {
                  const message = {
                    thread_id: get().thread_id,
                    prompt: input,
                    planning_mode: planningMode,
                    owner: owner,
                    files: uploadedFiles.map(file => ({
                      file_name: file.file_name,
                      file_path: file.file_path,
                      file_type: file.file_type
                    }))
                  };
                  console.log('📤 Sending WebSocket message:', { 
                    thread_id: message.thread_id,
                    has_prompt: !!message.prompt,
                    planning_mode: message.planning_mode,
                    has_owner: !!message.owner,
                    files_count: message.files.length
                  });
                  ws.send(JSON.stringify(message));
                  console.log(`✅ WebSocket message sent successfully on attempt ${attempt + 1}`);
                  return; // Successfully sent
                } catch (sendErr) {
                  console.error(`Failed to serialize/send message: ${sendErr instanceof Error ? sendErr.message : String(sendErr)}`);
                  throw new Error('Failed to send message to server');
                }
              } else if (ws.readyState === WebSocket.CONNECTING) {
                if (attempt % 5 === 0) {
                  console.log(`🔄 WebSocket connecting (attempt ${attempt + 1}/${maxAttempts})...`);
                }
              } else if (ws.readyState === WebSocket.CLOSING || ws.readyState === WebSocket.CLOSED) {
                console.warn(`❌ WebSocket in ${ws.readyState === WebSocket.CLOSING ? 'closing' : 'closed'} state`);
              }
              
              // Wait before retrying
              if (attempt < maxAttempts - 1) {
                await new Promise(resolve => setTimeout(resolve, delayMs));
              }
            } catch (retryErr) {
              console.error(`Error during send attempt ${attempt + 1}: ${retryErr instanceof Error ? retryErr.message : String(retryErr)}`);
              if (attempt < maxAttempts - 1) {
                await new Promise(resolve => setTimeout(resolve, delayMs));
              }
            }
          }
          
          // If we've exhausted all attempts, show an error
          console.error(`❌ WebSocket not connected after ${maxAttempts} attempts (~${maxWaitTime}ms)`);
          console.error('Backend might not be running. Please check: http://localhost:8000');
          set({ status: 'error', isLoading: false });
          
          // Add detailed error message to chat
          const errorSystemMessage: Message = {
            id: Date.now().toString(),
            type: 'system',
            content: `Error: Failed to establish connection after ${Math.round(maxWaitTime / 1000)} seconds. Please check if the backend server is running at http://localhost:8000 and refresh the page to try again.`,
            timestamp: new Date()
          };
          set((state: ConversationStore) => ({ messages: [...state.messages, errorSystemMessage] }));
        };
        
        await sendMessageToWebSocket();
        
        // Note: Timeout handling is done by the WebSocket connection itself
        // The backend will send __end__ or __error__ events to complete the request
        
      } catch (error: any) {
        const errorMessage = error instanceof Error ? error.message : (error?.message || 'An unknown error occurred');
        console.error('Error in startConversation:', errorMessage, error);
        
        // Don't add error messages here - the orchestrator will send its response via WebSocket
        // Just reset the loading state and let the WebSocket handle the response
        set({ isLoading: false, status: 'idle' });
        
        // The orchestrator's actual response will come through the WebSocket __end__ event
        // Don't throw - let the user continue
      }
    },

    continueConversation: async (input: string, files: File[] = [], planningMode: boolean = false, owner?: string) => {
      const thread_id = get().thread_id;
      if (!thread_id) {
        console.error('Cannot continue conversation without a thread ID.');
        return;
      }
      
      // Capture isWaitingForUser BEFORE we modify it
      const wasWaitingForUser = get().isWaitingForUser;
      
      console.log(`Continuing conversation with thread_id: ${thread_id}, planning_mode: ${planningMode}, wasWaitingForUser: ${wasWaitingForUser}`);
      set({ isLoading: true, status: 'processing', isWaitingForUser: false, task_statuses: {}, current_executing_task: null });

      try {
        let uploadedFiles: FileObject[] = [];
        if (files.length > 0) {
          uploadedFiles = await apiUploadFiles(files);
        }

        const attachments: Attachment[] = await Promise.all(
          files.map(async (file) => ({
            name: file.name,
            type: file.type,
            content: file.type.startsWith('image/') ? await readFileAsDataURL(file) : '',
          }))
        );

        // Check if this is an approval/cancel response (should not be shown as a message)
        const isApprovalResponse = wasWaitingForUser && get().approval_required === false && 
                                   (input.toLowerCase() === 'approve' || input.toLowerCase() === 'cancel');

        // Only add user message if it's not an approval response
        if (!isApprovalResponse) {
          const timestamp = Date.now();
          const messageId = createMessageId(input, 'human', timestamp);
          console.log(`Frontend creating user message (continue): id=${messageId}, timestamp=${timestamp}, content=${input.substring(0, 50)}`);
          const userMessage: Message = {
            id: messageId,
            type: 'user',
            content: input,
            timestamp: new Date(timestamp),
            attachments: attachments.length > 0 ? attachments : undefined,
          };

          set((state: ConversationStore) => ({
            messages: [...state.messages, userMessage],
            uploaded_files: [...(state.uploaded_files || []), ...uploadedFiles],
          }));
        } else {
          // For approval responses, just update uploaded files without adding a message
          console.log('Skipping message addition for approval response:', input);
          set((state: ConversationStore) => ({
            uploaded_files: [...(state.uploaded_files || []), ...uploadedFiles],
          }));
        }

        // Send message to WebSocket with connection wait
        const sendMessageToWebSocket = async () => {
          const maxAttempts = 30; // Increased from 10 to 30 attempts
          const delayMs = 500; // Increased from 200ms to 500ms
          
          for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const ws = (window as any).__websocket;
            if (ws && ws.readyState === WebSocket.OPEN) {
              // Use the captured state from before we modified it
              const isAnsweringQuestion = wasWaitingForUser;
              
              const messageData = {
                thread_id: thread_id,
                // Use 'user_response' only when answering a question, otherwise use 'prompt'
                ...(isAnsweringQuestion ? { user_response: input } : { prompt: input }),
                planning_mode: planningMode,
                owner: owner,
                files: uploadedFiles.map(file => ({
                  file_name: file.file_name,
                  file_path: file.file_path,
                  file_type: file.file_type
                }))
              };
              console.debug('=== FRONTEND: Sending WebSocket message ===');
              console.debug('  wasWaitingForUser:', wasWaitingForUser);
              console.debug('  isAnsweringQuestion:', isAnsweringQuestion);
              console.debug('  sending as:', isAnsweringQuestion ? 'user_response' : 'prompt');
              console.debug('  input:', input);
              console.debug('  planning_mode:', planningMode);
              console.debug('  full message:', messageData);
              
              try {
                ws.send(JSON.stringify(messageData));
                console.debug(`WebSocket message sent successfully on attempt ${attempt + 1}`);
                return; // Successfully sent
              } catch (sendErr) {
                console.error(`Failed to send message: ${sendErr instanceof Error ? sendErr.message : String(sendErr)}`);
                throw new Error('Failed to send message to server');
              }
            }
            
            console.debug(`WebSocket not ready, attempt ${attempt + 1}/${maxAttempts}, waiting ${delayMs}ms...`);
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, delayMs));
          }
          
          // If we've exhausted all attempts, show an error
          const maxWaitTime = maxAttempts * delayMs;
          console.error(`WebSocket not connected after ${maxAttempts} attempts (~${maxWaitTime}ms)`);
          set({ status: 'error', isLoading: false });
          
          // Add detailed error message to chat
          const errorSystemMessage: Message = {
            id: Date.now().toString(),
            type: 'system',
            content: `Error: Failed to send response after ${Math.round(maxWaitTime / 1000)} seconds. Connection may have been lost. Please refresh the page and try again.`,
            timestamp: new Date()
          };
          set((state: ConversationStore) => ({ messages: [...state.messages, errorSystemMessage] }));
        };
        
        await sendMessageToWebSocket();
        
        // Note: Timeout handling is done by the WebSocket connection itself
        // The backend will send __end__ or __error__ events to complete the request
        
      } catch (error: any) {
        const errorMessage = error.message || 'An unknown error occurred';
        console.error('Error in continueConversation:', errorMessage, error);
        
        // Don't add error messages here - the orchestrator will send its response via WebSocket
        // Just reset the loading state and let the WebSocket handle the response
        set({ isLoading: false, status: 'idle' });
        
        // The orchestrator's actual response will come through the WebSocket __end__ event
        // Don't throw - let the user continue and wait for WebSocket response
      }
    },

    loadConversation: async (threadId: string) => {
      if (!threadId) return;
      set({ isLoading: true });
      try {
        // Sanitize threadId - remove any appended index like :1 that LangGraph might add
        const cleanThreadId = threadId.split(':')[0];
        
        // Use authFetch helper which handles Clerk JWT properly
        const { authFetch } = await import('./auth-fetch');
        const response = await authFetch(`http://localhost:8000/api/conversations/${cleanThreadId}`);
        
        if (!response.ok) {
          if (response.status === 404) {
            console.log('Conversation not found, starting fresh');
            return;
          }
          const errorText = await response.text();
          console.error('Failed to load conversation:', response.status, errorText);
          throw new Error(`Failed to load conversation history: ${response.status}`);
        }
        
        const conversationData = await response.json();
        console.log('Loaded conversation data:', conversationData);
        
        // Convert message timestamps from strings to Date objects
        const messages = (conversationData.messages || []).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        
        // Set the full conversation state
        const conversationState: Partial<ConversationState> = {
          thread_id: conversationData.thread_id || threadId,
          status: conversationData.status || 'completed',
          messages: messages,
          isWaitingForUser: conversationData.pending_user_input || false,
          currentQuestion: conversationData.question_for_user || undefined,
          task_agent_pairs: conversationData.task_agent_pairs || [],
          final_response: conversationData.final_response || undefined,
          metadata: conversationData.metadata || {},
          uploaded_files: conversationData.uploaded_files || [],
          plan: conversationData.plan || [],
          canvas_content: conversationData.canvas_content,
          canvas_type: conversationData.canvas_type,
          has_canvas: conversationData.has_canvas || false,
        };
        
        get().actions._setConversationState(conversationState);
        
        // Explicitly set isLoading to false after loading
        set({ isLoading: false });
        
        // Save thread_id to localStorage for persistence (use clean version)
        if (typeof window !== 'undefined') {
          localStorage.setItem('thread_id', cleanThreadId);
        }
        
        console.log('Conversation loaded successfully:', threadId);
      } catch (error: any) {
        console.error('Failed to load conversation:', error);
        // If loading fails, just clear the localStorage to prevent infinite retry
        if (typeof window !== 'undefined') {
          localStorage.removeItem('thread_id');
        }
        // Don't reset - just log the error and keep current state
        console.warn('Could not restore conversation from localStorage');
      } finally {
        set({ isLoading: false });
      }
    },

    resetConversation: () => {
      set({
        thread_id: undefined,
        status: 'idle',
        messages: [],
        isWaitingForUser: false,
        currentQuestion: undefined,
        task_agent_pairs: [],
        final_response: undefined,
        metadata: {},
        uploaded_files: [],
        task_statuses: {},
        current_executing_task: null,
        isLoading: false,
      });
      // Also clear from localStorage
      if (typeof window !== 'undefined') {
        localStorage.removeItem('thread_id');
      }
    },

    _setConversationState: (newState: Partial<ConversationStore>) => {
      set((state: ConversationStore) => {
        // When loading a conversation (has thread_id but different from current), replace data
        // When updating current conversation (same thread_id), merge data
        const isLoadingConversation = newState.thread_id && newState.thread_id !== state.thread_id;
        
        // Handle uploaded_files
        let updatedUploadedFiles = state.uploaded_files || [];
        if (newState.uploaded_files !== undefined) {
          if (isLoadingConversation) {
            // Replace when loading
            updatedUploadedFiles = newState.uploaded_files;
          } else {
            // Merge when updating
            const newFiles = newState.uploaded_files.filter(
              (newFile: FileObject) => !updatedUploadedFiles.some(
                (existingFile: FileObject) => existingFile.file_path === newFile.file_path
              )
            );
            updatedUploadedFiles = [...updatedUploadedFiles, ...newFiles];
          }
        }
        
        // Handle task_agent_pairs
        let updatedTaskAgentPairs = state.task_agent_pairs || [];
        if (newState.task_agent_pairs !== undefined) {
          if (isLoadingConversation) {
            // Replace when loading
            updatedTaskAgentPairs = newState.task_agent_pairs;
          } else {
            // Merge when updating
            const newPairs = newState.task_agent_pairs.filter(
              (newPair: any) => !updatedTaskAgentPairs.some(
                (existingPair: any) => existingPair.task_name === newPair.task_name
              )
            );
            updatedTaskAgentPairs = [...updatedTaskAgentPairs, ...newPairs];
          }
        }
        
        // Handle plan
        let updatedPlan = state.plan || [];
        if (newState.plan !== undefined) {
          if (isLoadingConversation || (newState.plan && newState.plan.length > 0)) {
            // Replace when loading or when new plan exists
            updatedPlan = newState.plan;
          }
        }
        
        // Handle metadata
        let updatedMetadata = state.metadata || {};
        if (newState.metadata) {
          if (isLoadingConversation) {
            // Replace when loading
            updatedMetadata = newState.metadata;
          } else {
            // Deep merge metadata to preserve all fields
            // Accumulate completed_tasks and parsed_tasks
            const existingCompletedTasks = updatedMetadata.completed_tasks || [];
            const newCompletedTasks = newState.metadata.completed_tasks || [];
            const mergedCompletedTasks = [...existingCompletedTasks];
            
            // Add new completed tasks that aren't already in the list
            newCompletedTasks.forEach((newTask: any) => {
              if (!mergedCompletedTasks.some((existingTask: any) => existingTask.task_name === newTask.task_name)) {
                mergedCompletedTasks.push(newTask);
              }
            });
            
            const existingParsedTasks = updatedMetadata.parsed_tasks || [];
            const newParsedTasks = newState.metadata.parsed_tasks || [];
            const mergedParsedTasks = [...existingParsedTasks];
            
            // Add new parsed tasks that aren't already in the list
            newParsedTasks.forEach((newTask: any) => {
              if (!mergedParsedTasks.some((existingTask: any) => existingTask.task_name === newTask.task_name)) {
                mergedParsedTasks.push(newTask);
              }
            });
            
            updatedMetadata = {
              ...updatedMetadata,
              ...newState.metadata,
              completed_tasks: mergedCompletedTasks,
              parsed_tasks: mergedParsedTasks,
              // Preserve original_prompt throughout conversation
              original_prompt: newState.metadata.original_prompt || updatedMetadata.original_prompt || state.metadata?.original_prompt
            };
          }
        }

        // Handle messages - merge backend messages with frontend messages intelligently
        let updatedMessages = state.messages || [];
        if (newState.messages !== undefined) {
          // When loading a conversation (different thread_id), replace all messages
          if (isLoadingConversation) {
            updatedMessages = newState.messages
              .filter((msg: any) => {
                // Keep all non-assistant messages
                if (msg.type !== 'assistant') return true;
                // Keep assistant messages that have content
                return msg.content && msg.content.trim() !== '';
              })
              .map((msg: any) => ({
                ...msg,
                timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp || Date.now()),
              }));
          } else {
            // When updating current conversation, merge messages intelligently
            // Create a map of existing messages by ID for quick lookup
            const existingMessagesMap = new Map(
              updatedMessages.map((msg: any) => [msg.id, msg])
            );
            
            // Process backend messages
            const backendMessages = newState.messages
              .filter((msg: any) => {
                // Keep all non-assistant messages
                if (msg.type !== 'assistant') return true;
                // Keep assistant messages that have content
                return msg.content && msg.content.trim() !== '';
              })
              .map((msg: any) => ({
                ...msg,
                timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp || Date.now()),
              }));
            
            // Merge: Add new messages from backend that don't exist in frontend
            // Use content-based deduplication as fallback since IDs might not match
            backendMessages.forEach((backendMsg: any) => {
              // First try ID-based matching
              let isDuplicate = existingMessagesMap.has(backendMsg.id);
              
              // If not found by ID, check for content-based duplicates
              // (in case hash algorithms don't match between frontend and backend)
              if (!isDuplicate) {
                isDuplicate = updatedMessages.some((existingMsg: any) => 
                  existingMsg.type === backendMsg.type &&
                  existingMsg.content === backendMsg.content
                  // Don't check timestamp - same content + type = duplicate regardless of time
                );
              }
              
              if (!isDuplicate) {
                // New message from backend - add it
                console.log(`Adding new message from backend: id=${backendMsg.id}, type=${backendMsg.type}, content=${backendMsg.content?.substring(0, 50)}`);
                updatedMessages.push(backendMsg);
              } else {
                // Message exists - skip or update
                console.log(`Skipping duplicate message: id=${backendMsg.id}, type=${backendMsg.type}, content=${backendMsg.content?.substring(0, 50)}`);
                // Optionally update the existing message with backend data
                const existingMsg = existingMessagesMap.get(backendMsg.id) || 
                  updatedMessages.find((msg: any) => 
                    msg.type === backendMsg.type &&
                    msg.content === backendMsg.content
                  );
                if (existingMsg) {
                  Object.assign(existingMsg, backendMsg);
                }
              }
            });
            
            // Sort messages by timestamp to maintain order
            updatedMessages.sort((a: any, b: any) => {
              const timeA = a.timestamp instanceof Date ? a.timestamp.getTime() : new Date(a.timestamp).getTime();
              const timeB = b.timestamp instanceof Date ? b.timestamp.getTime() : new Date(b.timestamp).getTime();
              return timeA - timeB;
            });
          }
        }

        return {
          ...state,
          ...newState,
          uploaded_files: updatedUploadedFiles,
          task_agent_pairs: updatedTaskAgentPairs,
          plan: updatedPlan,
          metadata: updatedMetadata,
          messages: updatedMessages,
          // Handle canvas data
          canvas_content: newState.canvas_content !== undefined ? newState.canvas_content : state.canvas_content,
          canvas_type: newState.canvas_type !== undefined ? newState.canvas_type : state.canvas_type,
          has_canvas: newState.has_canvas !== undefined ? newState.has_canvas : state.has_canvas,
          // Handle browser and plan views
          browser_view: (newState as any).browser_view !== undefined ? (newState as any).browser_view : (state as any).browser_view,
          plan_view: (newState as any).plan_view !== undefined ? (newState as any).plan_view : (state as any).plan_view,
          current_view: (newState as any).current_view !== undefined ? (newState as any).current_view : (state as any).current_view,
        };
      });
      // Persist the thread_id to localStorage
      if (newState.thread_id && typeof window !== 'undefined') {
        localStorage.setItem('thread_id', newState.thread_id);
      }
    },
  },
}));
