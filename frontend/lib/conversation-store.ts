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

interface ConversationStore extends ConversationState {
  isLoading: boolean;
  actions: {
    startConversation: (input: string, files?: File[]) => Promise<void>;
    continueConversation: (input: string, files?: File[]) => Promise<void>;
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
  // Plan approval fields
  approval_required: false,
  estimated_cost: 0,
  task_count: 0,
  isLoading: false,

  actions: {
    startConversation: async (input: string, files: File[] = []) => {
      // Clear previous conversation state when starting a new conversation
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

        const userMessage: Message = {
          id: Date.now().toString(),
          type: 'user',
          content: input,
          timestamp: new Date(),
          attachments: attachments.length > 0 ? attachments : undefined,
        };

        set({
          messages: [userMessage],
          uploaded_files: uploadedFiles,
        });

        // Send message to WebSocket with connection wait
        const sendMessageToWebSocket = async () => {
          const maxAttempts = 30; // Increased from 10 to 30 attempts
          const delayMs = 500; // Increased from 200ms to 500ms
          
          for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const ws = (window as any).__websocket;
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                thread_id: get().thread_id,
                prompt: input,
                files: uploadedFiles.map(file => ({
                  file_name: file.file_name,
                  file_path: file.file_path,
                  file_type: file.file_type
                }))
              }));
              console.log(`WebSocket message sent successfully on attempt ${attempt + 1}`);
              return; // Successfully sent
            }
            
            console.log(`WebSocket not ready, attempt ${attempt + 1}/${maxAttempts}, waiting ${delayMs}ms...`);
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, delayMs));
          }
          
          // If we've exhausted all attempts, show an error
          console.error('WebSocket not connected after multiple attempts');
          set({ status: 'error', isLoading: false });
          
          // Add error message to chat
          const errorSystemMessage: Message = {
            id: Date.now().toString(),
            type: 'system',
            content: 'Error: WebSocket connection failed. Please check if the backend server is running on localhost:8000.',
            timestamp: new Date()
          };
          set((state: ConversationStore) => ({ messages: [...state.messages, errorSystemMessage] }));
        };
        
        await sendMessageToWebSocket();
        
        // Note: Timeout handling is done by the WebSocket connection itself
        // The backend will send __end__ or __error__ events to complete the request
        
      } catch (error: any) {
        const errorMessage = error.message || 'An unknown error occurred';
        set({ status: 'error', isLoading: false });
        // Optionally, add an error message to the chat
        const errorSystemMessage: Message = {
          id: Date.now().toString(),
          type: 'system',
          content: `Error: ${errorMessage}`,
          timestamp: new Date()
        };
        set((state: ConversationStore) => ({ messages: [...state.messages, errorSystemMessage] }));
      }
    },

    continueConversation: async (input: string, files: File[] = []) => {
      const thread_id = get().thread_id;
      if (!thread_id) {
        console.error('Cannot continue conversation without a thread ID.');
        return;
      }
      console.log(`Continuing conversation with thread_id: ${thread_id}`);
      set({ isLoading: true, status: 'processing', isWaitingForUser: false });

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

        const userMessage: Message = {
          id: Date.now().toString(),
          type: 'user',
          content: input,
          timestamp: new Date(),
          attachments: attachments.length > 0 ? attachments : undefined,
        };

        set((state: ConversationStore) => ({
          messages: [...state.messages, userMessage],
          uploaded_files: [...(state.uploaded_files || []), ...uploadedFiles],
        }));

        // Send message to WebSocket with connection wait
        const sendMessageToWebSocket = async () => {
          const maxAttempts = 30; // Increased from 10 to 30 attempts
          const delayMs = 500; // Increased from 200ms to 500ms
          
          for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const ws = (window as any).__websocket;
            if (ws && ws.readyState === WebSocket.OPEN) {
              const messageData = {
                thread_id: thread_id,
                prompt: input,  // Use 'prompt' for continuing conversation, not 'user_response'
                files: uploadedFiles.map(file => ({
                  file_name: file.file_name,
                  file_path: file.file_path,
                  file_type: file.file_type
                }))
              };
              console.log('Sending continue conversation message:', messageData);
              ws.send(JSON.stringify(messageData));
              console.log(`WebSocket message sent successfully on attempt ${attempt + 1}`);
              return; // Successfully sent
            }
            
            console.log(`WebSocket not ready, attempt ${attempt + 1}/${maxAttempts}, waiting ${delayMs}ms...`);
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, delayMs));
          }
          
          // If we've exhausted all attempts, show an error
          console.error('WebSocket not connected after multiple attempts');
          set({ status: 'error', isLoading: false });
          
          // Add error message to chat
          const errorSystemMessage: Message = {
            id: Date.now().toString(),
            type: 'system',
            content: 'Error: WebSocket connection failed. Please check if the backend server is running on localhost:8000.',
            timestamp: new Date()
          };
          set((state: ConversationStore) => ({ messages: [...state.messages, errorSystemMessage] }));
        };
        
        await sendMessageToWebSocket();
        
        // Note: Timeout handling is done by the WebSocket connection itself
        // The backend will send __end__ or __error__ events to complete the request
        
      } catch (error: any) {
        const errorMessage = error.message || 'An unknown error occurred';
        console.error('Error in continueConversation:', errorMessage);
        set({ status: 'error', isLoading: false });
        const errorSystemMessage: Message = {
          id: Date.now().toString(),
          type: 'system',
          content: `Error: ${errorMessage}`,
          timestamp: new Date()
        };
        set((state: ConversationStore) => ({ messages: [...state.messages, errorSystemMessage] }));
      }
    },

    loadConversation: async (threadId: string) => {
      if (!threadId) return;
      set({ isLoading: true });
      try {
        // Load full conversation history from backend
        const response = await fetch(`http://localhost:8000/api/chat/history/${threadId}`);
        if (!response.ok) {
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
        
        // Save thread_id to localStorage for persistence
        if (typeof window !== 'undefined') {
          localStorage.setItem('thread_id', threadId);
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

        // Handle messages - use backend messages as source of truth
        // Backend provides the complete, authoritative message history
        let updatedMessages = state.messages;
        if (newState.messages !== undefined) {
          // Use backend messages directly - they are already complete and deduplicated
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
        };
      });
      // Persist the thread_id to localStorage
      if (newState.thread_id && typeof window !== 'undefined') {
        localStorage.setItem('thread_id', newState.thread_id);
      }
    },
  },
}));
