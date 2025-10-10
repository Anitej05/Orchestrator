import { create } from 'zustand';
import {
  ConversationState,
  Message,
  FileObject,
  Attachment,
} from '@/lib/types';
import {
  startConversation as apiStartConversation,
  continueConversation as apiContinueConversation,
  uploadFiles as apiUploadFiles,
  getConversationStatus,
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

export const useConversationStore = create<ConversationStore>((set, get) => ({
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
  isLoading: false,

  actions: {
    startConversation: async (input, files = []) => {
      set({ isLoading: true, status: 'processing' });
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

        set((state) => ({
          messages: [...state.messages, userMessage],
          // Add uploaded files to state immediately for optimistic UI update
          uploaded_files: [...(state.uploaded_files || []), ...uploadedFiles],
        }));

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
          set(state => ({ messages: [...state.messages, errorSystemMessage] }));
        };
        
        await sendMessageToWebSocket();
        
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
        set(state => ({ messages: [...state.messages, errorSystemMessage] }));
      } finally {
        // isLoading will be set to false by the WebSocket 'end' message handler
      }
    },

    continueConversation: async (input, files = []) => {
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

        set((state) => ({
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
                user_response: input,
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
          set(state => ({ messages: [...state.messages, errorSystemMessage] }));
        };
        
        await sendMessageToWebSocket();
        
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
        set(state => ({ messages: [...state.messages, errorSystemMessage] }));
      }
    },

    loadConversation: async (threadId) => {
      if (!threadId) return;
      set({ isLoading: true });
      try {
        const status = await getConversationStatus(threadId);
        // Convert the status to a conversation state format
        const conversationState: Partial<ConversationState> = {
          thread_id: status.thread_id || undefined,
          status: 'completed', // Assume completed since we're loading history
          messages: [], // Will be populated by WebSocket or additional API call
          isWaitingForUser: !!status.question_for_user,
          currentQuestion: status.question_for_user || undefined,
          task_agent_pairs: status.task_agent_pairs || [],
          final_response: status.final_response || undefined,
        };
        get().actions._setConversationState(conversationState);
      } catch (error: any) {
        console.error('Failed to load conversation:', error);
        // If loading fails, reset to a clean state but keep the thread_id
        get().actions.resetConversation();
        set({ thread_id: threadId });
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

    _setConversationState: (newState) => {
      set((state) => {
        // Handle uploaded_files - accumulate rather than replace to maintain history
        let updatedUploadedFiles = state.uploaded_files || [];
        if (newState.uploaded_files) {
          // Merge new files with existing ones, avoiding duplicates
          const newFiles = newState.uploaded_files.filter(
            newFile => !updatedUploadedFiles.some(
              existingFile => existingFile.file_path === newFile.file_path
            )
          );
          updatedUploadedFiles = [...updatedUploadedFiles, ...newFiles];
        }
        
        // Handle task_agent_pairs - accumulate rather than replace to maintain history
        let updatedTaskAgentPairs = state.task_agent_pairs || [];
        if (newState.task_agent_pairs) {
          // Merge new pairs with existing ones, avoiding duplicates
          const newPairs = newState.task_agent_pairs.filter(
            newPair => !updatedTaskAgentPairs.some(
              existingPair => existingPair.task_name === newPair.task_name
            )
          );
          updatedTaskAgentPairs = [...updatedTaskAgentPairs, ...newPairs];
        }
        
        // Handle plan - preserve existing plan if new plan is empty or undefined
        let updatedPlan = state.plan || [];
        if (newState.plan && newState.plan.length > 0) {
          // For plans, we want to show the current plan, not accumulate
          updatedPlan = newState.plan;
        }
        
        // Handle metadata - merge deeply to preserve all information
        let updatedMetadata = state.metadata || {};
        if (newState.metadata) {
          // Deep merge metadata to preserve all fields
          updatedMetadata = {
            ...updatedMetadata,
            ...newState.metadata,
            // Specifically preserve completed_tasks and parsed_tasks if they exist
            completed_tasks: newState.metadata.completed_tasks || updatedMetadata.completed_tasks || [],
            parsed_tasks: newState.metadata.parsed_tasks || updatedMetadata.parsed_tasks || [],
            // Preserve original_prompt throughout conversation
            original_prompt: newState.metadata.original_prompt || updatedMetadata.original_prompt || state.metadata?.original_prompt
          };
        }

        return {
          ...state,
          ...newState,
          uploaded_files: updatedUploadedFiles,
          task_agent_pairs: updatedTaskAgentPairs,
          plan: updatedPlan,
          metadata: updatedMetadata,
          // Ensure messages are properly created with Date objects for timestamps
          messages: (newState.messages || state.messages || []).map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp || Date.now()),
          })),
        };
      });
      // Persist the thread_id to localStorage
      if (newState.thread_id && typeof window !== 'undefined') {
        localStorage.setItem('thread_id', newState.thread_id);
      }
    },
  },
}));
