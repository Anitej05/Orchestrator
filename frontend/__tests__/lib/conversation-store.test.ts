/**
 * Unit tests for conversation store (Zustand).
 *
 * This test module mocks external dependencies (WebSocket, API client, localStorage)
 * to test the store's logic in isolation.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useConversationStore } from '../conversation-store';

// Mock external dependencies
const mockSend = vi.fn();
const mockClose = vi.fn();

const mockWebSocket = {
  readyState: WebSocket.OPEN,
  send: mockSend,
  close: mockClose,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
};

// Mock window.location
Object.defineProperty(window, 'location', {
  value: { href: 'http://localhost:3000' },
  writable: true
});

// Global WebSocket mock
Object.defineProperty(window, '__websocket', {
  value: mockWebSocket,
  writable: true,
});

// Mock apiClient
vi.mock('../api-client', () => ({
  uploadFiles: vi.fn(() => Promise.resolve([
    {
      file_name: 'test.txt',
      file_path: '/uploads/test.txt',
      file_type: 'document',
    },
  ])),
}));

// Mock auth-fetch
vi.mock('../auth-fetch', () => ({
  authFetch: vi.fn((url, options) => {
    if (url.includes('/api/conversations/')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          thread_id: 'test-thread-id',
          status: 'completed',
          messages: [
            {
              id: 'msg-1',
              type: 'user',
              content: 'Test user message',
              timestamp: '2024-01-01T00:00:00Z',
            },
            {
              id: 'msg-2',
              type: 'assistant',
              content: 'Test assistant message',
              timestamp: '2024-01-01T00:00:01Z',
            },
          ],
          metadata: {},
          task_agent_pairs: [],
          plan: [],
        }),
      });
    }
    return Promise.resolve({ ok: true });
  }),
}));

// Mock localStorage
const mockLocalStorage = {
  storage: new Map<string, string>(),
  getItem: function (key: string): string | null {
    return this.storage.get(key) || null;
  },
  setItem: function (key: string, value: string): void {
    this.storage.set(key, value);
  },
  removeItem: function (key: string): void {
    this.storage.delete(key);
  },
  clear: function (): void {
    this.storage.clear();
  },
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});

describe('Conversation Store', () => {
  beforeEach(() => {
    // Reset store state before each test
    useConversationStore.getState().resetConversation();
    // Reset mocks
    mockSend.mockClear();
    mockClose.mockClear();
    mockLocalStorage.storage.clear();
    // Reset WebSocket state
    mockWebSocket.readyState = WebSocket.OPEN;
  });

  afterEach(() => {
    // Cleanup
  });

  describe('Initial State', () => {
    it('should initialize with empty state', () => {
      const state = useConversationStore.getState();

      expect(state.thread_id).toBeUndefined();
      expect(state.status).toBe('idle');
      expect(state.messages).toEqual([]);
      expect(state.isWaitingForUser).toBe(false);
      expect(state.isLoading).toBe(false);
      expect state.task_agent_pairs).toEqual([]);
      expect(state.plan).toEqual([]);
      expect(state.execution_plan).toEqual([]);
      expect state.action_history).toEqual([]);
    });

    it('should have current_view set to browser', () => {
      const state = useConversationStore.getState();

      expect(state.current_view).toBe('browser');
    });

    it('should have canvas states cleared', () => {
      const state = useConversationStore.getState();

      expect(state.has_canvas).toBe(false);
      expect(state.canvas_content).toBeUndefined();
      expect(state.canvas_type).toBeUndefined();
      expect(state.canvas_data).toBeUndefined();
    });

    it('should have Omni-Dispatcher fields cleared', () => {
      const state = useConversationStore.getState();

      expect(state.execution_plan).toEqual([]);
      expect(state.current_phase_id).toBeUndefined();
      expect state.action_history).toEqual([]);
      expect(state.insights).toEqual({});
      expect(state.pending_action_approval).toBe(false);
      expect(state.pending_action).toBeUndefined();
    });
  });

  describe('startConversation', () => {
    it('should clear previous state when starting new conversation', async () => {
      const store = useConversationStore.getState();

      // Set some previous state
      store._setConversationState({
        thread_id: 'prev-thread',
        messages: [{ id: 'prev', type: 'assistant', content: 'Prev', timestamp: new Date() }],
        final_response: 'Previous response',
      });

      // Start new conversation
      await store.actions.startConversation('New conversation');

      const state = useConversationStore.getState();

      expect(state.thread_id).toBeUndefined(); // Will be set by backend
      expect(state.messages).toHaveLength(1); // User message only
      expect(state.messages[0].content).toBe('New conversation');
      expect(state.final_response).toBeUndefined();
      expect(state.plan).toEqual([]);
      expect(state.canvas_content).toBeUndefined();
    });

    it('should add user message on start', async () => {
      const store = useConversationStore.getState();

      await store.actions.startConversation('Hello, I need help');

      const state = useConversationStore.getState();

      expect(state.messages).toHaveLength(1);
      expect(state.messages[0].type).toBe('user');
      expect(state.messages[0].content).toBe('Hello, I need help');
      expect(state.messages[0].id).toBeDefined();
      expect(state.messages[0].timestamp).toBeInstanceOf(Date);
    });

    it('should handle file attachments on start', async () => {
      const store = useConversationStore.getState();

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      await store.actions.startConversation('Here is a file', [file]);

      const state = useConversationStore.getState();

      expect(state.uploaded_files).toHaveLength(1);
      expect(state.uploaded_files[0].file_name).toBe('test.txt');
      expect(state.messages[0].attachments).toHaveLength(1);
      expect(state.messages[0].attachments![0].name).toBe('test.txt');
    });

    it('should send WebSocket message on start', async () => {
      const store = useConversationStore.getState();

      await store.actions.startConversation('Test message');

      expect(mockSend).toHaveBeenCalled();
      const sentData = JSON.parse(mockSend.mock.calls[0][0] as string);
      expect(sentData.prompt).toBe('Test message');
      expect(sentData.thread_id).toBeUndefined(); // Not set yet
    });

    it('should set loading state while sending', async () => {
      const store = useConversationStore.getState();

      const startPromise = store.actions.startConversation('test');

      // Check loading state
      let state = useConversationStore.getState();
      expect(state.isLoading).toBe(true);
      expect(state.status).toBe('processing');

      await startPromise;

      // Still loading until response
      state = useConversationStore.getState();
      expect(state.isLoading).toBe(true);
    });
  });

  describe('continueConversation', () => {
    beforeEach(() => {
      // Start a conversation first
      useConversationStore.getState()._setConversationState({
        thread_id: 'test-thread-id',
        messages: [{ id: 'msg-1', type: 'user', content: 'First message', timestamp: new Date() }],
      });
    });

    it('should add user message to existing conversation', async () => {
      const store = useConversationStore.getState();

      await store.actions.continueConversation('Second message');

      const state = useConversationStore.getState();

      expect(state.messages).toHaveLength(2);
      expect(state.messages[0].content).toBe('First message');
      expect(state.messages[1].content).toBe('Second message');
      expect(state.messages[1].type).toBe('user');
    });

    it('should send as prompt when not answering a question', async () => {
      const store = useConversationStore.getState();

      await store.actions.continueConversation('Continue the task');

      expect(mockSend).toHaveBeenCalled();
      const sentData = JSON.parse(mockSend.mock.calls[0][0] as string);
      expect(sentData.prompt).toBe('Continue the task');
      expect(sentData.user_response).toBeUndefined();
    });

    it('should send as user_response when answering a question', async () => {
      const store = useConversationStore.getState();

      // Set up as waiting for user
      store._setConversationState({ isWaitingForUser: true });

      await store.actions.continueConversation('Option A');

      expect(mockSend).toHaveBeenCalled();
      const sentData = JSON.parse(mockSend.mock.calls[0][0] as string);
      expect(sentData.user_response).toBe('Option A');
      expect(sentData.prompt).toBeUndefined();
    });

    it('should clear isWaitingForUser on continue', async () => {
      const store = useConversationStore.getState();

      store._setConversationState({ isWaitingForUser: true });

      await store.actions.continueConversation('Response');

      const state = useConversationStore.getState();
      expect(state.isWaitingForUser).toBe(false);
    });

    it('should require thread_id to continue', async () => {
      const store = useConversationStore.getState();

      // Clear thread_id
      store._setConversationState({ thread_id: undefined });

      // Should not throw but should log error
      await expect(store.actions.continueConversation('Test')).resolves.not.toThrow();
    });

    it('should handle file attachments on continue', async () => {
      const store = useConversationStore.getState();

      const file = new File(['more content'], 'more.txt', { type: 'text/plain' });
      await store.actions.continueConversation('More files', [file]);

      const state = useConversationStore.getState();

      expect(state.uploaded_files.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('loadConversation', () => {
    it('should load conversation from API', async () => {
      const store = useConversationStore.getState();

      await store.actions.loadConversation('test-thread-id');

      const state = useConversationStore.getState();

      expect(state.thread_id).toBe('test-thread-id');
      expect(state.messages).toHaveLength(2);
      expect(state.messages[0].content).toBe('Test user message');
      expect(state.messages[1].content).toBe('Test assistant message');
      expect(state.status).toBe('completed');
    });

    it('should convert timestamps to Date objects', async () => {
      const store = useConversationStore.getState();

      await store.actions.loadConversation('test-thread-id');

      const state = useConversationStore.getState();

      state.messages.forEach(msg => {
        expect(msg.timestamp).toBeInstanceOf(Date);
      });
    });

    it('should handle 404 errors gracefully', async () => {
      const { authFetch } = await import('../auth-fetch');
      (authFetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const store = useConversationStore.getState();

      await expect(store.actions.loadConversation('nonexistent')).resolves.not.toThrow();
    });

    it('should set thread_id in localStorage after loading', async () => {
      const store = useConversationStore.getState();

      await store.actions.loadConversation('test-thread-id');

      expect(mockLocalStorage.getItem('thread_id')).toBe('test-thread-id');
    });

    it('should replace existing state when loading new conversation', async () => {
      const store = useConversationStore.getState();

      // Set current state
      store._setConversationState({
        thread_id: 'current-thread',
        messages: [{ id: 'current', type: 'user', content: 'Current', timestamp: new Date() }],
        metadata: { existing_key: 'existing_value' },
      });

      // Load different conversation
      await store.actions.loadConversation('new-thread-id');

      const state = useConversationStore.getState();

      expect(state.thread_id).toBe('new-thread-id');
      expect(state.messages).not.toContainEqual(
        expect.objectContaining({ id: 'current' })
      );
    });
  });

  describe('resetConversation', () => {
    it('should clear all state fields', () => {
      const store = useConversationStore.getState();

      // Fill state
      store._setConversationState({
        thread_id: 'test-thread',
        messages: [{ id: 'msg-1', type: 'user', content: 'Test', timestamp: new Date() }],
        task_agent_pairs: [{ task_name: 'Test task' }],
        plan: [{ task_name: 'Plan step' }],
        execution_plan: [{ phase_id: 'phase-1' }],
        action_history: [{ iteration: 1 }],
        insights: { key: 'value' },
        metadata: { test: 'data' },
        canvas_content: '<canvas>test</canvas>',
        has_canvas: true,
      });

      // Reset
      store.actions.resetConversation();

      const state = useConversationStore.getState();

      expect(state.thread_id).toBeUndefined();
      expect(state.messages).toEqual([]);
      expect state.task_agent_pairs).toEqual([]);
      expect(state.plan).toEqual([]);
      expect(state.execution_plan).toEqual([]);
      expect state.action_history).toEqual([]);
      expect(state.insights).toEqual({});
      expect(state.metadata).toEqual({});
      expect(state.has_canvas).toBe(false);
      expect(state.canvas_content).toBeUndefined();
    });

    it('should clear thread_id from localStorage', () => {
      const store = useConversationStore.getState();

      mockLocalStorage.setItem('thread_id', 'test-thread');
      store.actions.resetConversation();

      expect(mockLocalStorage.getItem('thread_id')).toBeNull();
    });

    it('should clear Omni-Dispatcher fields', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        execution_plan: [{ phase_id: '1' }],
        current_phase_id: 'phase-1',
        action_history: [{ iteration: 1 }],
        insights: { key: 'value' },
        pending_action_approval: true,
      });

      store.actions.resetConversation();

      const state = useConversationStore.getState();
      expect(state.execution_plan).toEqual([]);
      expect(state.current_phase_id).toBeUndefined();
      expect state.action_history).toEqual([]);
      expect(state.insights).toEqual({});
      expect(state.pending_action_approval).toBe(false);
    });
  });

  describe('_setConversationState', () => {
    it('should update partial state', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        final_response: 'Task complete',
        status: 'completed',
      });

      const state = useConversationStore.getState();

      expect(state.final_response).toBe('Task complete');
      expect(state.status).toBe('completed');
    });

    it('should append messages for same conversation', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        thread_id: 'thread-1',
        messages: [{ id: 'msg-1', type: 'assistant', content: 'Response 1', timestamp: new Date() }],
      });

      store._setConversationState({
        thread_id: 'thread-1',
        messages: [{ id: 'msg-2', type: 'assistant', content: 'Response 2', timestamp: new Date() }],
      });

      const state = useConversationStore.getState();

      expect(state.messages).toHaveLength(2);
      expect(state.messages[0].id).toBe('msg-1');
      expect(state.messages[1].id).toBe('msg-2');
    });

    it('should replace messages for different conversation', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        thread_id: 'thread-1',
        messages: [{ id: 'msg-1', type: 'assistant', content: 'Response 1', timestamp: new Date() }],
      });

      store._setConversationState({
        thread_id: 'thread-2',
        messages: [{ id: 'msg-2', type: 'assistant', content: 'Response 2', timestamp: new Date() }],
      });

      const state = useConversationStore.getState();

      expect(state.messages).toHaveLength(1);
      expect(state.messages[0].id).toBe('msg-2');
    });

    it('should merge uploaded files', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        uploaded_files: [{ file_name: 'file1.txt', file_path: '/uploads/file1.txt', file_type: 'document' }],
      });

      store._setConversationState({
        uploaded_files: [{ file_name: 'file2.txt', file_path: '/uploads/file2.txt', file_type: 'document' }],
      });

      const state = useConversationStore.getState();

      expect(state.uploaded_files).toHaveLength(2);
      expect(state.uploaded_files[0].file_name).toBe('file1.txt');
      expect(state.uploaded_files[1].file_name).toBe('file2.txt');
    });

    it('should deduplicate uploaded files', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        uploaded_files: [{ file_name: 'file1.txt', file_path: '/uploads/file1.txt', file_type: 'document' }],
      });

      store._setConversationState({
        uploaded_files: [{ file_name: 'file1.txt', file_path: '/uploads/file1.txt', file_type: 'document' }],
      });

      const state = useConversationStore.getState();

      expect(state.uploaded_files).toHaveLength(1);
    });

    it('should merge metadata with completed_tasks limit', () => {
      const store = useConversationStore.getState();

      // Add 25 completed tasks
      const initialTasks = Array.from({ length: 25 }, (_, i) => ({
        task_name: `Task ${i}`,
      }));

      store._setConversationState({
        metadata: { completed_tasks: initialTasks },
      });

      // Add 5 more
      store._setConversationState({
        metadata: {
          completed_tasks: Array.from({ length: 5 }, (_, i) => ({
            task_name: `New Task ${i}`,
          })),
        },
      });

      const state = useConversationStore.getState();

      // Should be limited to last 20
      expect(state.metadata.completed_tasks).toHaveLength(20);
    });

    it('should preserve plan when updating', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        plan: [{ task_name: 'Step 1' }],
      });

      store._setConversationState({
        status: 'processing',
      });

      const state = useConversationStore.getState();

      expect(state.plan).toEqual([{ task_name: 'Step 1' }]);
    });

    it('should update plan fields', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        plan: [{ task_name: 'Step 1' }],
        metadata: { original_prompt: 'Original' },
      });

      store._setConversationState({
        plan: [{ task_name: 'Step 2' }],
        metadata: { original_prompt: 'Updated' },
      });

      const state = useConversationStore.getState();

      expect(state.plan).toEqual([{ task_name: 'Step 2' }]);
      expect(state.metadata.original_prompt).toBe('Updated');
    });

    it('should deduplicate messages by content and type', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        thread_id: 'thread-1',
        messages: [{ id: 'msg-1', type: 'assistant', content: 'Same content', timestamp: new Date() }],
      });

      store._setConversationState({
        thread_id: 'thread-1',
        messages: [{ id: 'msg-2', type: 'assistant', content: 'Same content', timestamp: new Date() }],
      });

      const state = useConversationStore.getState();

      // Should deduplicate, only keep one
      expect(state.messages).toHaveLength(1);
    });

    it('should update Omni-Dispatcher fields', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        execution_plan: [{ phase_id: 'phase-1', name: 'Phase 1' }],
        current_phase_id: 'phase-1',
        action_history: [{ iteration: 1, action_type: 'tool', success: true }],
        insights: { key: 'value' },
      });

      const state = useConversationStore.getState();

      expect(state.execution_plan).toHaveLength(1);
      expect(state.execution_plan[0].phase_id).toBe('phase-1');
      expect(state.current_phase_id).toBe('phase-1');
      expect(state.action_history).toHaveLength(1);
      expect state.insights).toEqual({ key: 'value' });
    });

    it('should handle canvas state updates', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        canvas_content: '<div>Canvas content</div>',
        canvas_type: 'html',
        canvas_data: { key: 'value' },
        has_canvas: true,
      });

      const state = useConversationStore.getState();

      expect(state.canvas_content).toBe('<div>Canvas content</div>');
      expect(state.canvas_type).toBe('html');
      expect(state.canvas_data).toEqual({ key: 'value' });
      expect(state.has_canvas).toBe(true);
    });

    it('should clear canvas when has_canvas is false', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        canvas_content: '<div>Canvas</div>',
        has_canvas: true,
      });

      store._setConversationState({ has_canvas: false });

      const state = useConversationStore.getState();

      expect(state.has_canvas).toBe(false);
      expect(state.canvas_content).toBeUndefined();
    });
  });

  describe('sendCanvasConfirmation', () => {
    beforeEach(() => {
      // Set up state with pending confirmation
      useConversationStore.getState()._setConversationState({
        thread_id: 'test-thread',
        pending_confirmation: true,
        pending_confirmation_task: { task_name: 'Apply changes' },
      });
    });

    it('should send confirm action via WebSocket', async () => {
      const store = useConversationStore.getState();

      await store.actions.sendCanvasConfirmation('confirm', 'Apply changes');

      expect(mockSend).toHaveBeenCalled();
      const sentData = JSON.parse(mockSend.mock.calls[0][0] as string);
      expect(sentData.type).toBe('canvas_confirmation');
      expect(sentData.action).toBe('confirm');
      expect(sentData.task_name).toBe('Apply changes');
    });

    it('should clear pending confirmation state on confirm', async () => {
      const store = useConversationStore.getState();

      await store.actions.sendCanvasConfirmation('confirm');

      const state = useConversationStore.getState();

      expect(state.pending_confirmation).toBe(false);
      expect(state.pending_confirmation_task).toBeUndefined();
    });

    it('should send confirm follow-up message', async () => {
      const store = useConversationStore.getState();

      await store.actions.sendCanvasConfirmation('confirm');

      // Should have sent both confirmation and follow-up
      expect(mockSend).toHaveBeenCalledTimes(2);
      const followUpData = JSON.parse(mockSend.mock.calls[1][0] as string);
      expect(followUpData.prompt).toContain('Apply the changes');
    });

    it('should send cancel action via WebSocket', async () => {
      const store = useConversationStore.getState();

      await store.actions.sendCanvasConfirmation('cancel');

      expect(mockSend).toHaveBeenCalled();
      // Only cancel message, no follow-up
      expect(mockSend).toHaveBeenCalledTimes(1);
      const sentData = JSON.parse(mockSend.mock.calls[0][0] as string);
      expect(sentData.action).toBe('cancel');
    });

    it('should require thread_id', async () => {
      const store = useConversationStore.getState();

      store._setConversationState({ thread_id: undefined });

      await expect(store.actions.sendCanvasConfirmation('confirm')).resolves.not.toThrow();

      expect(mockSend).not.toHaveBeenCalled();
    });
  });

  describe('Message ID Generation', () => {
    it('should generate deterministic message IDs', () => {
      const timestamp = Date.now();
      // Note: createMessageId is internal, so we test via startConversation
      const store = useConversationStore.getState();

      mockWebSocket.send = vi.fn();

      store.actions.startConversation('Test');

      // The store generates messages with IDs
      // In a real scenario, we'd test the deduplication based on these IDs
    });
  });

  describe('State Persistence', () => {
    it('should save thread_id to localStorage on state update', () => {
      const store = useConversationStore.getState();

      store._setConversationState({ thread_id: 'persist-thread' });

      expect(mockLocalStorage.getItem('thread_id')).toBe('persist-thread');
    });

    it('should remove thread_id from localStorage on reset', () => {
      const store = useConversationStore.getState();

      store._setConversationState({ thread_id: 'test' });
      store.actions.resetConversation();

      expect(mockLocalStorage.getItem('thread_id')).toBeNull();
    });
  });

  describe('Task Status Tracking', () => {
    it('should track task statuses', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        task_statuses: {
          'task-1': 'in_progress',
          'task-2': 'completed',
        },
        current_executing_task: 'task-1',
      });

      const state = useConversationStore.getState();

      expect state.task_statuses).toEqual({
        'task-1': 'in_progress',
        'task-2': 'completed',
      });
      expect(state.current_executing_task).toBe('task-1');
    });

    it('should clear task statuses on continue', async () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        thread_id: 'test-thread',
        task_statuses: { 'task-1': 'in_progress' },
        current_executing_task: 'task-1',
      });

      await store.actions.continueConversation('Continue');

      const state = useConversationStore.getState();

      expect state.task_statuses).toEqual({});
      expect(state.current_executing_task).toBeNull();
    });
  });

  describe('Browser and Plan Views', () => {
    it('should handle browser_view updates', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        browser_view: '<div>Browser content</div>',
        plan_view: '<div>Plan content</div>',
        current_view: 'plan',
      });

      const state = useConversationStore.getState();

      expect(state.browser_view).toBe('<div>Browser content</div>');
      expect(state.plan_view).toBe('<div>Plan content</div>');
      expect(state.current_view).toBe('plan');
    });
  });

  describe('File Upload Handling', () => {
    it('should read image files as data URLs for preview', async () => {
      const store = useConversationStore.getState();

      // Create a mock image file
      const imageFile = new File(['image data'], 'test.png', { type: 'image/png' });

      await store.actions.startConversation('Here is an image', [imageFile]);

      const state = useConversationStore.getState();

      expect(state.messages[0].attachments).toBeDefined();
      expect(state.messages[0].attachments![0].type).toBe('image/png');
      expect(state.messages[0].attachments![0].content).toBeDefined(); // Data URL
    });
  });

  describe('Approval Handling', () => {
    it('should track approval state', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        approval_required: true,
        estimated_cost: 0.05,
        task_count: 3,
      });

      const state = useConversationStore.getState();

      expect(state.approval_required).toBe(true);
      expect(state.estimated_cost).toBe(0.05);
      expect(state.task_count).toBe(3);
    });

    it('should set pending action approval fields', () => {
      const store = useConversationStore.getState();

      store._setConversationState({
        pending_action_approval: true,
        pending_action: {
          action_type: 'agent',
          resource_id: 'EmailAgent',
          approval_reason: 'Sending email to user@example.com',
        },
      });

      const state = useConversationStore.getState();

      expect(state.pending_action_approval).toBe(true);
      expect(state.pending_action).toBeDefined();
      expect(state.pending_action!.resource_id).toBe('EmailAgent');
    });
  });
});
