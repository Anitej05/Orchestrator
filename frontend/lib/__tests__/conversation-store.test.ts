import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useConversationStore } from '@/lib/conversation-store';
import { CANVAS_TYPES } from '@/lib/types';
import type { Message } from '@/lib/types';

// ── Module mocks ──────────────────────────────────────────────────────────────

vi.mock('@/lib/auth-fetch', () => ({
  authFetch: vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
  }),
}));

vi.mock('@/lib/api-client', () => ({
  uploadFiles: vi.fn().mockResolvedValue([]),
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

const makeMessage = (overrides: Partial<Message> = {}): Message => ({
  id: 'msg-1',
  type: 'assistant',
  content: 'Hello',
  timestamp: new Date('2024-01-01T00:00:00Z'),
  ...overrides,
});

const RESET_STATE = {
  thread_id: undefined as string | undefined,
  status: 'idle' as const,
  messages: [] as Message[],
  isWaitingForUser: false,
  task_agent_pairs: [],
  final_response: undefined as string | undefined,
  metadata: {},
  uploaded_files: [],
  plan: [],
  todo_list: [],
  task_statuses: {},
  current_executing_task: null as string | null,
  canvas_content: undefined as string | undefined,
  canvas_type: undefined,
  has_canvas: false,
  canvas_data: undefined,
  isLoading: false,
  approval_required: false,
  pending_action_approval: false,
  pending_action: undefined,
  pending_confirmation: false,
  pending_confirmation_task: undefined,
  canvas_requires_confirmation: false,
  canvas_confirmation_message: undefined,
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('CANVAS_TYPES constant', () => {
  it('contains all expected canvas type values', () => {
    const expected = ['html', 'markdown', 'pdf', 'spreadsheet', 'email_preview', 'document', 'image', 'json'];
    expected.forEach(type => expect(CANVAS_TYPES).toContain(type));
  });

  it('is a readonly tuple (no duplicates)', () => {
    const unique = new Set(CANVAS_TYPES);
    expect(unique.size).toBe(CANVAS_TYPES.length);
  });
});

describe('conversation-store — _setConversationState', () => {
  beforeEach(() => {
    useConversationStore.setState(RESET_STATE);
    vi.clearAllMocks();
  });

  const setConvState = () => useConversationStore.getState().actions._setConversationState;

  // ── Message deduplication ──────────────────────────────────────────────────

  describe('message deduplication (same thread)', () => {
    it('does not add a backend message that already exists by ID', () => {
      const msg = makeMessage({ id: 'msg-1', content: 'Hello' });
      useConversationStore.setState({ thread_id: 'thread-1', messages: [msg] });

      setConvState()({ thread_id: 'thread-1', messages: [msg] });

      expect(useConversationStore.getState().messages).toHaveLength(1);
    });

    it('adds new messages from backend that do not yet exist', () => {
      const existing = makeMessage({ id: 'msg-1', content: 'First' });
      const incoming = makeMessage({ id: 'msg-2', content: 'Second', type: 'user' });
      useConversationStore.setState({ thread_id: 'thread-1', messages: [existing] });

      setConvState()({ thread_id: 'thread-1', messages: [existing, incoming] });

      expect(useConversationStore.getState().messages).toHaveLength(2);
    });

    it('deduplicates by content+type when IDs differ (hash mismatch fallback)', () => {
      const frontendMsg = makeMessage({ id: 'frontend-hash', content: 'Same content', type: 'assistant' });
      const backendMsg  = makeMessage({ id: 'backend-hash',  content: 'Same content', type: 'assistant' });
      useConversationStore.setState({ thread_id: 'thread-1', messages: [frontendMsg] });

      setConvState()({ thread_id: 'thread-1', messages: [backendMsg] });

      expect(useConversationStore.getState().messages).toHaveLength(1);
    });

    it('removes duplicate IDs already present in state (final dedup pass)', () => {
      const msg = makeMessage({ id: 'dup-id' });
      // Artificially inject a state with duplicate IDs
      useConversationStore.setState({ thread_id: 'thread-1', messages: [msg, { ...msg }] });

      // Trigger _setConversationState with empty backend messages — the final pass cleans duplicates
      setConvState()({ thread_id: 'thread-1', messages: [] });

      expect(useConversationStore.getState().messages).toHaveLength(1);
    });

    it('filters out assistant messages with empty content', () => {
      useConversationStore.setState({ thread_id: 'thread-1', messages: [] });
      const emptyMsg = makeMessage({ id: 'empty-1', type: 'assistant', content: '' });

      setConvState()({ thread_id: 'thread-1', messages: [emptyMsg] });

      // Same-thread merge: empty assistant messages are skipped
      expect(useConversationStore.getState().messages).toHaveLength(0);
    });

    it('keeps messages in chronological order after merge', () => {
      const older = makeMessage({ id: 'old', timestamp: new Date('2024-01-01T10:00:00Z'), content: 'Older' });
      const newer = makeMessage({ id: 'new', timestamp: new Date('2024-01-01T12:00:00Z'), content: 'Newer', type: 'user' });
      useConversationStore.setState({ thread_id: 'thread-1', messages: [newer] });

      setConvState()({ thread_id: 'thread-1', messages: [older, newer] });

      const msgs = useConversationStore.getState().messages;
      expect(msgs[0].id).toBe('old');
      expect(msgs[1].id).toBe('new');
    });
  });

  // ── New conversation loading ───────────────────────────────────────────────

  describe('loading a new conversation (different thread_id)', () => {
    it('replaces messages instead of merging', () => {
      const oldMsg = makeMessage({ id: 'old-1', content: 'Old' });
      useConversationStore.setState({ thread_id: 'thread-1', messages: [oldMsg] });

      const newMsg = makeMessage({ id: 'new-1', content: 'New' });
      setConvState()({ thread_id: 'thread-2', messages: [newMsg] });

      const msgs = useConversationStore.getState().messages;
      expect(msgs).toHaveLength(1);
      expect(msgs[0].id).toBe('new-1');
    });

    it('replaces uploaded_files', () => {
      useConversationStore.setState({
        thread_id: 'thread-1',
        uploaded_files: [{ file_name: 'old.pdf', file_path: '/old.pdf', file_type: 'application/pdf' }],
      });

      setConvState()({
        thread_id: 'thread-2',
        uploaded_files: [{ file_name: 'new.xlsx', file_path: '/new.xlsx', file_type: 'application/vnd.ms-excel' }],
      });

      const files = useConversationStore.getState().uploaded_files!;
      expect(files).toHaveLength(1);
      expect(files[0].file_name).toBe('new.xlsx');
    });

    it('replaces task_agent_pairs', () => {
      const oldPair = { task_name: 'old-task', task_description: '', primary: { id: 'a1', name: 'OldAgent', description: '', capabilities: [], status: 'active' as const, endpoints: [] }, fallbacks: [] };
      const newPair = { task_name: 'new-task', task_description: '', primary: { id: 'a2', name: 'NewAgent', description: '', capabilities: [], status: 'active' as const, endpoints: [] }, fallbacks: [] };
      useConversationStore.setState({ thread_id: 'thread-1', task_agent_pairs: [oldPair] });

      setConvState()({ thread_id: 'thread-2', task_agent_pairs: [newPair] });

      const pairs = useConversationStore.getState().task_agent_pairs!;
      expect(pairs).toHaveLength(1);
      expect(pairs[0].task_name).toBe('new-task');
    });
  });

  // ── Canvas state ──────────────────────────────────────────────────────────

  describe('canvas state management', () => {
    it('clears all canvas fields when has_canvas is set to false', () => {
      useConversationStore.setState({
        thread_id: 'thread-1',
        has_canvas: true,
        canvas_content: '<h1>Chart</h1>',
        canvas_type: 'html',
        canvas_data: { rows: [] },
      });

      setConvState()({ thread_id: 'thread-1', has_canvas: false });

      const state = useConversationStore.getState();
      expect(state.has_canvas).toBe(false);
      expect(state.canvas_content).toBeUndefined();
      expect(state.canvas_data).toBeUndefined();
      expect(state.canvas_type).toBeUndefined();
    });

    it('preserves canvas when the update does not touch canvas fields', () => {
      useConversationStore.setState({
        thread_id: 'thread-1',
        has_canvas: true,
        canvas_content: '<p>Keep me</p>',
        canvas_type: 'html',
      });

      setConvState()({ thread_id: 'thread-1', status: 'completed' });

      const state = useConversationStore.getState();
      expect(state.canvas_content).toBe('<p>Keep me</p>');
      expect(state.canvas_type).toBe('html');
      expect(state.has_canvas).toBe(true);
    });

    it('sets new canvas content when provided', () => {
      useConversationStore.setState({ thread_id: 'thread-1' });

      setConvState()({
        thread_id: 'thread-1',
        has_canvas: true,
        canvas_content: '# Report',
        canvas_type: 'markdown',
      });

      const state = useConversationStore.getState();
      expect(state.canvas_content).toBe('# Report');
      expect(state.canvas_type).toBe('markdown');
      expect(state.has_canvas).toBe(true);
    });
  });

  // ── Plan logic ────────────────────────────────────────────────────────────

  describe('plan update logic', () => {
    it('replaces plan when loading a new thread', () => {
      useConversationStore.setState({ thread_id: 'thread-1', plan: [{ task: 'old' }] });

      setConvState()({ thread_id: 'thread-2', plan: [{ task: 'new' }] });

      const state = useConversationStore.getState();
      expect(state.plan).toHaveLength(1);
      expect((state.plan![0] as any).task).toBe('new');
    });

    it('replaces plan when new plan is non-empty on the same thread', () => {
      useConversationStore.setState({ thread_id: 'thread-1', plan: [{ task: 'old' }] });

      setConvState()({ thread_id: 'thread-1', plan: [{ task: 'updated' }] });

      expect((useConversationStore.getState().plan![0] as any).task).toBe('updated');
    });

    it('preserves existing plan when new plan is empty on the same thread', () => {
      useConversationStore.setState({ thread_id: 'thread-1', plan: [{ task: 'existing' }] });

      setConvState()({ thread_id: 'thread-1', plan: [] });

      expect(useConversationStore.getState().plan).toHaveLength(1);
    });
  });
});

describe('conversation-store — resetConversation', () => {
  beforeEach(() => {
    useConversationStore.setState(RESET_STATE);
    vi.clearAllMocks();
  });

  it('resets all state fields to their initial values', () => {
    useConversationStore.setState({
      thread_id: 'thread-1',
      status: 'completed',
      messages: [makeMessage()],
      isWaitingForUser: true,
      has_canvas: true,
      canvas_content: '<p>test</p>',
      canvas_type: 'html',
      plan: [{ task: 'something' }],
      task_statuses: { 'task-1': { status: 'running', taskName: 'Task 1' } },
      isLoading: true,
      approval_required: true,
    });

    useConversationStore.getState().actions.resetConversation();

    const state = useConversationStore.getState();
    expect(state.thread_id).toBeUndefined();
    expect(state.status).toBe('idle');
    expect(state.messages).toHaveLength(0);
    expect(state.isWaitingForUser).toBe(false);
    expect(state.has_canvas).toBe(false);
    expect(state.canvas_content).toBeUndefined();
    expect(state.plan).toHaveLength(0);
    expect(state.task_statuses).toEqual({});
    expect(state.isLoading).toBe(false);
  });

  it('clears thread_id from localStorage', () => {
    localStorage.setItem('thread_id', 'thread-1');
    useConversationStore.setState({ thread_id: 'thread-1' });

    useConversationStore.getState().actions.resetConversation();

    expect(localStorage.getItem('thread_id')).toBeNull();
  });
});
