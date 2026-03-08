import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useWebSocketManager } from '@/hooks/use-websocket-conversation';
import { useConversationStore } from '@/lib/conversation-store';

// ── WebSocket mock ─────────────────────────────────────────────────────────────

class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  static CONNECTING = 0;
  static CLOSING = 2;

  url: string;
  readyState: number = MockWebSocket.OPEN;
  onopen: ((e: any) => void) | null = null;
  onclose: ((e: any) => void) | null = null;
  onmessage: ((e: any) => void) | null = null;
  onerror: ((e: any) => void) | null = null;
  sentMessages: string[] = [];

  constructor(url: string) {
    this.url = url;
    // Simulate connection opening on next tick
    setTimeout(() => this.onopen?.({ type: 'open' }), 0);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close(code = 1000) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code, wasClean: code === 1000, reason: '' });
  }

  // Test helper to simulate receiving a message
  receive(data: Record<string, any>) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

let mockWsInstance: MockWebSocket;

// ── Store mock ─────────────────────────────────────────────────────────────────

const mockSetConversationState = vi.hoisted(() => vi.fn());

vi.mock('@/lib/conversation-store', () => {
  const setState = vi.fn();
  const storeHook = Object.assign(
    vi.fn((selector: any) => {
      if (typeof selector === 'function') {
        return selector({ actions: { _setConversationState: mockSetConversationState } });
      }
      return {};
    }),
    {
      getState: vi.fn().mockReturnValue({
        status: 'idle',
        task_statuses: {},
        messages: [],
        current_executing_task: null,
        metadata: {},
      }),
      setState,
      subscribe: vi.fn(),
    }
  );
  return { useConversationStore: storeHook };
});

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('useWebSocketManager', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Re-stub WebSocket each test using a regular function (not arrow — arrow functions
    // cannot be used as constructors, which Vitest may attempt internally).
    const WsMockConstructor = vi.fn(function (this: any, url: string) {
      const ws = new MockWebSocket(url);
      mockWsInstance = ws;
      return ws;
    }) as any;
    WsMockConstructor.OPEN = MockWebSocket.OPEN;
    WsMockConstructor.CLOSED = MockWebSocket.CLOSED;
    WsMockConstructor.CONNECTING = MockWebSocket.CONNECTING;
    WsMockConstructor.CLOSING = MockWebSocket.CLOSING;
    vi.stubGlobal('WebSocket', WsMockConstructor);

    (useConversationStore as any).getState.mockReturnValue({
      status: 'idle',
      task_statuses: {},
      messages: [],
      current_executing_task: null,
      metadata: {},
    });
  });

  it('connects to the provided WebSocket URL on mount', async () => {
    renderHook(() => useWebSocketManager({ url: 'ws://test-host/ws/chat' }));
    await act(async () => {});
    expect(vi.mocked(WebSocket)).toHaveBeenCalledWith('ws://test-host/ws/chat');
  });

  it('returns isConnected=true after connection opens', async () => {
    const { result } = renderHook(() =>
      useWebSocketManager({ url: 'ws://localhost:8000/ws/chat' })
    );
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
    expect(result.current.isConnected).toBe(true);
  });

  it('routes todo_list_update message → calls _setConversationState with todo_list', async () => {
    const mockSetState = vi.fn();
    vi.mocked(useConversationStore).mockImplementation((selector: any) =>
      selector({ actions: { _setConversationState: mockSetState } })
    );

    renderHook(() => useWebSocketManager({ url: 'ws://localhost:8000/ws/chat' }));
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });

    act(() => {
      mockWsInstance.receive({
        node: 'todo_list_update',
        thread_id: 'thread-1',
        data: {
          todo_list: [
            { task_id: 'task-1', description: 'Do A', status: 'in_progress' },
            { task_id: 'task-2', description: 'Do B', status: 'completed' },
          ],
          current_task_id: 'task-1',
        },
      });
    });

    expect(mockSetState).toHaveBeenCalledWith(
      expect.objectContaining({
        todo_list: expect.arrayContaining([
          expect.objectContaining({ task_id: 'task-1' }),
        ]),
        task_statuses: expect.any(Object),
      })
    );
  });

  it('maps in_progress → running in task_statuses', async () => {
    const mockSetState = vi.fn();
    vi.mocked(useConversationStore).mockImplementation((selector: any) =>
      selector({ actions: { _setConversationState: mockSetState } })
    );

    renderHook(() => useWebSocketManager({ url: 'ws://localhost:8000/ws/chat' }));
    await act(async () => await new Promise((r) => setTimeout(r, 10)));

    act(() => {
      mockWsInstance.receive({
        node: 'todo_list_update',
        thread_id: 'thread-1',
        data: {
          todo_list: [{ task_id: 't1', description: 'Task A', status: 'in_progress' }],
        },
      });
    });

    const call = mockSetState.mock.calls[0]?.[0];
    expect(call?.task_statuses?.['t1']?.status).toBe('running');
  });

  it('preserves completed status in task_statuses', async () => {
    const mockSetState = vi.fn();
    vi.mocked(useConversationStore).mockImplementation((selector: any) =>
      selector({ actions: { _setConversationState: mockSetState } })
    );

    renderHook(() => useWebSocketManager({ url: 'ws://localhost:8000/ws/chat' }));
    await act(async () => await new Promise((r) => setTimeout(r, 10)));

    act(() => {
      mockWsInstance.receive({
        node: 'todo_list_update',
        thread_id: 'thread-1',
        data: {
          todo_list: [{ task_id: 't2', description: 'Task B', status: 'completed' }],
        },
      });
    });

    const call = mockSetState.mock.calls[0]?.[0];
    expect(call?.task_statuses?.['t2']?.status).toBe('completed');
  });

  it('routes __start__ message → sets status: processing', async () => {
    const mockSetState = vi.fn();
    vi.mocked(useConversationStore).mockImplementation((selector: any) =>
      selector({ actions: { _setConversationState: mockSetState } })
    );

    renderHook(() => useWebSocketManager({ url: 'ws://localhost:8000/ws/chat' }));
    await act(async () => await new Promise((r) => setTimeout(r, 10)));

    act(() => {
      mockWsInstance.receive({
        node: '__start__',
        thread_id: 'thread-1',
        data: {},
      });
    });

    const processingCall = mockSetState.mock.calls.find(
      (c) => c[0]?.status === 'processing'
    );
    expect(processingCall).toBeTruthy();
  });

  it('attempts reconnect after unexpected close (code !== 1000)', async () => {
    vi.useFakeTimers();
    (useConversationStore as any).getState.mockReturnValue({ status: 'idle' });

    renderHook(() => useWebSocketManager({ url: 'ws://localhost:8000/ws/chat' }));

    // Advance fake timers to fire the MockWebSocket onopen setTimeout(0)
    await act(async () => { vi.advanceTimersByTime(10); });

    const firstInstance = mockWsInstance;

    act(() => {
      firstInstance.close(1006); // abnormal closure
    });

    // Advance past the 2000ms reconnect delay inside the hook
    await act(async () => { vi.advanceTimersByTime(2500); });

    // A new WebSocket should have been created for reconnect
    expect(vi.mocked(WebSocket)).toHaveBeenCalledTimes(2);

    vi.useRealTimers();
  });
});
