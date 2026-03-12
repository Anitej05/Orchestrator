import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import ConversationsDropdown from '@/components/conversations-dropdown';

// ── Strategy ───────────────────────────────────────────────────────────────────
//
// ConversationsDropdown uses `await import('@/lib/auth-fetch')` (dynamic import).
// vi.mock() cannot reliably intercept path-aliased dynamic imports.
// Instead we stub window.fetch directly — the underlying call authFetch makes —
// and provide a minimal window.Clerk mock so hasClerkSession() returns true.
//
// ──────────────────────────────────────────────────────────────────────────────

const mockFetch = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('fetch', mockFetch);
  // hasClerkSession() requires window.Clerk.session to be truthy.
  // authFetch calls session.getToken() to obtain a JWT.
  (window as any).Clerk = {
    session: {
      id: 'test-session',
      getToken: vi.fn().mockResolvedValue('test-token'),
    },
  };
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete (window as any).Clerk;
});

// ── Helpers ────────────────────────────────────────────────────────────────────

const makeConvResponse = (items: any[] = []) =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(items),
  });

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('ConversationsDropdown', () => {
  it('renders "Loading conversations…" while fetch is pending', () => {
    mockFetch.mockReturnValue(new Promise(() => {})); // never resolves
    render(
      <ConversationsDropdown
        onConversationSelect={vi.fn()}
        onNewConversation={vi.fn()}
      />
    );
    expect(screen.getByText(/Loading conversations/)).toBeTruthy();
  });

  it('renders conversation titles after successful fetch', async () => {
    mockFetch.mockReturnValue(
      makeConvResponse([
        { thread_id: 'tid-1', title: 'First Chat', created_at: '2024-01-01T00:00:00Z' },
        { thread_id: 'tid-2', title: 'Second Chat', created_at: '2024-01-02T00:00:00Z' },
      ])
    );
    render(
      <ConversationsDropdown
        onConversationSelect={vi.fn()}
        onNewConversation={vi.fn()}
      />
    );
    await waitFor(() => {
      expect(screen.getByText('First Chat')).toBeTruthy();
      expect(screen.getByText('Second Chat')).toBeTruthy();
    });
  });

  it('renders error message when fetch fails with a network error', async () => {
    // authFetch re-throws as "Network request failed for ...: Failed to fetch"
    // which contains "fetch", "Network", and "Failed to fetch" → component shows unable-to-connect msg
    mockFetch.mockRejectedValue(new Error('Failed to fetch'));
    render(
      <ConversationsDropdown
        onConversationSelect={vi.fn()}
        onNewConversation={vi.fn()}
      />
    );
    await waitFor(() => {
      expect(screen.getByText(/Unable to connect to server/)).toBeTruthy();
    });
  });

  it('renders "No conversations yet" for empty list', async () => {
    mockFetch.mockReturnValue(makeConvResponse([]));
    render(
      <ConversationsDropdown
        onConversationSelect={vi.fn()}
        onNewConversation={vi.fn()}
      />
    );
    await waitFor(() => {
      expect(screen.getByText('No conversations yet')).toBeTruthy();
    });
  });

  it('calls onConversationSelect with the correct thread_id on click', async () => {
    const onSelect = vi.fn();
    mockFetch.mockReturnValue(
      makeConvResponse([
        { thread_id: 'tid-42', title: 'Click Me', created_at: '2024-01-01T00:00:00Z' },
      ])
    );
    render(
      <ConversationsDropdown
        onConversationSelect={onSelect}
        onNewConversation={vi.fn()}
      />
    );
    await waitFor(() => screen.getByText('Click Me'));
    fireEvent.click(screen.getByText('Click Me'));
    expect(onSelect).toHaveBeenCalledWith('tid-42');
  });

  it('calls onNewConversation when "New Conversation" button is clicked', async () => {
    const onNew = vi.fn();
    mockFetch.mockReturnValue(makeConvResponse([]));
    render(
      <ConversationsDropdown
        onConversationSelect={vi.fn()}
        onNewConversation={onNew}
      />
    );
    await waitFor(() => screen.getByText('New Conversation'));
    fireEvent.click(screen.getByText('New Conversation'));
    expect(onNew).toHaveBeenCalled();
  });

  it('handles 404 response gracefully with empty list', async () => {
    mockFetch.mockReturnValue(
      Promise.resolve({ ok: false, status: 404, statusText: 'Not Found', json: () => Promise.resolve({}) })
    );
    render(
      <ConversationsDropdown
        onConversationSelect={vi.fn()}
        onNewConversation={vi.fn()}
      />
    );
    await waitFor(() => {
      expect(screen.getByText('No conversations yet')).toBeTruthy();
    });
  });
});
