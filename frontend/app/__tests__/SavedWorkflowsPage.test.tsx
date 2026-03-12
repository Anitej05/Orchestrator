import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import SavedWorkflowsPage from '@/app/saved-workflows/page';

// ── Strategy ───────────────────────────────────────────────────────────────────
//
// SavedWorkflowsPage uses `await import('@/lib/auth-fetch')` (dynamic import).
// vi.mock() cannot reliably intercept path-aliased dynamic imports.
// Instead we stub window.fetch — the underlying call authFetch makes.
//
// ──────────────────────────────────────────────────────────────────────────────

const mockFetch = vi.fn();
const mockPush = vi.fn();
const mockReplaceState = vi.fn();

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}));

vi.mock('@/components/ui/sidebar', () => ({
  SidebarInset: ({ children }: any) => <div>{children}</div>,
  SidebarTrigger: () => <button>Menu</button>,
  useSidebar: () => ({ open: true }),
}));

vi.mock('@/lib/conversation-store', () => ({
  useConversationStore: {
    getState: () => ({ actions: { loadConversation: vi.fn().mockResolvedValue(undefined) } }),
    setState: vi.fn(),
    subscribe: vi.fn(),
    getInitialState: vi.fn(),
  },
}));

vi.mock('sonner', () => ({
  toast: { info: vi.fn(), success: vi.fn(), error: vi.fn() },
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('fetch', mockFetch);
  // Provide Clerk so authFetch doesn't warn on every call
  (window as any).Clerk = {
    session: { id: 'test-session', getToken: vi.fn().mockResolvedValue('test-token') },
  };
  Object.defineProperty(window, 'history', {
    value: { replaceState: mockReplaceState },
    writable: true,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete (window as any).Clerk;
});

// ── Helpers ────────────────────────────────────────────────────────────────────

const makeWorkflow = (id: string, overrides: Record<string, any> = {}) => ({
  workflow_id: id,
  workflow_name: `Workflow ${id}`,
  workflow_description: `Description for ${id}`,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  task_count: 3,
  estimated_cost: 0.012,
  is_public: false,
  ...overrides,
});

// Returns a Response-like object (not wrapped in Promise — use mockResolvedValue / mockReturnValue)
const makeOkResponse = (body: any) => ({
  ok: true,
  status: 200,
  json: () => Promise.resolve(body),
  text: () => Promise.resolve(''),
});

const makeErrorFetchResponse = () => ({
  ok: false,
  status: 500,
  statusText: 'Server Error',
  json: () => Promise.resolve({}),
  text: () => Promise.resolve('error'),
});

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('SavedWorkflowsPage', () => {
  it('shows loading spinner during initial fetch', () => {
    mockFetch.mockReturnValue(new Promise(() => {}));
    render(<SavedWorkflowsPage />);
    expect(screen.getByText(/Loading workflows/i)).toBeTruthy();
  });

  it('renders workflow cards after successful fetch', async () => {
    mockFetch.mockResolvedValue(makeOkResponse([makeWorkflow('1'), makeWorkflow('2')]));
    render(<SavedWorkflowsPage />);
    await waitFor(() => {
      expect(screen.getByText('Workflow 1')).toBeTruthy();
      expect(screen.getByText('Workflow 2')).toBeTruthy();
    });
  });

  it('shows error message + Retry button on fetch failure', async () => {
    mockFetch.mockResolvedValue(makeErrorFetchResponse());
    render(<SavedWorkflowsPage />);
    await waitFor(() => {
      expect(screen.getByText(/Failed to load workflows/i)).toBeTruthy();
      expect(screen.getByRole('button', { name: /Retry/i })).toBeTruthy();
    });
  });

  it('Retry button triggers a second fetch', async () => {
    mockFetch
      .mockResolvedValueOnce(makeErrorFetchResponse())
      .mockResolvedValueOnce(makeOkResponse([]));
    render(<SavedWorkflowsPage />);
    await waitFor(() => screen.getByRole('button', { name: /Retry/i }));
    fireEvent.click(screen.getByRole('button', { name: /Retry/i }));
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });

  it('shows "No workflows yet" for empty list', async () => {
    mockFetch.mockResolvedValue(makeOkResponse([]));
    render(<SavedWorkflowsPage />);
    await waitFor(() => {
      expect(screen.getByText(/No workflows yet/i)).toBeTruthy();
    });
  });

  it('delete workflow calls DELETE /api/workflows/{id} after confirm', async () => {
    vi.stubGlobal('confirm', vi.fn().mockReturnValue(true));
    mockFetch
      .mockResolvedValueOnce(makeOkResponse([makeWorkflow('wf-del')])) // initial load
      .mockResolvedValueOnce(makeOkResponse(undefined))                 // DELETE
      .mockResolvedValueOnce(makeOkResponse([]));                       // reload after delete

    render(<SavedWorkflowsPage />);
    await waitFor(() => screen.getByText('Workflow wf-del'));

    // The delete button has variant="destructive" — only that variant adds bg-destructive.
    // The base Button class contains "aria-invalid:ring-destructive/..." which ALL buttons
    // share, so we must match "bg-destructive" (not just "destructive") to be specific.
    const allButtons = screen.getAllByRole('button');
    const deleteBtn = allButtons.find((b) => b.className?.includes('bg-destructive'));

    expect(deleteBtn).toBeTruthy();
    if (deleteBtn) {
      fireEvent.click(deleteBtn);
      await waitFor(() => {
        const deleteCall = mockFetch.mock.calls.find(
          (c) => c[1]?.method === 'DELETE' && (c[0] as string).includes('/api/workflows/wf-del')
        );
        expect(deleteCall).toBeTruthy();
      });
    }
  });

  it('clone workflow calls POST /api/workflows/{id}/clone', async () => {
    mockFetch
      .mockResolvedValueOnce(makeOkResponse([makeWorkflow('wf-clone')]))
      .mockResolvedValueOnce(makeOkResponse({ workflow_id: 'wf-clone-copy' }))
      .mockResolvedValueOnce(makeOkResponse([]));

    render(<SavedWorkflowsPage />);
    await waitFor(() => screen.getByText('Workflow wf-clone'));

    // Clone button uses variant="outline" — its unique class is "bg-background".
    // NOTE: the base Button class includes "outline-none", so includes('outline')
    // would match ALL buttons. We use 'bg-background' which is only on the outline variant.
    const allButtons = screen.getAllByRole('button');
    const cloneBtn = allButtons.find((b) => b.className?.includes('bg-background'));

    expect(cloneBtn).toBeTruthy();
    if (cloneBtn) {
      fireEvent.click(cloneBtn);
      await waitFor(() => {
        const cloneCall = mockFetch.mock.calls.find(
          (c) => (c[0] as string).includes('/clone') && c[1]?.method === 'POST'
        );
        expect(cloneCall).toBeTruthy();
      });
    }
  });
});
