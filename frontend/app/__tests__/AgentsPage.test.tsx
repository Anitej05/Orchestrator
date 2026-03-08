import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import AgentsPage from '@/app/(dashboard)/agents/page';

// ── Mocks ──────────────────────────────────────────────────────────────────────

const mockFetchAllAgents = vi.fn();

vi.mock('@/lib/api-client', () => ({
  fetchAllAgents: (...args: any[]) => mockFetchAllAgents(...args),
}));

// Sidebar primitives used by the page
vi.mock('@/components/ui/sidebar', () => ({
  SidebarInset: ({ children }: any) => <div>{children}</div>,
  SidebarTrigger: () => <button>Menu</button>,
  useSidebar: () => ({ open: true }),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

// ── Helpers ────────────────────────────────────────────────────────────────────

const makeAgent = (id: string) => ({
  id,
  name: `Agent ${id}`,
  description: `Description ${id}`,
  capabilities: ['reporting'],
  status: 'active',
  endpoints: [],
});

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('AgentsPage', () => {
  it('shows "Loading agents…" during fetch', () => {
    mockFetchAllAgents.mockReturnValue(new Promise(() => {})); // never resolves
    render(<AgentsPage />);
    expect(screen.getByText(/Loading agents/i)).toBeTruthy();
  });

  it('renders AgentGrid with fetched agents after success', async () => {
    mockFetchAllAgents.mockResolvedValue([makeAgent('1'), makeAgent('2')]);
    render(<AgentsPage />);
    await waitFor(() => {
      expect(screen.getByText('Agent 1')).toBeTruthy();
      expect(screen.getByText('Agent 2')).toBeTruthy();
    });
  });

  it('renders empty grid (no error UI) when fetch fails — documents known gap', async () => {
    mockFetchAllAgents.mockRejectedValue(new Error('Network error'));
    render(<AgentsPage />);
    await waitFor(() => {
      // No error message is shown — known functional gap (silent failure)
      expect(screen.queryByText(/error/i)).toBeNull();
      expect(screen.queryByText(/failed/i)).toBeNull();
      expect(screen.getByText('No agents found')).toBeTruthy();
    });
  });

  it('shows agent count in results line after load', async () => {
    mockFetchAllAgents.mockResolvedValue([makeAgent('1'), makeAgent('2'), makeAgent('3')]);
    render(<AgentsPage />);
    await waitFor(() => {
      expect(screen.getByText(/3 agents/i)).toBeTruthy();
    });
  });

  it('filters agents by search query', async () => {
    mockFetchAllAgents.mockResolvedValue([
      makeAgent('alpha'),
      { ...makeAgent('beta'), name: 'Beta Agent' },
    ]);
    render(<AgentsPage />);
    await waitFor(() => screen.getByText('Agent alpha'));

    const searchInput = screen.getByPlaceholderText(/Search agents/i);
    fireEvent.change(searchInput, { target: { value: 'beta' } });

    await waitFor(() => {
      expect(screen.getByText('Beta Agent')).toBeTruthy();
      expect(screen.queryByText('Agent alpha')).toBeNull();
    });
  });

  it('shows "1 agent" (singular) for a single result', async () => {
    mockFetchAllAgents.mockResolvedValue([makeAgent('solo')]);
    render(<AgentsPage />);
    await waitFor(() => {
      expect(screen.getByText(/^1 agent$/)).toBeTruthy();
    });
  });
});
