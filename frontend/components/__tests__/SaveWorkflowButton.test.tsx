import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import SaveWorkflowButton from '@/components/save-workflow-button';

// ── Mocks ──────────────────────────────────────────────────────────────────────

// vi.hoisted is used for consistency; this component uses a static import.
const mockAuthFetch = vi.hoisted(() => vi.fn());

vi.mock('@/lib/auth-fetch', () => ({
  authFetch: mockAuthFetch,
}));

// ScheduleWorkflowDialog is a heavy child — stub it to keep tests focused
vi.mock('@/components/schedule-workflow-dialog', () => ({
  ScheduleWorkflowDialog: ({ open }: { open: boolean }) =>
    open ? <div data-testid="schedule-dialog">ScheduleDialog</div> : null,
}));

beforeEach(() => {
  vi.clearAllMocks();
});

// ── Helpers ────────────────────────────────────────────────────────────────────

const makeOkResponse = (body = { workflow_id: 'wf-123' }) =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
  });

// Valid UUIDs required because the component added UUID validation in handleSave:
//   const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
//   if (!uuidRegex.test(threadId)) { alert(...); return; }
// Tests that click "Save Workflow" with a non-empty name MUST use a real UUID.
const VALID_UUID = 'a1b2c3d4-e5f6-4789-abcd-ef0123456789';
const VALID_UUID_2 = 'b2c3d4e5-f6a7-4890-bcde-f01234567890';

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('SaveWorkflowButton', () => {
  it('renders disabled when threadId is null', () => {
    render(<SaveWorkflowButton threadId={null} />);
    const btn = screen.getByRole('button', { name: /Save as Workflow/i });
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });

  it('renders enabled when threadId is provided', () => {
    render(<SaveWorkflowButton threadId="thread-1" />);
    const btn = screen.getByRole('button', { name: /Save as Workflow/i });
    expect((btn as HTMLButtonElement).disabled).toBe(false);
  });

  it('opens the dialog when the trigger button is clicked', async () => {
    render(<SaveWorkflowButton threadId="thread-1" />);
    fireEvent.click(screen.getByRole('button', { name: /Save as Workflow/i }));
    // Check for the name input field that only renders inside the open dialog
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/e.g., Company Research/i)).toBeTruthy();
    });
  });

  it('Save button is disabled when name field is empty', async () => {
    render(<SaveWorkflowButton threadId="thread-1" />);
    fireEvent.click(screen.getByRole('button', { name: /Save as Workflow/i }));
    await waitFor(() => screen.getByPlaceholderText(/e.g., Company Research/i));

    const saveBtn = screen.getAllByRole('button').find(
      (b) => b.textContent?.includes('Save Workflow')
    ) as HTMLButtonElement;
    expect(saveBtn.disabled).toBe(true);
  });

  it('calls POST /api/workflows with thread_id and name params', async () => {
    mockAuthFetch.mockReturnValue(makeOkResponse());
    render(<SaveWorkflowButton threadId={VALID_UUID_2} />);
    fireEvent.click(screen.getByRole('button', { name: /Save as Workflow/i }));
    await waitFor(() => screen.getByPlaceholderText(/e.g., Company Research/i));

    fireEvent.change(screen.getByPlaceholderText(/e.g., Company Research/i), {
      target: { value: 'My Workflow' },
    });

    const saveBtn = screen.getAllByRole('button').find(
      (b) => b.textContent?.includes('Save Workflow')
    )!;
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(mockAuthFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflows'),
        expect.objectContaining({ method: 'POST' })
      );
      const url: string = mockAuthFetch.mock.calls[0][0];
      expect(url).toContain(`thread_id=${VALID_UUID_2}`);
      expect(url).toContain('name=My%20Workflow');
    });
  });

  it('shows "Workflow Saved!" success state after successful save', async () => {
    mockAuthFetch.mockReturnValue(makeOkResponse({ workflow_id: 'wf-abc' }));
    render(<SaveWorkflowButton threadId={VALID_UUID} />);
    fireEvent.click(screen.getByRole('button', { name: /Save as Workflow/i }));
    await waitFor(() => screen.getByPlaceholderText(/e.g., Company Research/i));

    fireEvent.change(screen.getByPlaceholderText(/e.g., Company Research/i), {
      target: { value: 'Test Workflow' },
    });
    const saveBtn = screen.getAllByRole('button').find(
      (b) => b.textContent?.includes('Save Workflow')
    )!;
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(screen.getByText('Workflow Saved!')).toBeTruthy();
    });
  });

  it('shows "Schedule Now" button after successful save', async () => {
    mockAuthFetch.mockReturnValue(makeOkResponse({ workflow_id: 'wf-abc' }));
    render(<SaveWorkflowButton threadId={VALID_UUID} />);
    fireEvent.click(screen.getByRole('button', { name: /Save as Workflow/i }));
    await waitFor(() => screen.getByPlaceholderText(/e.g., Company Research/i));

    fireEvent.change(screen.getByPlaceholderText(/e.g., Company Research/i), {
      target: { value: 'Test Workflow' },
    });
    const saveBtn = screen.getAllByRole('button').find(
      (b) => b.textContent?.includes('Save Workflow')
    )!;
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Schedule Now/i })).toBeTruthy();
    });
  });

  it('opens ScheduleWorkflowDialog when "Schedule Now" is clicked', async () => {
    mockAuthFetch.mockReturnValue(makeOkResponse({ workflow_id: 'wf-abc' }));
    render(<SaveWorkflowButton threadId={VALID_UUID} />);
    fireEvent.click(screen.getByRole('button', { name: /Save as Workflow/i }));
    await waitFor(() => screen.getByPlaceholderText(/e.g., Company Research/i));

    fireEvent.change(screen.getByPlaceholderText(/e.g., Company Research/i), {
      target: { value: 'Test Workflow' },
    });
    const saveBtn = screen.getAllByRole('button').find(
      (b) => b.textContent?.includes('Save Workflow')
    )!;
    fireEvent.click(saveBtn);

    await waitFor(() => screen.getByRole('button', { name: /Schedule Now/i }));
    fireEvent.click(screen.getByRole('button', { name: /Schedule Now/i }));

    await waitFor(() => {
      expect(screen.getByTestId('schedule-dialog')).toBeTruthy();
    });
  });
});
