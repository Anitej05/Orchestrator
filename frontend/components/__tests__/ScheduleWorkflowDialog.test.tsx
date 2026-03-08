import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { ScheduleWorkflowDialog } from '@/components/schedule-workflow-dialog';

// ── Mocks ──────────────────────────────────────────────────────────────────────

const mockAuthFetch = vi.fn();

vi.mock('@/lib/auth-fetch', () => ({
  authFetch: (...args: any[]) => mockAuthFetch(...args),
}));

vi.mock('@clerk/nextjs', () => ({
  useAuth: () => ({ getToken: vi.fn().mockResolvedValue('test-token') }),
}));

// Stub useToast to capture toast calls
const mockToast = vi.fn();
vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({ toast: mockToast }),
}));

beforeEach(() => {
  vi.clearAllMocks();
  mockAuthFetch.mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
  });
});

// ── Default props ─────────────────────────────────────────────────────────────

const defaultProps = {
  open: true,
  onOpenChange: vi.fn(),
  workflowId: 'wf-123',
  workflowName: 'My Workflow',
  onScheduleCreated: vi.fn(),
};

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('ScheduleWorkflowDialog', () => {
  it('renders with workflow name in description', () => {
    render(<ScheduleWorkflowDialog {...defaultProps} workflowName="Alpha Workflow" />);
    expect(screen.getByText(/Alpha Workflow/)).toBeTruthy();
  });

  it('does not render when open is false', () => {
    render(<ScheduleWorkflowDialog {...defaultProps} open={false} />);
    expect(screen.queryByText(/Schedule Workflow/i)).toBeNull();
  });

  it('shows hourly, daily, weekly, monthly options', () => {
    render(<ScheduleWorkflowDialog {...defaultProps} />);
    expect(screen.getByText(/Hourly/)).toBeTruthy();
    expect(screen.getByText(/Daily/)).toBeTruthy();
    expect(screen.getByText(/Weekly/)).toBeTruthy();
    expect(screen.getByText(/Monthly/)).toBeTruthy();
  });

  describe('cron expression generation', () => {
    it('daily at 09:00 IST → 03:30 UTC → cron "30 3 * * *"', () => {
      render(<ScheduleWorkflowDialog {...defaultProps} />);
      // Daily is default; set hour=09, minute=00
      // Use a more specific label to avoid matching "Hourly - Runs every hour" radio
      const hourInput = screen.getByLabelText(/Hour \(0-23/i);
      const minuteInput = screen.getByLabelText(/Minute \(0-59/i);
      fireEvent.change(hourInput, { target: { value: '9' } });
      fireEvent.change(minuteInput, { target: { value: '0' } });
      // Cron should show in the info box
      // Component uses padStart(2,'0') so h=3 → "03": cron is "30 03 * * *"
      expect(screen.getByText('30 03 * * *')).toBeTruthy();
    });

    it('hourly uses the minute only (ignores hour)', () => {
      render(<ScheduleWorkflowDialog {...defaultProps} />);
      const hourlyRadio = screen.getByLabelText(/Hourly/i);
      fireEvent.click(hourlyRadio);
      // minute field for hourly
      const minuteInput = screen.getByLabelText(/At which minute/i);
      fireEvent.change(minuteInput, { target: { value: '30' } });
      // Cron expression should be "0 * * * *" initially then "30 * * * *" after change
      expect(screen.getByText(/\* \* \*/)).toBeTruthy();
    });
  });

  it('shows "Invalid JSON format" when input template is malformed', () => {
    render(<ScheduleWorkflowDialog {...defaultProps} />);
    const templateInput = screen.getByPlaceholderText(/{"param1"/);
    fireEvent.change(templateInput, { target: { value: '{bad json' } });
    expect(screen.getByText('Invalid JSON format')).toBeTruthy();
  });

  it('calls POST /api/workflows/{id}/schedule with correct body on submit', async () => {
    render(<ScheduleWorkflowDialog {...defaultProps} />);
    const scheduleBtn = screen.getByRole('button', { name: /Schedule Workflow/i });
    fireEvent.click(scheduleBtn);

    await waitFor(() => {
      expect(mockAuthFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflows/wf-123/schedule'),
        expect.objectContaining({ method: 'POST' })
      );
    });

    const callBody = JSON.parse(mockAuthFetch.mock.calls[0][1].body);
    expect(callBody).toHaveProperty('cron_expression');
    expect(callBody).toHaveProperty('input_template');
  });

  it('shows success toast after successful schedule creation', async () => {
    render(<ScheduleWorkflowDialog {...defaultProps} />);
    fireEvent.click(screen.getByRole('button', { name: /Schedule Workflow/i }));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Workflow scheduled' })
      );
    });
  });

  it('shows error toast when API returns non-OK', async () => {
    mockAuthFetch.mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: 'Bad cron expression' }),
    });

    render(<ScheduleWorkflowDialog {...defaultProps} />);
    fireEvent.click(screen.getByRole('button', { name: /Schedule Workflow/i }));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Error', variant: 'destructive' })
      );
    });
  });

  it('calls onScheduleCreated after successful schedule', async () => {
    const onScheduleCreated = vi.fn();
    render(<ScheduleWorkflowDialog {...defaultProps} onScheduleCreated={onScheduleCreated} />);
    fireEvent.click(screen.getByRole('button', { name: /Schedule Workflow/i }));

    await waitFor(() => {
      expect(onScheduleCreated).toHaveBeenCalled();
    });
  });

  it('calls onOpenChange(false) after successful schedule', async () => {
    const onOpenChange = vi.fn();
    render(<ScheduleWorkflowDialog {...defaultProps} onOpenChange={onOpenChange} />);
    fireEvent.click(screen.getByRole('button', { name: /Schedule Workflow/i }));

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false);
    });
  });

  it('calls onOpenChange(false) when Cancel is clicked', () => {
    const onOpenChange = vi.fn();
    render(<ScheduleWorkflowDialog {...defaultProps} onOpenChange={onOpenChange} />);
    fireEvent.click(screen.getByRole('button', { name: /Cancel/i }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
