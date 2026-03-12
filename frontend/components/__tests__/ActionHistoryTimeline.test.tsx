import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ActionHistoryTimeline from '@/components/action-history-timeline';

// ── Factory ─────────────────────────────────────────────────────────────────

interface ActionEntry {
  iteration: number;
  action_type: string;
  resource_id: string;
  instruction: string;
  success: boolean;
  result_summary: string;
  execution_time_ms: number;
  error?: string;
}

const makeEntry = (overrides: Partial<ActionEntry> = {}): ActionEntry => ({
  iteration: 1,
  action_type: 'agent',
  resource_id: 'spreadsheet_agent',
  instruction: 'Analyze the MRP data',
  success: true,
  result_summary: 'Found 12 pending requisitions.',
  execution_time_ms: 2500,
  ...overrides,
});

// ── Empty state ──────────────────────────────────────────────────────────────

describe('ActionHistoryTimeline', () => {
  describe('empty state', () => {
    it('shows "No actions yet" when history is empty', () => {
      render(<ActionHistoryTimeline history={[]} />);
      expect(screen.getByText('No actions yet')).toBeTruthy();
    });

    it('shows hint text in empty state', () => {
      render(<ActionHistoryTimeline history={[]} />);
      expect(screen.getByText(/Agent actions will appear here as they execute/)).toBeTruthy();
    });

    it('handles null/undefined history gracefully', () => {
      render(<ActionHistoryTimeline history={null as any} />);
      expect(screen.getByText('No actions yet')).toBeTruthy();
    });
  });

  // ── Entry rendering ──────────────────────────────────────────────────────

  describe('entry rendering', () => {
    it('renders agent name formatted from resource_id', () => {
      render(<ActionHistoryTimeline history={[makeEntry()]} />);
      // "spreadsheet_agent" → "Spreadsheet Agent"
      expect(screen.getByText('Spreadsheet Agent')).toBeTruthy();
    });

    it('renders action_type as badge', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ action_type: 'agent' })]} />);
      expect(screen.getByText('agent')).toBeTruthy();
    });

    it('renders iteration number', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ iteration: 3 })]} />);
      expect(screen.getByText('Iteration 3')).toBeTruthy();
    });

    it('renders instruction text', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ instruction: 'Find all pending orders' })]} />);
      expect(screen.getByText('Find all pending orders')).toBeTruthy();
    });

    it('renders result summary', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ result_summary: 'Found 42 items.' })]} />);
      expect(screen.getByText('Found 42 items.')).toBeTruthy();
    });

    it('renders execution time in seconds', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ execution_time_ms: 2500 })]} />);
      expect(screen.getByText('2.50s')).toBeTruthy();
    });

    it('renders multiple entries', () => {
      const entries = [
        makeEntry({ iteration: 1, resource_id: 'spreadsheet_agent', instruction: 'Step one' }),
        makeEntry({ iteration: 2, resource_id: 'document_agent', instruction: 'Step two' }),
      ];
      render(<ActionHistoryTimeline history={entries} />);
      expect(screen.getByText('Spreadsheet Agent')).toBeTruthy();
      expect(screen.getByText('Document Agent')).toBeTruthy();
      expect(screen.getByText('Step one')).toBeTruthy();
      expect(screen.getByText('Step two')).toBeTruthy();
    });
  });

  // ── Success / failure styling ────────────────────────────────────────────

  describe('success and failure', () => {
    it('renders error details section when entry has failed with error', () => {
      const entry = makeEntry({ success: false, error: 'Connection timed out', result_summary: '' });
      render(<ActionHistoryTimeline history={[entry]} />);
      expect(screen.getByText('Error')).toBeTruthy();
      expect(screen.getByText('Connection timed out')).toBeTruthy();
    });

    it('does not render error details section for successful entries', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ success: true })]} />);
      expect(screen.queryByText('Error')).toBeNull();
    });

    it('shows "Completed successfully" when result_summary is empty and success', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ result_summary: '', success: true })]} />);
      expect(screen.getByText('Completed successfully')).toBeTruthy();
    });

    it('shows "Operation failed" when result_summary is empty and failure', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ result_summary: '', success: false })]} />);
      expect(screen.getByText('Operation failed')).toBeTruthy();
    });
  });

  // ── Instruction parsing ──────────────────────────────────────────────────

  describe('instruction parsing', () => {
    it('renders plain text instructions as-is', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ instruction: 'Simple plain text' })]} />);
      expect(screen.getByText('Simple plain text')).toBeTruthy();
    });

    it('extracts instruction from JSON object format', () => {
      const jsonInstruction = JSON.stringify({ instruction: 'Read the MRP file' });
      render(<ActionHistoryTimeline history={[makeEntry({ instruction: jsonInstruction })]} />);
      expect(screen.getByText('Read the MRP file')).toBeTruthy();
    });

    it('shows "No instruction provided" for empty instruction', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ instruction: '' })]} />);
      expect(screen.getByText('No instruction provided')).toBeTruthy();
    });
  });

  // ── Result summary parsing ───────────────────────────────────────────────

  describe('result summary parsing', () => {
    it('renders plain text result directly', () => {
      render(<ActionHistoryTimeline history={[makeEntry({ result_summary: 'Analysis complete.' })]} />);
      expect(screen.getByText('Analysis complete.')).toBeTruthy();
    });

    it('extracts task_summary from JSON result', () => {
      const jsonResult = JSON.stringify({ task_summary: 'Processing done with 5 rows.' });
      render(<ActionHistoryTimeline history={[makeEntry({ result_summary: jsonResult })]} />);
      expect(screen.getByText('Processing done with 5 rows.')).toBeTruthy();
    });

    it('extracts task_summary from single-quoted Python-style dict', () => {
      const pythonResult = "result: {'task_summary': 'Pivot table created successfully.'}";
      render(<ActionHistoryTimeline history={[makeEntry({ result_summary: pythonResult })]} />);
      expect(screen.getByText('Pivot table created successfully.')).toBeTruthy();
    });
  });

  // ── Agent icons ──────────────────────────────────────────────────────────

  describe('agent icon mapping', () => {
    const iconCases: [string, string][] = [
      ['spreadsheet_agent', '📊'],
      ['document_agent', '📄'],
      ['gmail_agent', '📧'],
      ['browser_agent', '🌐'],
      ['coding_agent', '💻'],
      ['unknown_xyz', '🔷'],   // default fallback
    ];

    iconCases.forEach(([resourceId, icon]) => {
      it(`shows ${icon} icon for resource_id "${resourceId}"`, () => {
        render(
          <ActionHistoryTimeline
            history={[makeEntry({ resource_id: resourceId })]}
          />
        );
        expect(screen.getByText(icon)).toBeTruthy();
      });
    });
  });

  // ── Deduplication ────────────────────────────────────────────────────────

  describe('deduplication', () => {
    it('renders only one entry when exact duplicates are passed', () => {
      const entry = makeEntry({ iteration: 1, resource_id: 'spreadsheet_agent', execution_time_ms: 1000 });
      // Same entry twice — same unique key
      render(<ActionHistoryTimeline history={[entry, { ...entry }]} />);
      // There should only be ONE "Spreadsheet Agent" heading
      const agentNames = screen.getAllByText('Spreadsheet Agent');
      expect(agentNames).toHaveLength(1);
    });

    it('renders both entries when they differ by iteration', () => {
      const entry1 = makeEntry({ iteration: 1, resource_id: 'spreadsheet_agent', execution_time_ms: 1000 });
      const entry2 = makeEntry({ iteration: 2, resource_id: 'spreadsheet_agent', execution_time_ms: 1000 });
      render(<ActionHistoryTimeline history={[entry1, entry2]} />);
      const agentNames = screen.getAllByText('Spreadsheet Agent');
      expect(agentNames).toHaveLength(2);
    });
  });
});
