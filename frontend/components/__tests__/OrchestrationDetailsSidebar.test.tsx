/**
 * OrchestrationDetailsSidebar tests
 *
 * The sidebar reads directly from the Zustand conversation store, so we mock
 * the entire store module and seed it with controlled state per test.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React, { useState } from 'react';

// ── Mock Radix UI Tabs so click-to-switch works in jsdom ─────────────────────
// Radix UI Tabs relies on pointer/focus APIs not available in jsdom.
// We replace it with a simple controlled implementation that reacts to clicks.
vi.mock('@/components/ui/tabs', () => {
  const Tabs = ({ children, value, onValueChange, className }: any) => (
    <div data-testid="tabs" data-value={value} className={className}>
      {React.Children.map(children, (child: any) =>
        child ? React.cloneElement(child, { _activeTab: value, _onTabChange: onValueChange }) : null
      )}
    </div>
  );
  const TabsList = ({ children, _activeTab, _onTabChange, className }: any) => (
    <div role="tablist" className={className}>
      {React.Children.map(children, (child: any) =>
        child ? React.cloneElement(child, { _activeTab, _onTabChange }) : null
      )}
    </div>
  );
  const TabsTrigger = ({ children, value, _activeTab, _onTabChange, className }: any) => (
    <button
      role="tab"
      aria-selected={_activeTab === value}
      onClick={() => _onTabChange?.(value)}
      className={className}
    >
      {children}
    </button>
  );
  const TabsContent = ({ children, value, _activeTab, className }: any) =>
    _activeTab === value ? <div role="tabpanel" className={className}>{children}</div> : null;
  return { Tabs, TabsList, TabsTrigger, TabsContent };
});

// ── Mock heavy child components ──────────────────────────────────────────────

vi.mock('@/components/canvas-renderer', () => ({
  CanvasRenderer: () => <div data-testid="canvas-renderer">Canvas</div>,
}));
vi.mock('@/components/action-history-timeline', () => ({
  default: ({ history }: { history: any[] }) => (
    <div data-testid="action-history">{history.length} actions</div>
  ),
}));
vi.mock('@/components/task-card-list', () => ({
  default: ({ todoList }: { todoList: any[] }) => (
    <div data-testid="task-card-list">{todoList.length} tasks</div>
  ),
}));
vi.mock('@/components/task-list-view', () => ({
  default: () => <div data-testid="task-list-view" />,
}));
vi.mock('@/components/save-workflow-button', () => ({
  default: () => <button>Save Workflow</button>,
}));
vi.mock('@/components/document-viewer', () => ({
  default: () => <div data-testid="document-viewer" />,
}));
vi.mock('@/lib/canvas-api', () => ({
  dismissCanvas: vi.fn().mockResolvedValue(undefined),
}));
vi.mock('@/lib/config', () => ({
  API_BASE_URL: 'http://localhost:8000',
}));

// ── Zustand store mock ───────────────────────────────────────────────────────

const BASE_STORE_STATE = {
  thread_id: 'test-thread-1',
  status: 'idle' as const,
  messages: [],
  isWaitingForUser: false,
  task_agent_pairs: [],
  todo_list: [],
  action_history: [],
  brain_reasoning: undefined,
  uploaded_files: [],
  plan: [],
  task_statuses: {},
  has_canvas: false,
  canvas_content: undefined,
  canvas_data: undefined,
  canvas_type: undefined,
  canvas_title: undefined,
  canvas_metadata: undefined,
  browser_view: undefined,
  metadata: { currentStage: 'idle' },
  actions: {
    sendCanvasConfirmation: vi.fn(),
    continueConversation: vi.fn(),
  },
};

let storeState = { ...BASE_STORE_STATE };

vi.mock('@/lib/conversation-store', () => ({
  useConversationStore: (selector?: (s: any) => any) => {
    if (selector) return selector(storeState);
    return storeState;
  },
}));

// ── Import AFTER mocks ───────────────────────────────────────────────────────

import OrchestrationDetailsSidebar from '@/components/orchestration-details-sidebar';

// ── Helpers ──────────────────────────────────────────────────────────────────

function renderSidebar(props: {
  executionResults?: any[];
  threadId?: string | null;
} = {}) {
  return render(
    <OrchestrationDetailsSidebar
      executionResults={props.executionResults ?? []}
      threadId={props.threadId ?? 'test-thread-1'}
    />
  );
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('OrchestrationDetailsSidebar', () => {
  beforeEach(() => {
    storeState = { ...BASE_STORE_STATE };
  });

  // ── Tab rendering ──────────────────────────────────────────────────────

  describe('tab structure', () => {
    it('renders all four tabs', () => {
      renderSidebar();
      expect(screen.getByRole('tab', { name: /Plan/i })).toBeTruthy();
      expect(screen.getByRole('tab', { name: /History/i })).toBeTruthy();
      expect(screen.getByRole('tab', { name: /Files/i })).toBeTruthy();
      expect(screen.getByRole('tab', { name: /Canvas/i })).toBeTruthy();
    });

    it('shows Plan tab by default', () => {
      renderSidebar();
      // OrchestrationFlow renders EmptyState when no flow activity yet
      expect(screen.getByText('No activity yet')).toBeTruthy();
    });
  });

  // ── Plan tab ──────────────────────────────────────────────────────────

  describe('Plan tab', () => {
    it('shows OrchestrationFlow inside Plan tab', () => {
      storeState = {
        ...BASE_STORE_STATE,
        todo_list: [
          { task_id: '1', description: 'Task one', status: 'pending' },
          { task_id: '2', description: 'Task two', status: 'pending' },
        ],
      };
      renderSidebar();
      // Plan tab renders OrchestrationFlow, which shows EmptyState when flow_events is empty
      expect(screen.getByText('No activity yet')).toBeTruthy();
    });

    it('shows empty state in idle state when no flow events', () => {
      renderSidebar();
      expect(screen.getByText('No activity yet')).toBeTruthy();
    });

    it('shows live indicator when status is processing', () => {
      storeState = {
        ...BASE_STORE_STATE,
        status: 'processing',
      };
      renderSidebar();
      // OrchestrationFlow shows ThinkingRow when processing with no flow events yet
      expect(screen.getByText('Analysing your request')).toBeTruthy();
    });

    it('shows "Running" badge on Plan tab trigger when executing', () => {
      storeState = {
        ...BASE_STORE_STATE,
        metadata: { currentStage: 'executing' },
      };
      renderSidebar();
      // Sidebar adds a "Running" badge to the Plan tab trigger when currentStage is executing
      expect(screen.getByText('Running')).toBeTruthy();
    });

    it('shows "All done" when completed with flow events', () => {
      storeState = {
        ...BASE_STORE_STATE,
        status: 'completed',
        flow_events: [
          { id: 'e1', type: 'brain', action_type: 'agent', status: 'completed', label: 'Selected Spreadsheet Agent' },
        ],
      };
      renderSidebar();
      expect(screen.getByText('All done')).toBeTruthy();
    });

    it('shows "Running" badge on Plan tab trigger when validating', () => {
      storeState = {
        ...BASE_STORE_STATE,
        metadata: { currentStage: 'validating' },
      };
      renderSidebar();
      // Sidebar adds a "Running" badge to the Plan tab trigger when currentStage is validating
      expect(screen.getByText('Running')).toBeTruthy();
    });

    it('shows Save Workflow button when status is completed', () => {
      storeState = {
        ...BASE_STORE_STATE,
        status: 'completed',
        thread_id: 'test-thread-1',
      };
      renderSidebar();
      expect(screen.getByText('Save Workflow')).toBeTruthy();
    });
  });

  // ── History tab ───────────────────────────────────────────────────────

  describe('History tab', () => {
    it('switches to History tab on click', async () => {
      renderSidebar();
      fireEvent.click(screen.getByRole('tab', { name: /History/i }));
      await waitFor(() => expect(screen.getByTestId('action-history')).toBeTruthy());
    });

    it('passes action_history to ActionHistoryTimeline', async () => {
      storeState = {
        ...BASE_STORE_STATE,
        action_history: [
          {
            iteration: 1,
            action_type: 'agent',
            resource_id: 'spreadsheet_agent',
            instruction: 'Analyze data',
            success: true,
            result_summary: 'Done',
            execution_time_ms: 1000,
          },
          {
            iteration: 2,
            action_type: 'agent',
            resource_id: 'document_agent',
            instruction: 'Read doc',
            success: true,
            result_summary: 'Done',
            execution_time_ms: 500,
          },
        ],
      };
      renderSidebar();
      fireEvent.click(screen.getByRole('tab', { name: /History/i }));
      await waitFor(() => expect(screen.getByText('2 actions')).toBeTruthy());
    });
  });

  // ── Files / Attachments tab ───────────────────────────────────────────

  describe('Attachments tab', () => {
    it('switches to Files tab on click', async () => {
      renderSidebar();
      fireEvent.click(screen.getByRole('tab', { name: /Files/i }));
      await waitFor(() => expect(screen.getByText('No Attachments')).toBeTruthy());
    });

    it('shows "No Attachments" when no files are uploaded', async () => {
      storeState = { ...BASE_STORE_STATE, uploaded_files: [] };
      renderSidebar();
      fireEvent.click(screen.getByRole('tab', { name: /Files/i }));
      await waitFor(() => expect(screen.getByText('No Attachments')).toBeTruthy());
    });

    it('renders attachment file cards for uploaded files', async () => {
      storeState = {
        ...BASE_STORE_STATE,
        uploaded_files: [
          { file_name: 'report.pdf', file_path: '/tmp/report.pdf', file_type: 'application/pdf' },
          { file_name: 'data.xlsx', file_path: '/tmp/data.xlsx', file_type: 'application/vnd.ms-excel' },
        ],
      };
      renderSidebar();
      fireEvent.click(screen.getByRole('tab', { name: /Files/i }));
      await waitFor(() => {
        expect(screen.getByText('report.pdf')).toBeTruthy();
        expect(screen.getByText('data.xlsx')).toBeTruthy();
      });
    });
  });

  // ── Canvas tab ────────────────────────────────────────────────────────

  describe('Canvas tab', () => {
    it('shows "No Canvas Content" message when no canvas', async () => {
      renderSidebar();
      fireEvent.click(screen.getByRole('tab', { name: /Canvas/i }));
      await waitFor(() => expect(screen.getByText('No Canvas Content')).toBeTruthy());
    });

    it('renders CanvasRenderer when has_canvas is true with content', async () => {
      storeState = {
        ...BASE_STORE_STATE,
        has_canvas: true,
        canvas_content: '<html><body>Hello</body></html>',
        canvas_type: 'html',
      };
      renderSidebar();
      fireEvent.click(screen.getByRole('tab', { name: /Canvas/i }));
      await waitFor(() => expect(screen.getByTestId('canvas-renderer')).toBeTruthy());
    });

    it('shows canvas type badge in header when canvas is present', async () => {
      storeState = {
        ...BASE_STORE_STATE,
        has_canvas: true,
        canvas_content: '# Hello',
        canvas_type: 'markdown',
      };
      renderSidebar();
      fireEvent.click(screen.getByRole('tab', { name: /Canvas/i }));
      await waitFor(() => expect(screen.getByText('markdown')).toBeTruthy());
    });
  });
});
