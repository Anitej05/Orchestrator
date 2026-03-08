import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TaskListView from '@/components/task-list-view';

const makeTask = (overrides: Record<string, any> = {}) => ({
  task_id: 'task-1',
  description: 'Do something important',
  status: 'pending',
  ...overrides,
});

describe('TaskListView', () => {
  // ── Empty state ─────────────────────────────────────────────────────────────

  it('shows "No Tasks" when both todoList and pendingTasks are empty', () => {
    render(<TaskListView todoList={[]} taskStatuses={{}} pendingTasks={[]} />);
    expect(screen.getByText('No Tasks')).toBeTruthy();
  });

  it('shows helper text in empty state', () => {
    render(<TaskListView todoList={[]} taskStatuses={{}} pendingTasks={[]} />);
    expect(screen.getByText(/Tasks will appear here/)).toBeTruthy();
  });

  // ── Task rendering ──────────────────────────────────────────────────────────

  it('renders all tasks from todoList', () => {
    const todoList = [
      makeTask({ task_id: '1', description: 'First task' }),
      makeTask({ task_id: '2', description: 'Second task' }),
    ];
    render(<TaskListView todoList={todoList} taskStatuses={{}} pendingTasks={[]} />);
    expect(screen.getByText('First task')).toBeTruthy();
    expect(screen.getByText('Second task')).toBeTruthy();
  });

  it('renders task description as title', () => {
    render(
      <TaskListView
        todoList={[makeTask({ description: 'My specific task' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText('My specific task')).toBeTruthy();
  });

  // ── Status icons ─────────────────────────────────────────────────────────────

  it('renders ○ icon for pending status', () => {
    render(
      <TaskListView
        todoList={[makeTask({ status: 'pending' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText('○')).toBeTruthy();
  });

  it('renders ● icon with animate-pulse for in_progress status', () => {
    render(
      <TaskListView
        todoList={[makeTask({ status: 'in_progress' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    const icon = screen.getByText('●');
    expect(icon).toBeTruthy();
    expect(icon.className).toContain('animate-pulse');
  });

  it('renders ● icon with animate-pulse for in-progress (hyphenated) status', () => {
    render(
      <TaskListView
        todoList={[makeTask({ status: 'in-progress' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    const icon = screen.getByText('●');
    expect(icon.className).toContain('animate-pulse');
  });

  it('renders ✓ icon for completed status', () => {
    render(
      <TaskListView
        todoList={[makeTask({ status: 'completed' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText('✓')).toBeTruthy();
  });

  it('renders ✕ icon for failed status', () => {
    render(
      <TaskListView
        todoList={[makeTask({ status: 'failed' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText('✕')).toBeTruthy();
  });

  // ── Badges ───────────────────────────────────────────────────────────────────

  it('shows priority badge when priority field is present', () => {
    render(
      <TaskListView
        todoList={[makeTask({ priority: 'high' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText('high')).toBeTruthy();
  });

  it('does not show priority badge when priority is absent', () => {
    render(
      <TaskListView
        todoList={[makeTask({ priority: undefined })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    // Only the status badge should appear; no priority-related classes
    expect(screen.queryByText('medium')).toBeNull();
    expect(screen.queryByText('high')).toBeNull();
  });

  it('shows assigned_tool badge when field is present', () => {
    render(
      <TaskListView
        todoList={[makeTask({ assigned_tool: 'spreadsheet_agent' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText('spreadsheet_agent')).toBeTruthy();
  });

  // ── Error & result blocks ────────────────────────────────────────────────────

  it('shows error block when task has error field', () => {
    render(
      <TaskListView
        todoList={[makeTask({ error: 'Connection timeout' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText(/Connection timeout/)).toBeTruthy();
    expect(screen.getByText('Error:')).toBeTruthy();
  });

  it('shows result block for completed tasks with a result', () => {
    render(
      <TaskListView
        todoList={[makeTask({ status: 'completed', result: 'Done successfully' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText(/Done successfully/)).toBeTruthy();
    expect(screen.getByText('Result:')).toBeTruthy();
  });

  it('truncates long string results to 150 chars with ellipsis', () => {
    const longResult = 'x'.repeat(200);
    render(
      <TaskListView
        todoList={[makeTask({ status: 'completed', result: longResult })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.getByText(/x{150}\.\.\./)).toBeTruthy();
  });

  it('does not show result block for non-completed tasks', () => {
    render(
      <TaskListView
        todoList={[makeTask({ status: 'pending', result: 'Some result' })]}
        taskStatuses={{}}
        pendingTasks={[]}
      />
    );
    expect(screen.queryByText('Result:')).toBeNull();
  });

  // ── Fallback to pendingTasks ─────────────────────────────────────────────────

  it('falls back to pendingTasks.flat() when todoList is empty', () => {
    const pendingTasks = [
      [{ id: 'p1', description: 'Pending task A', status: 'pending' }],
    ];
    render(
      <TaskListView todoList={[]} taskStatuses={{}} pendingTasks={pendingTasks} />
    );
    expect(screen.getByText('Pending task A')).toBeTruthy();
  });

  it('prefers todoList over pendingTasks when both are provided', () => {
    const todoList = [makeTask({ task_id: 't1', description: 'Todo task' })];
    const pendingTasks = [[{ id: 'p1', description: 'Pending task', status: 'pending' }]];
    render(
      <TaskListView todoList={todoList} taskStatuses={{}} pendingTasks={pendingTasks} />
    );
    expect(screen.getByText('Todo task')).toBeTruthy();
    expect(screen.queryByText('Pending task')).toBeNull();
  });
});
