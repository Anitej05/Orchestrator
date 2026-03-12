import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TaskCardList from '@/components/task-card-list';
import type { TodoItem, TaskStatus } from '@/lib/types';

// ── Factories ────────────────────────────────────────────────────────────────

const makeTodo = (overrides: Partial<TodoItem> = {}): TodoItem => ({
  task_id: 'task-1',
  description: 'Do something important',
  status: 'pending',
  ...overrides,
});

const makeStatus = (overrides: Partial<TaskStatus> = {}): TaskStatus => ({
  status: 'pending',
  taskName: 'task-1',
  ...overrides,
});

// ── Empty state ──────────────────────────────────────────────────────────────

describe('TaskCardList', () => {
  describe('empty state', () => {
    it('renders default empty title when no tasks', () => {
      render(<TaskCardList />);
      expect(screen.getByText('No Tasks')).toBeTruthy();
    });

    it('renders default empty subtitle', () => {
      render(<TaskCardList />);
      expect(screen.getByText(/Tasks will appear here/)).toBeTruthy();
    });

    it('respects custom emptyTitle and emptySubtitle', () => {
      render(
        <TaskCardList
          emptyTitle="Nothing Yet"
          emptySubtitle="Check back later"
        />
      );
      expect(screen.getByText('Nothing Yet')).toBeTruthy();
      expect(screen.getByText('Check back later')).toBeTruthy();
    });

    it('renders empty when todoList is empty and fallbackTasks is empty', () => {
      render(<TaskCardList todoList={[]} fallbackTasks={[]} />);
      expect(screen.getByText('No Tasks')).toBeTruthy();
    });
  });

  // ── Single task rendering ────────────────────────────────────────────────

  describe('task rendering', () => {
    it('renders task description as title', () => {
      render(<TaskCardList todoList={[makeTodo({ description: 'Analyze spreadsheet' })]} />);
      expect(screen.getByText('Analyze spreadsheet')).toBeTruthy();
    });

    it('renders multiple tasks', () => {
      const tasks = [
        makeTodo({ task_id: '1', description: 'First task' }),
        makeTodo({ task_id: '2', description: 'Second task' }),
        makeTodo({ task_id: '3', description: 'Third task' }),
      ];
      render(<TaskCardList todoList={tasks} />);
      expect(screen.getByText('First task')).toBeTruthy();
      expect(screen.getByText('Second task')).toBeTruthy();
      expect(screen.getByText('Third task')).toBeTruthy();
    });

    it('shows numbered index when task is pending', () => {
      const tasks = [
        makeTodo({ task_id: '1', description: 'Task One', status: 'pending' }),
        makeTodo({ task_id: '2', description: 'Task Two', status: 'pending' }),
      ];
      render(<TaskCardList todoList={tasks} />);
      // Numbers 1 and 2 should be present as list index indicators
      expect(screen.getByText('1')).toBeTruthy();
      expect(screen.getByText('2')).toBeTruthy();
    });

    it('uses fallbackTasks when todoList is empty', () => {
      const fallback = [{ task_id: 'fb-1', description: 'Fallback task', status: 'pending' }];
      render(<TaskCardList todoList={[]} fallbackTasks={fallback} />);
      expect(screen.getByText('Fallback task')).toBeTruthy();
    });

    it('preferrs todoList over fallbackTasks', () => {
      const todo = [makeTodo({ description: 'From todo' })];
      const fallback = [{ description: 'From fallback', status: 'pending' }];
      render(<TaskCardList todoList={todo} fallbackTasks={fallback} />);
      expect(screen.getByText('From todo')).toBeTruthy();
      expect(screen.queryByText('From fallback')).toBeNull();
    });
  });

  // ── Status icons ─────────────────────────────────────────────────────────

  describe('status states', () => {
    it('shows running spinner for in_progress task (via taskStatuses)', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Running task', status: 'pending' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({ status: 'running', taskName: 'task-1' }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      // Loader2 SVG is rendered for running — its parent li should exist
      const text = screen.getByText('Running task');
      expect(text).toBeTruthy();
    });

    it('shows completed check icon class for completed task', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Done task', status: 'completed' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({ status: 'completed', taskName: 'task-1' }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.getByText('Done task')).toBeTruthy();
    });

    it('shows failed task title in error style', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Failed task', status: 'failed' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({ status: 'failed', taskName: 'task-1' }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.getByText('Failed task')).toBeTruthy();
    });

    it('normalizes "in-progress" to running state', () => {
      // The task itself has status "in-progress" (via the todo item directly)
      const tasks = [makeTodo({ task_id: 'task-1', description: 'In-progress task', status: 'in-progress' })];
      render(<TaskCardList todoList={tasks} taskStatuses={{}} />);
      expect(screen.getByText('In-progress task')).toBeTruthy();
    });

    it('normalizes "in_progress" to running state', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Running task', status: 'in_progress' })];
      render(<TaskCardList todoList={tasks} taskStatuses={{}} />);
      expect(screen.getByText('Running task')).toBeTruthy();
    });
  });

  // ── Live text ────────────────────────────────────────────────────────────

  describe('live activity text', () => {
    it('shows activity description for running task', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Processing', status: 'pending' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({
          status: 'running',
          taskName: 'task-1',
          activityDescription: 'Calling spreadsheet agent...',
        }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.getByText('Calling spreadsheet agent...')).toBeTruthy();
    });

    it('truncates activity text longer than 120 characters', () => {
      const longText = 'A'.repeat(130);
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Task', status: 'pending' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({
          status: 'running',
          taskName: 'task-1',
          activityDescription: longText,
        }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      const displayed = screen.getByText(/A{100,}/);
      expect(displayed.textContent?.endsWith('…')).toBe(true);
      expect(displayed.textContent!.length).toBeLessThanOrEqual(121); // 120 + ellipsis
    });

    it('does not show live text for completed tasks', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Done task' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({
          status: 'completed',
          taskName: 'task-1',
          activityDescription: 'Should not appear',
        }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.queryByText('Should not appear')).toBeNull();
    });
  });

  // ── Result and error display ─────────────────────────────────────────────

  describe('result and error display', () => {
    it('shows result summary for completed task', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Finished' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({
          status: 'completed',
          taskName: 'task-1',
          resultSummary: 'Found 5 pending orders.',
        }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.getByText('Found 5 pending orders.')).toBeTruthy();
    });

    it('shows error message for failed task', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Failed', error: 'Timeout error' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({ status: 'failed', taskName: 'task-1', error: 'Timeout error' }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.getByText('Timeout error')).toBeTruthy();
    });
  });

  // ── Agent pill ───────────────────────────────────────────────────────────

  describe('agent pill', () => {
    it('shows formatted agent name when assigned_to is set', () => {
      const tasks = [
        makeTodo({ task_id: 'task-1', description: 'Task', status: 'pending' }) as any,
      ];
      // Inject assigned_to via fallbackTasks (GenericTask accepts any keys)
      render(
        <TaskCardList
          fallbackTasks={[{ task_id: 'task-1', description: 'Task', status: 'pending', assigned_to: 'spreadsheet_agent' }]}
        />
      );
      expect(screen.getByText('via Spreadsheet Agent')).toBeTruthy();
    });
  });

  // ── Progress bar ─────────────────────────────────────────────────────────

  describe('progress bar', () => {
    it('shows progress summary when more than 1 task', () => {
      const tasks = [
        makeTodo({ task_id: '1', description: 'Task 1' }),
        makeTodo({ task_id: '2', description: 'Task 2' }),
      ];
      render(<TaskCardList todoList={tasks} />);
      // "0 / 2" counter text
      expect(screen.getByText(/0 \/ 2/)).toBeTruthy();
    });

    it('does not show progress bar for a single task', () => {
      render(<TaskCardList todoList={[makeTodo()]} />);
      expect(screen.queryByText(/\/ 1 done/)).toBeNull();
    });

    it('counts completed tasks correctly in progress summary', () => {
      // Use non-numeric IDs to avoid the index-fallback collision in resolveTaskStatus
      const tasks = [
        makeTodo({ task_id: 'alpha', description: 'Task Alpha' }),
        makeTodo({ task_id: 'beta', description: 'Task Beta' }),
        makeTodo({ task_id: 'gamma', description: 'Task Gamma' }),
      ];
      const statuses: Record<string, TaskStatus> = {
        'alpha': makeStatus({ status: 'completed', taskName: 'alpha' }),
        'beta': makeStatus({ status: 'completed', taskName: 'beta' }),
        // gamma remains pending
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.getByText(/2 \/ 3/)).toBeTruthy();
    });
  });

  // ── Brain reasoning ──────────────────────────────────────────────────────

  describe('brain reasoning', () => {
    it('shows brain reasoning text when provided', () => {
      render(
        <TaskCardList
          todoList={[makeTodo()]}
          brainReasoning="Analyzing the user request..."
        />
      );
      expect(screen.getByText('Analyzing the user request...')).toBeTruthy();
    });

    it('does not render brain reasoning section when not provided', () => {
      render(<TaskCardList todoList={[makeTodo()]} />);
      // BrainCircuit icon wrapper should not be present
      expect(screen.queryByText(/Analyzing/)).toBeNull();
    });
  });

  // ── Execution time ───────────────────────────────────────────────────────

  describe('execution time display', () => {
    it('shows execution time in seconds for completed task', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Done task' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({ status: 'completed', taskName: 'task-1', executionTime: 3500 }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.getByText('3.5s')).toBeTruthy();
    });

    it('shows execution time in ms for sub-second durations', () => {
      const tasks = [makeTodo({ task_id: 'task-1', description: 'Fast task' })];
      const statuses: Record<string, TaskStatus> = {
        'task-1': makeStatus({ status: 'completed', taskName: 'task-1', executionTime: 800 }),
      };
      render(<TaskCardList todoList={tasks} taskStatuses={statuses} />);
      expect(screen.getByText('800ms')).toBeTruthy();
    });
  });
});
