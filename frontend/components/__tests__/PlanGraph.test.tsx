/**
 * PlanGraph Component Unit Tests
 * 
 * Tests the PlanGraph component's rendering, layout algorithm,
 * and status update behavior.
 * 
 * Run with: npm test -- PlanGraph.test.tsx
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import PlanGraph from '../PlanGraph';
import type { TaskStatus } from '@/lib/types';

// Mock ReactFlow to avoid rendering complexity in tests
vi.mock('reactflow', () => ({
  __esModule: true,
  default: ({ nodes, edges }: any) => (
    <div data-testid="react-flow">
      <div data-testid="nodes-count">{nodes.length}</div>
      <div data-testid="edges-count">{edges.length}</div>
      {nodes.map((node: any) => (
        <div key={node.id} data-testid={`node-${node.id}`} data-status={node.data.status}>
          {node.data.label}
        </div>
      ))}
    </div>
  ),
  Controls: () => <div data-testid="controls" />,
  Background: () => <div data-testid="background" />,
  Position: {
    Top: 'top',
    Bottom: 'bottom',
  },
  Handle: () => null,
}));

describe('PlanGraph Component', () => {
  describe('Empty State', () => {
    it('should render empty state when no tasks provided', () => {
      const plan = { pendingTasks: [], completedTasks: [] };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      expect(screen.getByText(/Workflow Plan/i)).toBeInTheDocument();
      expect(screen.getByText(/execution plan graph will appear here/i)).toBeInTheDocument();
    });

    it('should render empty state when planData is null', () => {
      render(<PlanGraph planData={null as any} taskStatuses={{}} />);
      
      expect(screen.getByText(/Workflow Plan/i)).toBeInTheDocument();
    });
  });

  describe('Basic Rendering', () => {
    it('should render correct number of nodes for single tasks', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'First task', agent: 'Agent 1' },
          { task: 'Task 2', description: 'Second task', agent: 'Agent 2' },
        ],
        completedTasks: [],
      };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      const nodesCount = screen.getByTestId('nodes-count');
      expect(nodesCount).toHaveTextContent('2');
    });

    it('should render correct number of nodes for parallel tasks', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'First task', agent: 'Agent 1' },
          [
            { task: 'Task 2A', description: 'Parallel task A', agent: 'Agent 2' },
            { task: 'Task 2B', description: 'Parallel task B', agent: 'Agent 3' },
          ],
        ],
        completedTasks: [],
      };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      const nodesCount = screen.getByTestId('nodes-count');
      expect(nodesCount).toHaveTextContent('3'); // 1 + 2 parallel tasks
    });

    it('should render task names correctly', () => {
      const plan = {
        pendingTasks: [
          { task: 'Search Emails', description: 'Search for emails', agent: 'Gmail Agent' },
        ],
        completedTasks: [],
      };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      expect(screen.getByText('Search Emails')).toBeInTheDocument();
    });
  });

  describe('Status Updates', () => {
    it('should apply pending status by default', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      const node = screen.getByTestId('node-task-Task 1');
      expect(node).toHaveAttribute('data-status', 'pending');
    });

    it('should apply running status from taskStatuses', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      const taskStatuses: Record<string, TaskStatus> = {
        'Task 1': { status: 'running', taskName: 'Task 1', startTime: Date.now() },
      };
      
      render(<PlanGraph planData={plan} taskStatuses={taskStatuses} />);
      
      const node = screen.getByTestId('node-task-Task 1');
      expect(node).toHaveAttribute('data-status', 'running');
    });

    it('should apply completed status with execution time', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      const taskStatuses: Record<string, TaskStatus> = {
        'Task 1': { 
          status: 'completed',
          taskName: 'Task 1', 
          executionTime: 2.45,
          startTime: Date.now() 
        },
      };
      
      render(<PlanGraph planData={plan} taskStatuses={taskStatuses} />);
      
      const node = screen.getByTestId('node-task-Task 1');
      expect(node).toHaveAttribute('data-status', 'completed');
    });

    it('should apply failed status with error message', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      const taskStatuses: Record<string, TaskStatus> = {
        'Task 1': { 
          status: 'failed',
          taskName: 'Task 1', 
          error: 'Connection timed out',
          startTime: Date.now() 
        },
      };
      
      render(<PlanGraph planData={plan} taskStatuses={taskStatuses} />);
      
      const node = screen.getByTestId('node-task-Task 1');
      expect(node).toHaveAttribute('data-status', 'failed');
    });

    it('should update status when taskStatuses prop changes', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      const { rerender } = render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      let node = screen.getByTestId('node-task-Task 1');
      expect(node).toHaveAttribute('data-status', 'pending');
      
      // Update status to running
      const taskStatuses: Record<string, TaskStatus> = {
        'Task 1': { status: 'running', taskName: 'Task 1', startTime: Date.now() },
      };
      
      rerender(<PlanGraph planData={plan} taskStatuses={taskStatuses} />);
      
      node = screen.getByTestId('node-task-Task 1');
      expect(node).toHaveAttribute('data-status', 'running');
    });
  });

  describe('Edge Creation', () => {
    it('should create edges between sequential tasks', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'First', agent: 'Agent 1' },
          { task: 'Task 2', description: 'Second', agent: 'Agent 2' },
        ],
        completedTasks: [],
      };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      const edgesCount = screen.getByTestId('edges-count');
      expect(edgesCount).toHaveTextContent('1'); // 1 edge: Task 1 → Task 2
    });

    it('should create edges from parent to all parallel children', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Parent', agent: 'Agent 1' },
          [
            { task: 'Task 2A', description: 'Child A', agent: 'Agent 2' },
            { task: 'Task 2B', description: 'Child B', agent: 'Agent 3' },
          ],
        ],
        completedTasks: [],
      };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      const edgesCount = screen.getByTestId('edges-count');
      expect(edgesCount).toHaveTextContent('2'); // 2 edges: Task 1 → Task 2A, Task 1 → Task 2B
    });

    it('should create edges from all parallel parents to single child', () => {
      const plan = {
        pendingTasks: [
          [
            { task: 'Task 1A', description: 'Parent A', agent: 'Agent 1' },
            { task: 'Task 1B', description: 'Parent B', agent: 'Agent 2' },
          ],
          { task: 'Task 2', description: 'Child', agent: 'Agent 3' },
        ],
        completedTasks: [],
      };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      const edgesCount = screen.getByTestId('edges-count');
      expect(edgesCount).toHaveTextContent('2'); // 2 edges: Task 1A → Task 2, Task 1B → Task 2
    });
  });

  describe('Progress Calculation', () => {
    it('should calculate 0% progress when no tasks completed', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
          { task: 'Task 2', description: 'Task', agent: 'Agent 2' },
        ],
        completedTasks: [],
      };
      
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      
      expect(screen.getByText(/0 of 2 tasks completed/i)).toBeInTheDocument();
    });

    it('should calculate 50% progress when half completed', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
          { task: 'Task 2', description: 'Task', agent: 'Agent 2' },
        ],
        completedTasks: [],
      };
      
      const taskStatuses: Record<string, TaskStatus> = {
        'Task 1': { status: 'completed', taskName: 'Task 1', startTime: Date.now() },
        'Task 2': { status: 'pending', taskName: 'Task 2', startTime: Date.now() },
      };
      
      render(<PlanGraph planData={plan} taskStatuses={taskStatuses} />);
      
      expect(screen.getByText(/1 of 2 tasks completed/i)).toBeInTheDocument();
    });

    it('should show completion banner at 100%', async () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      const taskStatuses: Record<string, TaskStatus> = {
        'Task 1': { status: 'completed', taskName: 'Task 1', startTime: Date.now() },
      };
      
      render(<PlanGraph planData={plan} taskStatuses={taskStatuses} />);
      
      // Use findByText for async rendering with timeout
      const banner = await screen.findByText(/All tasks completed successfully/i, {}, { timeout: 3000 });
      expect(banner).toBeInTheDocument();
    });

    it('should hide progress bar at 100% completion', async () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      const taskStatuses: Record<string, TaskStatus> = {
        'Task 1': { status: 'completed', taskName: 'Task 1', startTime: Date.now() },
      };
      
      render(<PlanGraph planData={plan} taskStatuses={taskStatuses} />);
      
      // Wait for completion banner to appear
      await screen.findByText(/All tasks completed successfully/i, {}, { timeout: 3000 });
      
      // Verify progress bar text is not shown (it only appears when progress < 100)
      expect(screen.queryByText(/Progress:/i)).not.toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it.skip('should render 50 tasks in reasonable time', () => {
      const tasks = Array.from({ length: 50 }, (_, i) => ({
        task: `Task ${i + 1}`,
        description: `Description ${i + 1}`,
        agent: `Agent ${(i % 3) + 1}`,
      }));
      
      const plan = {
        pendingTasks: tasks,
        completedTasks: [],
      };
      
      const startTime = performance.now();
      render(<PlanGraph planData={plan} taskStatuses={{}} />);
      const renderTime = performance.now() - startTime;
      
      // Relaxed timing - should render in <1000ms (1 second)
      expect(renderTime).toBeLessThan(1000);
      
      const nodesCount = screen.getByTestId('nodes-count');
      expect(nodesCount).toHaveTextContent('50');
    });
  });

  describe('Error Handling', () => {
    it('should handle missing taskStatuses gracefully', () => {
      const plan = {
        pendingTasks: [
          { task: 'Task 1', description: 'Task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      // Pass undefined taskStatuses
      render(<PlanGraph planData={plan} taskStatuses={undefined} />);
      
      const node = screen.getByTestId('node-task-Task 1');
      expect(node).toHaveAttribute('data-status', 'pending');
    });

    it('should handle malformed task names', () => {
      const plan = {
        pendingTasks: [
          { task: '', description: 'Empty name task', agent: 'Agent 1' },
        ],
        completedTasks: [],
      };
      
      expect(() => {
        render(<PlanGraph planData={plan} taskStatuses={{}} />);
      }).not.toThrow();
    });
  });
});
