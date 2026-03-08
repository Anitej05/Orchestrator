import { describe, it, expect } from 'vitest';
import { convertTasksToExecutionResults } from '@/lib/execution-utils';
import type { TaskAgentPair } from '@/lib/types';

// ── Factory ──────────────────────────────────────────────────────────────────

const makePair = (overrides: Partial<TaskAgentPair> = {}): TaskAgentPair => ({
  task_name: 'analyze_spreadsheet',
  task_description: 'Analyze the spreadsheet data',
  primary: {
    id: 'spreadsheet-1',
    name: 'Spreadsheet Agent',
    description: 'Handles spreadsheet operations',
    capabilities: ['read', 'write'],
    status: 'active',
    price_per_call_usd: 0.01,
    endpoints: [],
  },
  fallbacks: [],
  ...overrides,
});

// ── convertTasksToExecutionResults ──────────────────────────────────────────

describe('convertTasksToExecutionResults', () => {
  it('returns empty array for empty input', () => {
    expect(convertTasksToExecutionResults([])).toEqual([]);
  });

  it('returns empty array for null/undefined input', () => {
    expect(convertTasksToExecutionResults(null as any)).toEqual([]);
    expect(convertTasksToExecutionResults(undefined as any)).toEqual([]);
  });

  it('maps task_name to taskId', () => {
    const result = convertTasksToExecutionResults([makePair({ task_name: 'my_task' })]);
    expect(result[0].taskId).toBe('my_task');
  });

  it('formats task_name as human-readable taskDescription', () => {
    const result = convertTasksToExecutionResults([makePair({ task_name: 'analyze_mrp_data' })]);
    expect(result[0].taskDescription).toBe('Analyze Mrp Data');
  });

  it('uses primary agent name', () => {
    const pair = makePair();
    pair.primary.name = 'Document Agent';
    const result = convertTasksToExecutionResults([pair]);
    expect(result[0].agentName).toBe('Document Agent');
  });

  it('falls back to "Unknown Agent" when primary name is missing', () => {
    const pair = makePair();
    (pair.primary as any).name = undefined;
    const result = convertTasksToExecutionResults([pair]);
    expect(result[0].agentName).toBe('Unknown Agent');
  });

  it('always returns status "success"', () => {
    const result = convertTasksToExecutionResults([makePair()]);
    expect(result[0].status).toBe('success');
  });

  it('uses provided finalResponse as output', () => {
    const result = convertTasksToExecutionResults([makePair()], 'All done!');
    expect(result[0].output).toBe('All done!');
  });

  it('generates fallback output from task_name when no finalResponse', () => {
    const result = convertTasksToExecutionResults([makePair({ task_name: 'send_report' })]);
    expect(result[0].output).toContain('send report');
  });

  it('uses primary price_per_call_usd as cost', () => {
    const pair = makePair();
    pair.primary.price_per_call_usd = 0.05;
    const result = convertTasksToExecutionResults([pair]);
    expect(result[0].cost).toBe(0.05);
  });

  it('defaults cost to 0 when price_per_call_usd is missing', () => {
    const pair = makePair();
    (pair.primary as any).price_per_call_usd = undefined;
    const result = convertTasksToExecutionResults([pair]);
    expect(result[0].cost).toBe(0);
  });

  it('converts multiple pairs to results array preserving order', () => {
    const pairs = [
      makePair({ task_name: 'first_task' }),
      makePair({ task_name: 'second_task' }),
      makePair({ task_name: 'third_task' }),
    ];
    const results = convertTasksToExecutionResults(pairs);
    expect(results).toHaveLength(3);
    expect(results[0].taskId).toBe('first_task');
    expect(results[1].taskId).toBe('second_task');
    expect(results[2].taskId).toBe('third_task');
  });

  it('executionTime is a positive number', () => {
    const result = convertTasksToExecutionResults([makePair()]);
    expect(result[0].executionTime).toBeGreaterThan(0);
    expect(typeof result[0].executionTime).toBe('number');
  });
});
