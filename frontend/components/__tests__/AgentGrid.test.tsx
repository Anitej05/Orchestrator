import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import AgentGrid from '@/components/agent-grid';
import type { Agent } from '@/lib/types';

const makeAgent = (id: string, overrides: Partial<Agent> = {}): Agent => ({
  id,
  name: `Agent ${id}`,
  description: `Description for ${id}`,
  capabilities: ['reporting'],
  status: 'active',
  endpoints: [],
  ...overrides,
});

describe('AgentGrid', () => {
  it('renders a card for each agent', () => {
    const agents = [makeAgent('1'), makeAgent('2'), makeAgent('3')];
    render(<AgentGrid agents={agents} />);
    expect(screen.getByText('Agent 1')).toBeTruthy();
    expect(screen.getByText('Agent 2')).toBeTruthy();
    expect(screen.getByText('Agent 3')).toBeTruthy();
  });

  it('shows "No agents found" when agents array is empty', () => {
    render(<AgentGrid agents={[]} />);
    expect(screen.getByText('No agents found')).toBeTruthy();
  });

  it('shows "No agents found" when search has no results', () => {
    const agents = [makeAgent('1', { name: 'Data Bot' })];
    render(<AgentGrid agents={agents} searchQuery="xyz-nonexistent" />);
    expect(screen.getByText('No agents found')).toBeTruthy();
  });

  it('filters agents by name search query (case-insensitive)', () => {
    const agents = [makeAgent('1', { name: 'Alpha' }), makeAgent('2', { name: 'Beta' })];
    render(<AgentGrid agents={agents} searchQuery="alpha" />);
    expect(screen.getByText('Alpha')).toBeTruthy();
    expect(screen.queryByText('Beta')).toBeNull();
  });

  it('filters agents by description search query', () => {
    const agents = [
      makeAgent('1', { description: 'Handles invoices' }),
      makeAgent('2', { description: 'Handles emails' }),
    ];
    render(<AgentGrid agents={agents} searchQuery="invoice" />);
    expect(screen.getByText('Agent 1')).toBeTruthy();
    expect(screen.queryByText('Agent 2')).toBeNull();
  });

  it('filters agents by capability search query', () => {
    const agents = [
      makeAgent('1', { capabilities: ['data_analysis'] }),
      makeAgent('2', { capabilities: ['scheduling'] }),
    ];
    render(<AgentGrid agents={agents} searchQuery="analysis" />);
    expect(screen.getByText('Agent 1')).toBeTruthy();
    expect(screen.queryByText('Agent 2')).toBeNull();
  });

  it('shows all agents when searchQuery is empty string', () => {
    const agents = [makeAgent('1'), makeAgent('2')];
    render(<AgentGrid agents={agents} searchQuery="" />);
    expect(screen.getByText('Agent 1')).toBeTruthy();
    expect(screen.getByText('Agent 2')).toBeTruthy();
  });

  it('updates filtered results when agents prop changes', () => {
    const { rerender } = render(<AgentGrid agents={[makeAgent('1')]} searchQuery="" />);
    expect(screen.getByText('Agent 1')).toBeTruthy();

    rerender(<AgentGrid agents={[makeAgent('1'), makeAgent('2')]} searchQuery="" />);
    expect(screen.getByText('Agent 1')).toBeTruthy();
    expect(screen.getByText('Agent 2')).toBeTruthy();
  });
});
