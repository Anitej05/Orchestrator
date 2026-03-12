import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import AgentCard from '@/components/agent-card';
import type { Agent } from '@/lib/types';

const makeAgent = (overrides: Partial<Agent> = {}): Agent => ({
  id: 'agent-1',
  name: 'Test Agent',
  description: 'A helpful agent',
  capabilities: ['data_analysis', 'reporting'],
  status: 'active',
  endpoints: [],
  ...overrides,
});

describe('AgentCard', () => {
  it('renders the agent name', () => {
    render(<AgentCard agent={makeAgent({ name: 'My Agent' })} />);
    expect(screen.getByText('My Agent')).toBeTruthy();
  });

  it('renders "active" badge for active agents', () => {
    render(<AgentCard agent={makeAgent({ status: 'active' })} />);
    expect(screen.getByText('active')).toBeTruthy();
  });

  it('renders "inactive" badge for inactive agents', () => {
    render(<AgentCard agent={makeAgent({ status: 'inactive' })} />);
    expect(screen.getByText('inactive')).toBeTruthy();
  });

  it('renders up to 3 capability badges', () => {
    const agent = makeAgent({ capabilities: ['cap_a', 'cap_b', 'cap_c'] });
    render(<AgentCard agent={agent} />);
    expect(screen.getByText('cap a')).toBeTruthy();
    expect(screen.getByText('cap b')).toBeTruthy();
    expect(screen.getByText('cap c')).toBeTruthy();
  });

  it('shows "+N" overflow badge when agent has more than 3 capabilities', () => {
    const agent = makeAgent({ capabilities: ['a', 'b', 'c', 'd', 'e'] });
    render(<AgentCard agent={agent} />);
    expect(screen.getByText('+2')).toBeTruthy();
  });

  it('does not show overflow badge when capabilities <= 3', () => {
    const agent = makeAgent({ capabilities: ['a', 'b'] });
    render(<AgentCard agent={agent} />);
    expect(screen.queryByText(/^\+/)).toBeNull();
  });

  it('strips markdown bold from description', () => {
    const agent = makeAgent({ description: '**Bold** and *italic* text' });
    render(<AgentCard agent={agent} />);
    expect(screen.getByText('Bold and italic text')).toBeTruthy();
  });

  it('strips backtick code from description', () => {
    const agent = makeAgent({ description: 'Use `code` here' });
    render(<AgentCard agent={agent} />);
    expect(screen.getByText('Use code here')).toBeTruthy();
  });

  it('replaces underscores in capability names with spaces', () => {
    const agent = makeAgent({ capabilities: ['data_analysis'] });
    render(<AgentCard agent={agent} />);
    expect(screen.getByText('data analysis')).toBeTruthy();
  });
});
