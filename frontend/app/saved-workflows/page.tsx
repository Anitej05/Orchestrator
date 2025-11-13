'use client'

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Play, Trash2, Copy, Calendar, Clock, DollarSign, Zap } from 'lucide-react';

interface Workflow {
  workflow_id: string;
  workflow_name: string;
  workflow_description: string;
  created_at: string;
  updated_at: string;
  task_count: number;
  estimated_cost: number;
  is_public: boolean;
}

export default function SavedWorkflowsPage() {
  const router = useRouter();
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      const response = await authFetch('http://localhost:8000/api/workflows');
      
      if (!response.ok) {
        throw new Error('Failed to load workflows');
      }
      
      const data = await response.json();
      setWorkflows(data);
    } catch (err) {
      console.error('Failed to load workflows:', err);
      setError(err instanceof Error ? err.message : 'Failed to load workflows');
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteWorkflow = async (workflowId: string) => {
    // TODO: Implement workflow execution
    console.log('Execute workflow:', workflowId);
  };

  const handleDeleteWorkflow = async (workflowId: string) => {
    if (!confirm('Are you sure you want to delete this workflow?')) return;
    
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete workflow');
      }
      
      // Reload workflows
      loadWorkflows();
    } catch (err) {
      console.error('Failed to delete workflow:', err);
      alert('Failed to delete workflow');
    }
  };

  const handleCloneWorkflow = async (workflowId: string) => {
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}/clone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: 'Copy of workflow' })
      });
      
      if (!response.ok) {
        throw new Error('Failed to clone workflow');
      }
      
      // Reload workflows
      loadWorkflows();
    } catch (err) {
      console.error('Failed to clone workflow:', err);
      alert('Failed to clone workflow');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-white mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading workflows...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400">{error}</p>
          <Button onClick={loadWorkflows} className="mt-4">Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Saved Workflows</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Your reusable automation workflows. Execute, edit, or delete them anytime.
        </p>
      </div>

      {workflows.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Zap className="w-16 h-16 text-gray-400 mb-4" />
            <h3 className="text-xl font-semibold mb-2">No workflows yet</h3>
            <p className="text-gray-600 dark:text-gray-400 text-center mb-4">
              Start a conversation and click "Save as Workflow" to create your first reusable workflow.
            </p>
            <Button onClick={() => router.push('/')}>
              Start Conversation
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {workflows.map((workflow) => (
            <Card key={workflow.workflow_id} className="flex flex-col">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-lg mb-1">{workflow.workflow_name}</CardTitle>
                    <CardDescription className="line-clamp-2">
                      {workflow.workflow_description || 'No description'}
                    </CardDescription>
                  </div>
                  {workflow.is_public && (
                    <Badge variant="secondary" className="ml-2">Public</Badge>
                  )}
                </div>
              </CardHeader>
              
              <CardContent className="flex-1 flex flex-col justify-between">
                <div className="space-y-2 mb-4">
                  <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                    <Clock className="w-4 h-4 mr-2" />
                    <span>{workflow.task_count} tasks</span>
                  </div>
                  
                  {workflow.estimated_cost > 0 && (
                    <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                      <DollarSign className="w-4 h-4 mr-2" />
                      <span>${workflow.estimated_cost.toFixed(4)}</span>
                    </div>
                  )}
                  
                  <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                    <Calendar className="w-4 h-4 mr-2" />
                    <span>{new Date(workflow.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
                
                <div className="flex gap-2">
                  <Button 
                    size="sm" 
                    className="flex-1"
                    onClick={() => handleExecuteWorkflow(workflow.workflow_id)}
                  >
                    <Play className="w-4 h-4 mr-1" />
                    Run
                  </Button>
                  
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => handleCloneWorkflow(workflow.workflow_id)}
                  >
                    <Copy className="w-4 h-4" />
                  </Button>
                  
                  <Button 
                    size="sm" 
                    variant="destructive"
                    onClick={() => handleDeleteWorkflow(workflow.workflow_id)}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
