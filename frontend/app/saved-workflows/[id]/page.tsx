'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, Play, Copy, Trash2, Calendar, Clock, DollarSign } from 'lucide-react';
import PlanGraph from '@/components/PlanGraph';
import { toast } from 'sonner';

interface Workflow {
  workflow_id: string;
  workflow_name: string;
  workflow_description: string;
  created_at: string;
  updated_at: string;
  task_count: number;
  estimated_cost: number;
  is_public: boolean;
  conversation_history: any;
}

export default function WorkflowDetailPage() {
  const router = useRouter();
  const params = useParams();
  const workflowId = params.id as string;
  
  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);

  useEffect(() => {
    loadWorkflow();
  }, [workflowId]);

  const loadWorkflow = async () => {
    setLoading(true);
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}`);
      
      if (!response.ok) {
        throw new Error('Failed to load workflow');
      }
      
      const data = await response.json();
      setWorkflow(data);
    } catch (err) {
      console.error('Failed to load workflow:', err);
      toast.error('Failed to load workflow');
      router.push('/saved-workflows');
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteWorkflow = async () => {
    if (!workflow) return;
    
    setExecuting(true);
    try {
      // Get the original prompt from the workflow
      const originalPrompt = workflow.conversation_history?.messages?.[0]?.content || 
                           workflow.conversation_history?.original_prompt ||
                           "Execute saved workflow";
      
      toast.success('Redirecting to execute workflow...');
      
      // Navigate to home page and trigger the workflow with the saved prompt
      // The home page will handle connecting to WebSocket and executing
      router.push(`/?prompt=${encodeURIComponent(originalPrompt)}&executeNow=true`);
      
    } catch (error) {
      console.error('Error executing workflow:', error);
      toast.error('Failed to start workflow');
      setExecuting(false);
    }
  };

  const handleCloneWorkflow = async () => {
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      toast.info('Cloning workflow...');
      
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}/clone`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to clone workflow');
      }

      const data = await response.json();
      toast.success('Workflow cloned successfully');
      
      // Navigate to the cloned workflow
      if (data.workflow_id) {
        router.push(`/saved-workflows/${data.workflow_id}`);
      }
    } catch (error) {
      console.error('Error cloning workflow:', error);
      toast.error('Failed to clone workflow');
    }
  };

  const handleDeleteWorkflow = async () => {
    if (!confirm('Are you sure you want to delete this workflow?')) {
      return;
    }

    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      toast.info('Deleting workflow...');
      
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete workflow');
      }

      toast.success('Workflow deleted successfully');
      router.push('/saved-workflows');
    } catch (error) {
      console.error('Error deleting workflow:', error);
      toast.error('Failed to delete workflow');
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  if (!workflow) {
    return null;
  }

  // Extract plan from conversation history
  const plan = workflow.conversation_history?.plan || workflow.conversation_history?.task_plan || [];
  const taskStatuses = workflow.conversation_history?.task_statuses || {};

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <div className="mb-6">
        <Button
          variant="ghost"
          onClick={() => router.push('/saved-workflows')}
          className="mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Workflows
        </Button>

        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl font-bold">{workflow.workflow_name}</h1>
              {workflow.is_public && (
                <Badge variant="secondary">Public</Badge>
              )}
            </div>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {workflow.workflow_description || 'No description'}
            </p>

            {/* Metadata */}
            <div className="flex gap-6 text-sm text-gray-600 dark:text-gray-400">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4" />
                <span>{workflow.task_count || 0} tasks</span>
              </div>
              {workflow.estimated_cost && (
                <div className="flex items-center gap-2">
                  <DollarSign className="w-4 h-4" />
                  <span>${workflow.estimated_cost.toFixed(4)}</span>
                </div>
              )}
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                <span>Created {new Date(workflow.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button 
              onClick={handleExecuteWorkflow}
              disabled={executing}
              size="lg"
            >
              <Play className="w-4 h-4 mr-2" />
              {executing ? 'Starting...' : 'Run Workflow'}
            </Button>
            <Button 
              onClick={handleCloneWorkflow}
              variant="outline"
              size="lg"
            >
              <Copy className="w-4 h-4 mr-2" />
              Clone
            </Button>
            <Button 
              onClick={handleDeleteWorkflow}
              variant="destructive"
              size="lg"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Delete
            </Button>
          </div>
        </div>
      </div>

      {/* Workflow Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>Workflow Structure</CardTitle>
          <CardDescription>
            View the task execution plan and dependencies
          </CardDescription>
        </CardHeader>
        <CardContent>
          {plan && plan.length > 0 ? (
            <div className="h-[600px]">
              <PlanGraph 
                planData={plan}
                taskStatuses={taskStatuses}
              />
            </div>
          ) : (
            <div className="flex items-center justify-center h-[400px] text-gray-500">
              <div className="text-center">
                <p className="text-lg mb-2">No workflow structure available</p>
                <p className="text-sm">This workflow may not have been executed yet</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
