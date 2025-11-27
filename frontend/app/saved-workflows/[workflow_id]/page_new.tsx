'use client'

import { useState, useEffect, useRef, useMemo } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, Play, Trash2, Calendar, Clock, Users, ChevronDown, ChevronUp, Loader2, MessageSquare } from 'lucide-react';
import { authFetch } from '@/lib/auth-fetch';
import { toast } from 'sonner';
import PlanGraph from '@/components/PlanGraph';
import WorkflowExecutionChat from '@/components/workflow-execution-chat';
import type { TaskStatus } from '@/lib/types';

interface WorkflowDetail {
  workflow_id: string;
  name: string;
  description: string;
  blueprint: {
    original_prompt: string;
    task_agent_pairs: Array<{
      task_name: string;
      task_description: string;
      primary: { id: string; name: string };
    }>;
    task_plan: any[];
    completed_tasks?: any[];
    created_at: string;
  };
  created_at: string;
  updated_at: string;
}

export default function WorkflowDetailPage() {
  const router = useRouter();
  const params = useParams();
  const workflowId = params?.workflow_id as string;
  
  const [workflow, setWorkflow] = useState<WorkflowDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showExecutionChat, setShowExecutionChat] = useState(false);
  
  // Legacy execution state (kept for compatibility)
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionThreadId, setExecutionThreadId] = useState<string | null>(null);
  const [taskStatuses, setTaskStatuses] = useState<Record<string, TaskStatus>>({});
  const [taskResults, setTaskResults] = useState<Array<{ task_name: string; result: any; timestamp: string }>>([]);
  const [resultsExpanded, setResultsExpanded] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (workflowId) {
      loadWorkflow();
    }
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [workflowId]);

  const loadWorkflow = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}`);
      
      if (!response.ok) {
        throw new Error('Failed to load workflow');
      }
      
      const data = await response.json();
      setWorkflow(data);
    } catch (err) {
      console.error('Failed to load workflow:', err);
      setError(err instanceof Error ? err.message : 'Failed to load workflow');
    } finally {
      setLoading(false);
    }
  };

  const connectWebSocket = (threadId: string) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/chat?thread_id=${threadId}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected for workflow execution');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'task_started') {
          setTaskStatuses(prev => ({
            ...prev,
            [data.task_name]: {
              status: 'running',
              agentName: data.agent_name,
              startTime: Date.now()
            }
          }));
        }
        
        if (data.type === 'task_completed') {
          const executionTime = data.execution_time || 0;
          setTaskStatuses(prev => ({
            ...prev,
            [data.task_name]: {
              status: 'completed',
              agentName: data.agent_name,
              executionTime,
              startTime: prev[data.task_name]?.startTime || Date.now()
            }
          }));
          
          setTaskResults(prev => [...prev, {
            task_name: data.task_name,
            result: data.result,
            timestamp: new Date().toISOString()
          }]);
        }
        
        if (data.type === 'task_failed') {
          setTaskStatuses(prev => ({
            ...prev,
            [data.task_name]: {
              status: 'failed',
              agentName: data.agent_name,
              error: data.error
            }
          }));
        }
        
        if (data.type === 'final_response') {
          setIsExecuting(false);
          toast.success('Workflow execution completed!');
        }
        
        if (data.type === 'error') {
          setIsExecuting(false);
          toast.error(data.message || 'Execution error');
        }
      } catch (err) {
        console.error('WebSocket message parse error:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('Connection error during execution');
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setIsExecuting(false);
    };

    wsRef.current = ws;
  };
  const handleExecute = async () => {
    if (!workflow) return;
    
    // Show the execution chat interface
    setShowExecutionChat(true);
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this workflow? This action cannot be undone.')) {
      return;
    }
    
    try {
      toast.info('Deleting workflow...');
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete workflow');
      }
      
      toast.success('Workflow deleted successfully');
      router.push('/saved-workflows');
    } catch (err) {
      console.error('Failed to delete workflow:', err);
      toast.error('Failed to delete workflow');
    }
  };

  // Build plan data for graph from task_plan
  const planData = useMemo(() => {
    if (!workflow) return { pendingTasks: [], completedTasks: [] };
    
    console.log('Workflow blueprint:', workflow.blueprint);
    console.log('Task plan:', workflow.blueprint.task_plan);
    console.log('Task agent pairs:', workflow.blueprint.task_agent_pairs);
    
    // Use task_plan if available (preferred), fallback to task_agent_pairs
    const taskPlan = workflow.blueprint.task_plan || [];
    
    if (taskPlan.length > 0) {
      // task_plan is an array of batches (parallel tasks)
      const allTasks: any[] = [];
      taskPlan.forEach((batch: any) => {
        if (Array.isArray(batch)) {
          batch.forEach((task: any) => {
            allTasks.push({
              task: task.task_name || task.name || task.task || 'Unnamed Task',
              description: task.task_description || task.description || '',
              agent: task.primary?.name || task.agent?.name || task.agent || 'N/A',
            });
          });
        } else if (typeof batch === 'object') {
          // Single task object
          allTasks.push({
            task: batch.task_name || batch.name || batch.task || 'Unnamed Task',
            description: batch.task_description || batch.description || '',
            agent: batch.primary?.name || batch.agent?.name || batch.agent || 'N/A',
          });
        }
      });
      
      console.log('Built tasks from task_plan:', allTasks);
      return { pendingTasks: allTasks, completedTasks: [] };
    }
    
    // Fallback to task_agent_pairs
    const pendingTasks = (workflow.blueprint.task_agent_pairs || []).map((t: any) => ({
      task: t.task_name,
      description: t.task_description,
      agent: t.primary?.name || 'N/A',
    }));
    
    console.log('Using task_agent_pairs fallback:', pendingTasks);
    return { pendingTasks, completedTasks: [] };
  }, [workflow]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-white mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading workflow...</p>
        </div>
      </div>
    );
  }

  if (error || !workflow) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400">{error || 'Workflow not found'}</p>
          <Button onClick={() => router.push('/saved-workflows')} className="mt-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Workflows
          </Button>
        </div>
      </div>
    );
  }

  const taskCount = workflow.blueprint.task_agent_pairs?.length || 0;
  const agents = Array.from(new Set(
    workflow.blueprint.task_agent_pairs?.map(t => t.primary.name) || []
  ));

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Show execution chat when running */}
      {showExecutionChat ? (
        <div>
          <Button 
            variant="ghost" 
            onClick={() => setShowExecutionChat(false)}
            className="mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Workflow Details
          </Button>
          <h1 className="text-2xl font-bold mb-4">{workflow.name} - Execution</h1>
          <WorkflowExecutionChat 
            workflowId={workflowId}
            workflow={workflow}
            onCancel={() => setShowExecutionChat(false)}
          />
        </div>
      ) : (
        <>
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
              <div>
                <h1 className="text-3xl font-bold mb-2">{workflow.name}</h1>
                <p className="text-gray-600 dark:text-gray-400 max-w-3xl">
                  {workflow.description || 'No description'}
                </p>
              </div>
              
              <div className="flex gap-2">
                <Button onClick={handleExecute} disabled={isExecuting}>
                  {isExecuting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    <>
                      <MessageSquare className="w-4 h-4 mr-2" />
                      Run Workflow
                    </>
                  )}
                </Button>
                <Button variant="destructive" onClick={handleDelete} disabled={isExecuting}>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </Button>
              </div>
            </div>
          </div>
          
          {/* Metadata Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  <Calendar className="w-4 h-4 inline mr-2" />
                  Created
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg font-semibold">
                  {new Date(workflow.created_at).toLocaleDateString()}
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  <Clock className="w-4 h-4 inline mr-2" />
                  Tasks
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg font-semibold">{taskCount}</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  <Users className="w-4 h-4 inline mr-2" />
                  Agents
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg font-semibold">{agents.length}</p>
              </CardContent>
            </Card>
          </div>

              {/* Plan Visualization - Always show */}
          <div className="mb-6">
            <Card>
              <CardHeader>
                <CardTitle>Workflow Plan Visualization</CardTitle>
                <CardDescription>
                  Task execution flow
                </CardDescription>
              </CardHeader>
              <CardContent className="h-[500px]">
                <PlanGraph planData={planData} taskStatuses={taskStatuses} />
              </CardContent>
            </Card>
          </div>

          {/* Original Prompt */}
          {workflow.blueprint.original_prompt && (
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>Original Prompt</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                  {workflow.blueprint.original_prompt}
                </p>
              </CardContent>
            </Card>
          )}

          {/* Task Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Task Breakdown</CardTitle>
              <CardDescription>
                {taskCount} task{taskCount !== 1 ? 's' : ''} in this workflow
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!workflow.blueprint.task_agent_pairs || workflow.blueprint.task_agent_pairs.length === 0 ? (
                <div className="text-center py-12">
                  <p className="text-gray-500 dark:text-gray-400 mb-4">
                    No workflow structure available
                  </p>
                  <p className="text-sm text-gray-400 dark:text-gray-500">
                    This workflow may not have been executed yet, or was saved before task tracking was implemented.
                    Try running a new conversation and saving it as a workflow.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {workflow.blueprint.task_agent_pairs?.map((task, index) => (
                    <div 
                      key={index}
                      className="flex items-start gap-4 p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
                    >
                      <div className="flex-shrink-0 w-8 h-8 bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 rounded-full flex items-center justify-center font-semibold">
                        {index + 1}
                      </div>
                      
                      <div className="flex-1">
                        <h3 className="font-semibold mb-1">{task.task_name}</h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                          {task.task_description}
                        </p>
                        <Badge variant="secondary" className="text-xs">
                          {task.primary.name}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
