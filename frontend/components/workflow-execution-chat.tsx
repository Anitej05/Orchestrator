'use client'

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { CheckCircle, XCircle, Loader2, Play, Edit, MessageSquare } from 'lucide-react';
import PlanGraph from '@/components/PlanGraph';
import type { TaskStatus } from '@/lib/types';
import { authFetch } from '@/lib/auth-fetch';
import { toast } from 'sonner';
import { useTaskExecutionWebSocket } from '@/hooks/use-task-execution-websocket';
import { API_BASE_URL } from '@/lib/config';

interface WorkflowExecutionChatProps {
  workflowId: string;
  workflow: {
    name: string;
    description: string;
    blueprint: {
      original_prompt: string;
      task_plan: any[];
      task_agent_pairs: any[];
      todo_list?: any[];
    };
  };
  onCancel?: () => void;
}

interface Message {
  id: string;
  type: 'system' | 'user' | 'assistant' | 'plan_approval';
  content: string;
  timestamp: number;
  planData?: any;
  taskStatuses?: Record<string, TaskStatus>;
}

export default function WorkflowExecutionChat({ workflowId, workflow, onCancel }: WorkflowExecutionChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [planApproved, setPlanApproved] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionThreadId, setExecutionThreadId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [taskStatuses, setTaskStatuses] = useState<Record<string, TaskStatus>>({});

  // Initialize WebSocket hook
  const { connect, disconnect, taskStatuses: wsTaskStatuses } = useTaskExecutionWebSocket({
    onMessage: (data) => {
      if (data.type === 'task_status') {
        setTaskStatuses(prev => ({
          ...prev,
          [data.task_name]: data
        }));
      }
    }
  });

  // Build plan data from todo_list (new system) or task_plan (old system)
  const planData = {
    pendingTasks: workflow.blueprint.todo_list?.map((task: any) => ({
      task: task.title || 'Untitled Task',
      description: task.description || task.instructions || '',
      agent: task.assigned_to || 'N/A',
      status: task.status || 'pending',
      id: task.task_id || task.id
    })) || 
    workflow.blueprint.task_plan?.flatMap((batch: any[]) => 
      batch.map((task: any) => ({
        task: task.task_name || task.name,
        description: task.task_description || task.description || '',
        agent: task.primary?.name || task.agent?.name || 'N/A',
      }))
    ) || [],
    completedTasks: []
  };

  useEffect(() => {
    // Show initial plan approval message
    setMessages([
      {
        id: 'welcome',
        type: 'system',
        content: `Ready to execute workflow: ${workflow.name}`,
        timestamp: Date.now()
      },
      {
        id: 'plan-approval',
        type: 'plan_approval',
        content: 'Please review the execution plan below. Click "Approve & Execute" to run the workflow, or "Modify Plan" to make changes.',
        timestamp: Date.now(),
        planData,
        taskStatuses: {}
      }
    ]);

    return () => {
      if (disconnect) disconnect();
    };
  }, []);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Removed inline WebSocket implementation - now using useTaskExecutionWebSocket hook

  const addMessage = (type: Message['type'], content: string) => {
    setMessages(prev => [...prev, {
      id: `msg-${Date.now()}-${Math.random()}`,
      type,
      content,
      timestamp: Date.now()
    }]);
  };

  const handleApprove = async () => {
    setPlanApproved(true);
    setIsExecuting(true);
    addMessage('user', '✓ Plan approved. Starting execution...');
    
    try {
      // Execute workflow using the saved task_plan directly
      const response = await authFetch(`${API_BASE_URL}/api/workflows/${workflowId}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      
      if (!response.ok) {
        throw new Error('Failed to start execution');
      }
      
      const data = await response.json();
      const threadId = data.thread_id;
      setExecutionThreadId(threadId);
      
      // Connect WebSocket for real-time updates using the hook
      connect(threadId);
      
    } catch (err) {
      console.error('Failed to execute workflow:', err);
      toast.error('Failed to start workflow execution');
      setIsExecuting(false);
      addMessage('system', '⚠️ Failed to start execution. Please try again.');
    }
  };

  const handleModify = () => {
    addMessage('user', 'Requested plan modification');
    addMessage('system', '🔧 Plan modification feature coming soon! For now, you can create a new conversation and re-plan the workflow.');
    // TODO: Implement modification flow - send to orchestrator for re-planning
  };

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      {/* Messages Area */}
      <ScrollArea className="flex-1 p-6" ref={scrollRef}>
        <div className="space-y-4 max-w-4xl mx-auto">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.type === 'plan_approval' ? (
                <Card className="w-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <MessageSquare className="w-5 h-5" />
                      Execution Plan Review
                    </CardTitle>
                    <CardDescription>{msg.content}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {/* Plan Graph */}
                    <div className="h-[400px] mb-4 border rounded-lg overflow-hidden">
                      <PlanGraph 
                        planData={msg.planData} 
                        todoList={msg.planData?.todoList || workflow.blueprint.todo_list}
                      />
                    </div>
                    
                    {/* Plan Details */}
                    <div className="mb-4">
                      <h4 className="font-semibold mb-2">Tasks ({msg.planData?.pendingTasks?.length || 0})</h4>
                      <div className="space-y-2 max-h-48 overflow-y-auto">
                        {msg.planData?.pendingTasks?.map((task: any, idx: number) => (
                          <div key={`task-${idx}-${task.task || ''}`} className="flex items-start gap-2 text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                            <span className="font-semibold min-w-6">{idx + 1}.</span>
                            <div className="flex-1">
                              <p className="font-medium">{task.task}</p>
                              <p className="text-gray-600 dark:text-gray-400 text-xs">{task.description}</p>
                              <Badge variant="secondary" className="mt-1 text-xs">{task.agent}</Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    {/* Action Buttons */}
                    {!planApproved && (
                      <div className="flex gap-3">
                        <Button 
                          onClick={handleApprove} 
                          disabled={isExecuting}
                          className="flex-1"
                        >
                          {isExecuting ? (
                            <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Executing...</>
                          ) : (
                            <><CheckCircle className="w-4 h-4 mr-2" /> Approve & Execute</>
                          )}
                        </Button>
                        <Button 
                          onClick={handleModify} 
                          variant="outline"
                          disabled={isExecuting}
                          className="flex-1"
                        >
                          <Edit className="w-4 h-4 mr-2" />
                          Modify Plan
                        </Button>
                      </div>
                    )}
                    
                    {planApproved && (
                      <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
                        <CheckCircle className="w-4 h-4" />
                        Plan approved and executing...
                      </div>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <div className={`max-w-[80%] ${
                  msg.type === 'user' 
                    ? 'bg-blue-500 text-white' 
                    : msg.type === 'system'
                    ? 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
                    : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
                } rounded-lg p-3 shadow-sm`}>
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  <p className="text-xs opacity-70 mt-1">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              )}
            </div>
          ))}
          
          {/* Live Plan Graph Update (when executing) */}
          {isExecuting && planApproved && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Live Execution Progress
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <PlanGraph 
                    planData={planData} 
                    todoList={workflow.blueprint.todo_list}
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </ScrollArea>
      
      {/* Footer Actions */}
      <div className="border-t p-4 bg-gray-50 dark:bg-gray-900">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {isExecuting ? (
              <span className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Executing workflow...
              </span>
            ) : planApproved ? (
              'Execution complete'
            ) : (
              'Awaiting plan approval'
            )}
          </div>
          {onCancel && (
            <Button variant="outline" onClick={onCancel} disabled={isExecuting}>
              {isExecuting ? 'Cancel Execution' : 'Close'}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

