'use client'

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { CheckCircle, Loader2, Edit, MessageSquare, Play } from 'lucide-react';
import PlanChecklist from '@/components/PlanChecklist';
import type { TaskItem } from '@/lib/types';
import { authFetch } from '@/lib/auth-fetch';
import { toast } from 'sonner';

interface WorkflowExecutionChatProps {
  workflowId: string;
  workflow: {
    name: string;
    description: string;
    blueprint: {
      original_prompt: string;
      task_plan: any[];
      task_agent_pairs: any[];
    };
  };
  onCancel?: () => void;
}

interface Message {
  id: string;
  type: 'system' | 'user' | 'assistant' | 'plan_approval';
  content: string;
  timestamp: number;
  // New: Store full todo list snapshot if available
  todoList?: TaskItem[];
}

export default function WorkflowExecutionChat({ workflowId, workflow, onCancel }: WorkflowExecutionChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [planApproved, setPlanApproved] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);

  // New State for To-Do List
  const [todoList, setTodoList] = useState<TaskItem[]>([]);
  const [currentTaskId, setCurrentTaskId] = useState<string | undefined>(undefined);

  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Initialize generic plan from legacy blueprint if available
  useEffect(() => {
    const initialTasks: TaskItem[] = (workflow.blueprint.task_plan || []).flatMap((batch: any) => {
      const tasks = Array.isArray(batch) ? batch : [batch];
      return tasks.map((t: any, idx: number) => ({
        id: t.id || `init-${idx}-${Date.now()}`,
        description: t.description || t.task_description || t.task || "Unknown Task",
        status: 'pending',
        priority: 'medium'
      }));
    });

    // If no initial plan (new dynamic architecture), start empty or with placeholder
    if (initialTasks.length === 0) {
      initialTasks.push({
        id: 'planning',
        description: "Analyze user request and create a detailed plan",
        status: 'pending'
      });
    }

    setTodoList(initialTasks);

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
        todoList: initialTasks
      }
    ]);

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const connectWebSocket = (threadId: string) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/chat?thread_id=${threadId}`);

    ws.onopen = () => {
      console.log('WebSocket connected for workflow execution');
      addMessage('system', 'Connected to execution server. Starting workflow...');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Handle To-Do List Updates (Full Snapshot)
        if (data.type === 'todo_list_update' && data.todo_list) {
          setTodoList(data.todo_list);
        }

        // Handle Legacy/Granular Events to update local state if full snapshot missing
        if (data.type === 'task_started') {
          setCurrentTaskId(data.task_id); // Assuming backend sends task_id
          setTodoList(prev => prev.map(t =>
            t.description === data.task_name ? { ...t, status: 'in_progress' } : t
          ));
          addMessage('assistant', `🚀 Starting: ${data.task_name}`);
        }

        if (data.type === 'task_completed') {
          setTodoList(prev => prev.map(t =>
            t.description === data.task_name ? { ...t, status: 'completed', result: data.result } : t
          ));
          addMessage('assistant', `✅ Completed: ${data.task_name}`);
        }

        if (data.type === 'task_failed') {
          setTodoList(prev => prev.map(t =>
            t.description === data.task_name ? { ...t, status: 'failed', result: data.error } : t
          ));
          addMessage('assistant', `❌ Failed: ${data.task_name} - ${data.error}`);
        }

        if (data.type === 'final_response') {
          setIsExecuting(false);
          addMessage('system', '🎉 Workflow execution completed successfully!');
          if (data.content) {
            addMessage('assistant', data.content);
          }
        }

        if (data.type === 'error') {
          setIsExecuting(false);
          addMessage('system', `⚠️ Error: ${data.message || 'Execution error'}`);
        }
      } catch (err) {
        console.error('WebSocket message parse error:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      addMessage('system', '⚠️ Connection error during execution');
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setIsExecuting(false);
    };

    wsRef.current = ws;
  };

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
      // Execute workflow
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });

      if (!response.ok) {
        throw new Error('Failed to start execution');
      }

      const data = await response.json();
      const threadId = data.thread_id;

      // Connect WebSocket for real-time updates
      connectWebSocket(threadId);

      // Send execution command
      setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            type: 'execute_workflow',
            workflow_id: workflowId
          }));
        }
      }, 500);

    } catch (err) {
      console.error('Failed to execute workflow:', err);
      toast.error('Failed to start workflow execution');
      setIsExecuting(false);
      addMessage('system', '⚠️ Failed to start execution. Please try again.');
    }
  };

  const handleModify = () => {
    addMessage('user', 'Requested plan modification');
    const instruction = prompt("Enter instructions to modify the plan:");
    if (instruction) {
      // In a real scenario, this would send a message to the Orchestrator (Planning Mode)
      // For now, we simulate user feedback
      addMessage('user', instruction);
      addMessage('system', 'Feedback received. Re-planning...');
      // Ideally we POST to backend to update state.priority/tasks
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      {/* Messages Area */}
      <ScrollArea className="flex-1 p-6" ref={scrollRef}>
        <div className="space-y-4 max-w-4xl mx-auto">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.type === 'plan_approval' ? (
                <Card className="w-full max-w-2xl bg-white dark:bg-gray-900 border-2 border-blue-100 dark:border-blue-900">
                  <CardHeader className="pb-3 border-b bg-blue-50/30 dark:bg-blue-900/10">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <MessageSquare className="w-5 h-5 text-blue-600" />
                      Execution Plan Review
                    </CardTitle>
                    <CardDescription>{msg.content}</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-4">
                    {/* Plan Checklist (Preview State) */}
                    <div className="h-[300px] mb-4 border rounded-lg bg-gray-50 dark:bg-gray-950 overflow-hidden">
                      <PlanChecklist
                        todoList={msg.todoList || todoList}
                        isExecuting={false}
                      />
                    </div>

                    {/* Action Buttons */}
                    {!planApproved && (
                      <div className="flex gap-3 mt-4">
                        <Button
                          onClick={handleApprove}
                          disabled={isExecuting}
                          className="flex-1 bg-green-600 hover:bg-green-700 text-white"
                        >
                          {isExecuting ? (
                            <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Starting...</>
                          ) : (
                            <><CheckCircle className="w-4 h-4 mr-2" /> Can confirm, proceed</>
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
                      <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400 mt-2 font-medium bg-green-50 dark:bg-green-900/20 p-2 rounded">
                        <CheckCircle className="w-4 h-4" />
                        Plan approved.
                      </div>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <div className={`max-w-[80%] ${msg.type === 'user'
                  ? 'bg-blue-600 text-white shadow-md'
                  : msg.type === 'system'
                    ? 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700'
                    : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-sm'
                  } rounded-xl p-4`}>
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  <p className="text-[10px] opacity-70 mt-2 text-right">
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              )}
            </div>
          ))}

          {/* Live Execution Plan (Sticky at bottom or appended) */}
          {isExecuting && planApproved && (
            <div className="mt-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <Card className="border-blue-200 dark:border-blue-800 shadow-lg">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-white dark:from-blue-900/20 dark:to-gray-900 border-b pb-3">
                  <CardTitle className="flex items-center gap-2 text-base text-blue-700 dark:text-blue-300">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Live Execution Progress
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-[400px]">
                    <PlanChecklist
                      todoList={todoList}
                      currentTaskId={currentTaskId}
                      isExecuting={true}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Footer Status Bar */}
      <div className="border-t p-3 bg-white dark:bg-gray-950 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-10">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
            {isExecuting ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin text-blue-500" />
                <span className="font-medium text-blue-600 dark:text-blue-400">Orchestrator Active</span>
                <span className="text-xs">• Processing step {todoList.filter(t => t.status === 'completed').length + 1} of {todoList.length}</span>
              </>
            ) : planApproved ? (
              <span className="text-green-600 dark:text-green-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Execution Finished
              </span>
            ) : (
              'Waiting for plan approval...'
            )}
          </div>
          {onCancel && (
            <Button variant="ghost" size="sm" onClick={onCancel} disabled={isExecuting} className="text-gray-500 hover:text-red-600">
              {isExecuting ? 'Cancel' : 'Close'}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
