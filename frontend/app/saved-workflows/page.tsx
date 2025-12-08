'use client'

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useConversationStore } from '@/lib/conversation-store';
import AppSidebar from "@/components/app-sidebar"
import Navbar from "@/components/navbar"
import { SidebarProvider, SidebarInset, SidebarTrigger, useSidebar } from "@/components/ui/sidebar"
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

function SavedWorkflowsContent() {
  const router = useRouter();
  const { open } = useSidebar();
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

  const handleExecuteWorkflow = async (workflowId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click navigation
    
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      const { toast } = await import('sonner');
      
      toast.info('Loading workflow...');
      
      // Create a new conversation pre-seeded with the workflow plan
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}/create-conversation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed to create conversation:', errorText);
        throw new Error('Failed to load workflow');
      }
      
      const data = await response.json();
      const threadId = data.thread_id;
      
      toast.success('Workflow loaded! Review the plan and click to execute.');
      
      // Load conversation and navigate to home (ChatGPT-style)
      // Update URL for sharing/bookmarking but don't full-page reload
      const { loadConversation } = useConversationStore.getState().actions;
      await loadConversation(threadId);
      
      // Update URL without navigation
      window.history.replaceState({}, '', `/c/${threadId}`);
      
      // Navigate to home where conversation is already loaded
      router.push('/');
      
    } catch (err) {
      console.error('Failed to execute workflow:', err);
      const { toast } = await import('sonner');
      toast.error('Failed to load workflow');
    }
  };

  const handleDeleteWorkflow = async (workflowId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click navigation
    if (!confirm('Are you sure you want to delete this workflow?')) return;
    
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      const { toast } = await import('sonner');
      
      toast.info('Deleting workflow...');
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete workflow');
      }
      
      toast.success('Workflow deleted successfully');
      loadWorkflows();
    } catch (err) {
      console.error('Failed to delete workflow:', err);
      const { toast } = await import('sonner');
      toast.error('Failed to delete workflow');
    }
  };

  const handleCloneWorkflow = async (workflowId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click navigation
    
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      const { toast } = await import('sonner');
      
      toast.info('Cloning workflow...');
      const response = await authFetch(`http://localhost:8000/api/workflows/${workflowId}/clone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: 'Copy of workflow' })
      });
      
      if (!response.ok) {
        throw new Error('Failed to clone workflow');
      }
      
      toast.success('Workflow cloned successfully');
      loadWorkflows();
    } catch (err) {
      console.error('Failed to clone workflow:', err);
      const { toast } = await import('sonner');
      toast.error('Failed to clone workflow');
    }
  };

  if (loading) {
    return (
      <SidebarInset className={!open ? "ml-16" : ""}>
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b dark:border-gray-700 px-6 py-4">
            <div className="flex items-center space-x-4">
              <SidebarTrigger />
            </div>
          </div>
          <main className="p-6">
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-4 text-gray-600 dark:text-gray-400">Loading workflows...</p>
              </div>
            </div>
          </main>
        </div>
      </SidebarInset>
    );
  }

  if (error) {
    return (
      <SidebarInset className={!open ? "ml-16" : ""}>
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b dark:border-gray-700 px-6 py-4">
            <div className="flex items-center space-x-4">
              <SidebarTrigger />
            </div>
          </div>
          <main className="p-6">
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <p className="text-red-600 dark:text-red-400">{error}</p>
                <Button onClick={loadWorkflows} className="mt-4">Retry</Button>
              </div>
            </div>
          </main>
        </div>
      </SidebarInset>
    );
  }

  return (
    <SidebarInset className={!open ? "ml-16" : ""}>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
        {/* Header */}
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b dark:border-gray-700 px-6 py-4">
          <div className="flex items-center space-x-4">
            <SidebarTrigger />
          </div>
        </div>

        {/* Main Content */}
        <main className="p-6">
          {/* Title Section */}
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-blue-600 dark:text-blue-400">Saved Workflows</h1>
            <p className="text-gray-600 dark:text-gray-300 mt-2">
              Your reusable automation workflows. Execute, edit, or delete them anytime.
            </p>
          </div>

          {workflows.length === 0 ? (
            <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
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
                <Card key={workflow.workflow_id} className="flex flex-col bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-blue-500 dark:hover:border-blue-400 transition-all">
                  <CardHeader 
                    className="cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                    onClick={() => router.push(`/saved-workflows/${workflow.workflow_id}`)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-lg mb-1">{workflow.workflow_name || 'Untitled Workflow'}</CardTitle>
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
                        onClick={(e) => handleExecuteWorkflow(workflow.workflow_id, e)}
                      >
                        <Play className="w-4 h-4 mr-1" />
                        Run
                      </Button>
                      
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={(e) => handleCloneWorkflow(workflow.workflow_id, e)}
                      >
                        <Copy className="w-4 h-4" />
                      </Button>
                      
                      <Button 
                        size="sm" 
                        variant="destructive"
                        onClick={(e) => handleDeleteWorkflow(workflow.workflow_id, e)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </main>
      </div>
    </SidebarInset>
  );
}

export default function SavedWorkflowsPage() {
  return (
    <>
      <Navbar />
      <SidebarProvider>
        <AppSidebar />
        <SavedWorkflowsContent />
      </SidebarProvider>
    </>
  );
}
