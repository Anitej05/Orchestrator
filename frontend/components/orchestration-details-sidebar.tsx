// components/orchestration-details-sidebar.tsx
"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { DollarSign, Clock, FileIcon, FileText, Image as ImageIcon } from "lucide-react"
import CollapsibleSection from "@/components/CollapsibleSection"
import PlanGraph from "@/components/PlanGraph"
import SaveWorkflowButton from "@/components/save-workflow-button"
import { useEffect, useState } from "react"
import { InteractiveStarRating, StarRating } from "@/components/ui/star-rating"
import { useConversationStore } from "@/lib/conversation-store"
import Markdown from '@/components/ui/markdown'
import { CanvasRenderer } from '@/components/canvas-renderer'
import type { Agent, Message, TaskAgentPair } from "@/lib/types"
import { cn } from "@/lib/utils"


interface ExecutionResult {
    taskId: string
    taskDescription: string
    agentName: string
    status: string
    output: string
    cost: number
    executionTime: number
}

interface OrchestrationDetailsSidebarProps {
    executionResults: ExecutionResult[],
    threadId: string | null;
    className?: string;
    onThreadIdUpdate?: (threadId: string) => void;
    onAcceptPlan?: (modifiedPrompt?: string) => Promise<void>;
    onRejectPlan?: () => void;
}

interface Plan {
    pendingTasks: { task: string; description: string; agent: string; }[];
    completedTasks: { task: string; result: string; }[];
}

import { forwardRef, useImperativeHandle } from 'react';

export interface OrchestrationDetailsSidebarRef {
  refreshPlan: () => void;
  viewCanvas: (canvasContent: string, canvasType: 'html' | 'markdown') => void;
}

const OrchestrationDetailsSidebar = forwardRef<OrchestrationDetailsSidebarRef, OrchestrationDetailsSidebarProps>(
 ({ executionResults, threadId, className, onThreadIdUpdate, onAcceptPlan, onRejectPlan }, ref) => {
    const [plan, setPlan] = useState<Plan>({ pendingTasks: [], completedTasks: [] });
    const [isLoadingPlan, setIsLoadingPlan] = useState(false);
    const [agents, setAgents] = useState<Agent[]>([]);
    const [activeTab, setActiveTab] = useState<string>("metadata");
    const [lastCanvasContent, setLastCanvasContent] = useState<string | undefined>(undefined);
    // State for viewing specific canvas content from messages
    const [viewedCanvasContent, setViewedCanvasContent] = useState<string | undefined>(undefined);
    const [viewedCanvasType, setViewedCanvasType] = useState<'html' | 'markdown' | undefined>(undefined);
    
    // Get conversation state from Zustand store
    const conversationState = useConversationStore();
    const taskAgentPairs = conversationState.task_agent_pairs || [];
    const messages = conversationState.messages || [];
    const uploadedFiles = conversationState.uploaded_files || [];
    const planData = conversationState.plan || [];
    const hasCanvas = conversationState.has_canvas;
    const canvasContent = conversationState.canvas_content;
    const canvasData = (conversationState as any).canvas_data;
    const canvasType = conversationState.canvas_type;
    const browserView = (conversationState as any).browser_view;
    const taskStatuses = conversationState.task_statuses || {};
    
    // Determine which canvas to display - viewed canvas takes precedence
    // Browser view is now shown in chat interface, not in canvas
    // Support both canvas_content (string) and canvas_data (structured object)
    const displayCanvasContent = viewedCanvasContent || canvasContent || canvasData;
    const displayCanvasType = viewedCanvasType || canvasType;

    // Process plan data from conversation store
    useEffect(() => {
        // ONLY use planData for the workflow structure
        // Real-time status comes from task_statuses, NOT from completed_tasks
        // This prevents duplicates in the graph
        const pendingTasks: Plan['pendingTasks'] = [];
        
        // Process pending tasks from planData (the original plan structure)
        if (planData && planData.length > 0) {
            planData.forEach((batch: any) => {
                if (Array.isArray(batch)) {
                    batch.forEach((task: any) => {
                        if (task && typeof task === 'object') {
                            pendingTasks.push({
                                task: task.task_name || 'Unknown Task',
                                description: task.task_description || 'No description',
                                agent: task.primary?.id || task.primary?.name || 'Unknown Agent'
                            });
                        }
                    });
                }
            });
        }
        
        // FALLBACK: If planData is empty but we have task_agent_pairs, use those
        // This handles older conversations where task_plan wasn't saved
        if (pendingTasks.length === 0 && taskAgentPairs && taskAgentPairs.length > 0) {
            console.log('Plan data empty, building from task_agent_pairs for visualization');
            taskAgentPairs.forEach((pair: TaskAgentPair) => {
                pendingTasks.push({
                    task: pair.task_name || 'Unknown Task',
                    description: pair.task_description || 'No description',
                    agent: pair.primary?.name || pair.primary?.id || 'Unknown Agent'
                });
            });
        }
        
        // Don't use completedTasks - status updates come from task_statuses instead
        setPlan({ 
            pendingTasks: pendingTasks, 
            completedTasks: [] // Always empty - we use task_statuses for real-time updates
        });
        setIsLoadingPlan(false);
    }, [planData, taskAgentPairs]);

    // Auto-switch to Plan tab when plan is created (validate_plan_for_execution starts)
    useEffect(() => {
        const currentStage = conversationState.metadata?.currentStage;
        
        // Switch to plan tab when validation starts or execution begins
        if (currentStage === 'validating' || currentStage === 'executing') {
            if (plan.pendingTasks.length > 0 || plan.completedTasks.length > 0) {
                console.log('Auto-switching to plan tab - execution started');
                setActiveTab('plan');
            }
        }
    }, [conversationState.metadata?.currentStage, plan.pendingTasks.length, plan.completedTasks.length]);
    
    // Auto-switch to canvas tab when NEW canvas content is created
    useEffect(() => {
        // Only switch if:
        // 1. We have canvas content OR canvas data
        // 2. The canvas content/data is different from what we've seen before (NEW content)
        // 3. Not currently executing (don't override plan view during execution)
        const currentStage = conversationState.metadata?.currentStage;
        const isExecuting = currentStage === 'executing' || currentStage === 'validating';
        const canvasData = (conversationState as any).canvas_data;
        
        // Check if we have either canvas content or canvas data
        const hasCanvasData = canvasContent || canvasData;
        
        // Create a unique identifier for the current canvas state
        // This handles both canvas_content (string) and canvas_data (object) changes
        const currentCanvasIdentifier = canvasContent || (canvasData ? JSON.stringify(canvasData) : undefined);
        
        if (hasCanvas && hasCanvasData && currentCanvasIdentifier !== lastCanvasContent && !isExecuting) {
            console.log('Auto-switching to canvas tab due to NEW canvas content/data', {
                hasCanvas,
                hasCanvasContent: !!canvasContent,
                hasCanvasData: !!canvasData,
                canvasType: conversationState.canvas_type
            });
            setActiveTab('canvas');
            setLastCanvasContent(currentCanvasIdentifier);
            // Clear any viewed canvas content to show the latest
            setViewedCanvasContent(undefined);
            setViewedCanvasType(undefined);
        }
    }, [hasCanvas, canvasContent, (conversationState as any).canvas_data, lastCanvasContent, conversationState.metadata?.currentStage]);

    useEffect(() => {
        const uniqueAgentsFromPairs = Array.from(new Set(taskAgentPairs.map((pair: TaskAgentPair) => pair.primary.id)))
            .map(id => taskAgentPairs.find((pair: TaskAgentPair) => pair.primary.id === id)!.primary);
        setAgents(uniqueAgentsFromPairs);
    }, [taskAgentPairs]);

    const handleRatingUpdate = (agentId: string, newRating: number) => {
        setAgents(prevAgents =>
            prevAgents.map(agent =>
                agent.id === agentId ? { ...agent, rating: newRating, rating_count: (agent.rating_count ?? 0) + 1 } : agent
            )
        );
    };

    // Simplified refreshPlan function that just sets loading state
    const refreshPlan = async () => {
        setIsLoadingPlan(true);
        // In a real implementation, you might want to trigger a refresh of the conversation store here
        // For now, we'll just reset the loading state
        setTimeout(() => setIsLoadingPlan(false), 500);
    };
    
    // Method to view specific canvas content from a message
    const viewCanvas = (canvasContent: string, canvasType: 'html' | 'markdown') => {
        setViewedCanvasContent(canvasContent);
        setViewedCanvasType(canvasType);
        setActiveTab('canvas');
    };
    
    // Expose methods via ref
    useImperativeHandle(ref, () => ({
        refreshPlan,
        viewCanvas
    }));


    const totalCost = executionResults.reduce((sum, result) => sum + result.cost, 0)
    const totalTime = executionResults.reduce((sum, result) => sum + result.executionTime, 0)
    const allTasks = [...plan.pendingTasks, ...plan.completedTasks];
    // Collect attachments - merge uploadedFiles with message attachments to get content
    const messageAttachments = messages.flatMap((m: Message) => m.attachments || []);
    
    // Create a map of message attachments by name for quick lookup
    const messageAttachmentMap = new Map<string, any>();
    messageAttachments.forEach(att => {
      messageAttachmentMap.set(att.name.toLowerCase(), att);
    });
    
    // Map uploadedFiles and enrich with content from message attachments if available
    const allAttachments = (uploadedFiles || []).map((file: any) => {
      const fileName = file.file_name || file.name || 'Unknown File';
      const messageAtt = messageAttachmentMap.get(fileName.toLowerCase());
      
      // For images, try to get content from message attachments or construct URL from file path
      let content = messageAtt?.content || file.content || '';
      
      // If no content but we have a file_path for an image, try to construct a URL
      if (!content && file.file_path) {
        const fileExt = fileName.toLowerCase();
        if (fileExt.endsWith('.jpg') || fileExt.endsWith('.jpeg') || 
            fileExt.endsWith('.png') || fileExt.endsWith('.gif') || 
            fileExt.endsWith('.webp')) {
          // Construct URL to serve the image from backend
          content = `http://localhost:8000/api/files/${encodeURIComponent(file.file_path)}`;
        }
      }
      
      return {
        name: fileName,
        type: file.file_type || file.type || 'unknown',
        content: content
      };
    });

    const hasResults = executionResults.length > 0 || allTasks.length > 0

    return (
        <aside className={cn("border-l border-gray-200 dark:border-gray-800/30 bg-white dark:bg-gray-900 p-4 flex flex-col h-full", className)}>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
                <TabsList className="grid w-full grid-cols-4 bg-gray-100/80 dark:bg-gray-800/50 backdrop-blur-xl border border-gray-200/50 dark:border-gray-700/50 shadow-lg">
                    <TabsTrigger value="metadata">Metadata</TabsTrigger>
                    <TabsTrigger value="plan" className="relative">
                        Plan
                        {(conversationState.metadata?.currentStage === 'executing' || 
                          conversationState.metadata?.currentStage === 'validating') && (
                            <span className="ml-1 px-1.5 py-0.5 text-[10px] font-semibold bg-blue-500 dark:bg-blue-600 text-white rounded-full animate-pulse">
                                Running
                            </span>
                        )}
                    </TabsTrigger>
                    <TabsTrigger value="attachments">Attachments</TabsTrigger>
                    <TabsTrigger value="canvas">Canvas</TabsTrigger>
                </TabsList>
                <TabsContent value="metadata" className="flex-1 overflow-y-auto mt-4 space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>Execution Summary</CardTitle>
                            <CardDescription>Overview of task execution and performance metrics</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            {isLoadingPlan ? (
                                <p className="text-sm text-gray-500 text-center py-4">Loading plan...</p>
                            ) : hasResults ? (
                                <>
                                    <Table>
                                        <TableBody>
                                            <TableRow>
                                                <TableCell><span className="text-gray-600 flex items-center"><DollarSign className="w-4 h-4 mr-2" /> Total Cost</span></TableCell>
                                                <TableCell className="text-right font-semibold">${totalCost.toFixed(4)}</TableCell>
                                            </TableRow>
                                            <TableRow>
                                                <TableCell><span className="text-gray-600 flex items-center"><Clock className="w-4 h-4 mr-2" /> Total Time</span></TableCell>
                                                <TableCell className="text-right font-semibold">{totalTime.toFixed(1)}s</TableCell>
                                            </TableRow>
                                        </TableBody>
                                    </Table>
                                    <CollapsibleSection title="Agents Used" count={agents.length}>
                                        <div className="space-y-3 pt-2">
                                            {agents.map((agent) => (
                                                <div key={agent.id} className="flex items-center justify-between text-sm">
                                                    <span>{agent.name}</span>
                                                    <div className="flex items-center justify-end space-x-2">
                                                        <StarRating currentRating={agent.rating} readonly size="sm" showValue={false}/>
                                                        <span className="text-xs text-gray-500 w-16 text-right">
                                                            ({(agent.rating ?? 0).toFixed(1)} / {agent.rating_count ?? 0})
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </CollapsibleSection>
                                    <CollapsibleSection title="Tasks" count={allTasks.length}>
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>Task</TableHead>
                                                    <TableHead>Description</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {allTasks.map((task, index) => (
                                                    <TableRow key={`${task.task}-${index}`}>
                                                        <TableCell className="font-semibold">{task.task}</TableCell>
                                                        <TableCell className="text-gray-600">{"description" in task && task.description}</TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </CollapsibleSection>
                                </>
                            ) : (
                                <p className="text-sm text-gray-500 text-center py-4">Run a workflow to see the summary.</p>
                            )}
                        </CardContent>
                    </Card>

                    {hasResults && agents.length > 0 && (
                        <Card>
                            <CardHeader>
                                <CardTitle>Rate Agents</CardTitle>
                                <CardDescription>Provide feedback on the agents used in this workflow.</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    {agents.map((agent) => (
                                        <div
                                            key={agent.id}
                                            className="flex items-center justify-between"
                                        >
                                            <span className="text-sm font-medium">{agent.name}</span>
                                            <InteractiveStarRating
                                                agentId={agent.id}
                                                agentName={agent.name}
                                                currentRating={agent.rating}
                                                onRatingUpdate={(newRating) => handleRatingUpdate(agent.id, newRating)}
                                            />
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </TabsContent>
                <TabsContent value="plan" className="flex-1 flex flex-col">
                    {/* Plan Tab Header with Save Workflow Button */}
                    <div className="flex items-center justify-between mt-4 px-4 py-3">
                        <div className="flex-1">
                            <div className="flex items-center gap-2">
                                <h3 className="text-xl font-semibold">Workflow Visualization</h3>
                                {plan.pendingTasks.length > 0 && (() => {
                                    const completedCount = Object.values(taskStatuses).filter((t: any) => t.status === 'completed').length;
                                    const totalTasks = plan.pendingTasks.length;
                                    return (
                                        <span className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded-full font-medium">
                                            {completedCount} / {totalTasks} tasks
                                        </span>
                                    );
                                })()}
                            </div>
                            <p className="text-sm text-gray-500 mt-1">
                                {conversationState.metadata?.currentStage === 'executing' 
                                    ? `Executing tasks... ${conversationState.current_executing_task ? `(${conversationState.current_executing_task})` : ''}`
                                    : conversationState.metadata?.currentStage === 'validating'
                                    ? 'Validating execution plan...'
                                    : 'View workflow structure'}
                            </p>
                        </div>
                        
                        {/* Approval buttons have been moved to the interactive chat interface */}
                        
                        {/* Show save button after execution completes */}
                        {conversationState.status === 'completed' && (
                            <SaveWorkflowButton 
                                threadId={threadId || ''} 
                                disabled={!threadId || plan.pendingTasks.length === 0}
                            />
                        )}
                    </div>
                    
                    {/* Real-time Graph with Task Statuses */}
                    <div className="flex-1 flex items-center justify-center">
                        <PlanGraph 
                            key={JSON.stringify(plan)} 
                            planData={plan}
                            taskStatuses={conversationState.task_statuses || {}}
                        />
                    </div>
                </TabsContent>
                <TabsContent value="attachments" className="flex-1 overflow-y-auto mt-4">
                    {allAttachments.length > 0 ? (
                        <div className="grid grid-cols-2 gap-3">
                        {allAttachments.map((att: any, index: number) => {
                            // Check both type and file extension for images
                            const fileName = att.name.toLowerCase();
                            const isImageType = att.type.startsWith('image/');
                            const isImageExt = fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || 
                                             fileName.endsWith('.png') || fileName.endsWith('.gif') || 
                                             fileName.endsWith('.webp') || fileName.endsWith('.svg');
                            const isImage = isImageType || isImageExt;
                            const isPdf = fileName.endsWith('.pdf');
                            const isDoc = fileName.endsWith('.doc') || fileName.endsWith('.docx');
                            const isExcel = fileName.endsWith('.xls') || fileName.endsWith('.xlsx');
                            
                            return (
                                <div key={`${att.name}-${index}`} className="flex flex-col items-center p-3 rounded-xl bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:shadow-lg hover:scale-105 transition-all duration-200">
                                    {isImage && att.content ? (
                                        <div className="w-full aspect-square rounded-lg overflow-hidden bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-900 dark:to-black mb-2">
                                            <img src={att.content} alt={att.name} className="w-full h-full object-cover" />
                                        </div>
                                    ) : (
                                        <div className="w-full aspect-square rounded-md bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center mb-2">
                                            {isPdf ? (
                                                <FileText className="w-16 h-16 text-red-500" />
                                            ) : isDoc ? (
                                                <FileText className="w-16 h-16 text-blue-500" />
                                            ) : isExcel ? (
                                                <FileText className="w-16 h-16 text-green-500" />
                                            ) : isImage ? (
                                                <ImageIcon className="w-16 h-16 text-purple-500" />
                                            ) : (
                                                <FileIcon className="w-16 h-16 text-gray-400" />
                                            )}
                                        </div>
                                    )}
                                    <span className="text-xs text-center text-gray-700 font-medium truncate w-full" title={att.name}>
                                        {att.name}
                                    </span>
                                </div>
                            );
                        })}
                        </div>
                    ) : (
                        <div className="text-center text-gray-500 py-8">
                            <FileIcon className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                            <p className="font-semibold">No Attachments</p>
                            <p className="text-sm mt-2">Files you upload will appear here.</p>
                        </div>
                    )}
                </TabsContent>
                <TabsContent value="canvas" className="flex-1 overflow-y-auto mt-4">
                    {/* Canvas now only shows non-browser content (HTML/Markdown from LLM responses) */}
                    {/* Browser live stream is shown in the chat interface instead */}
                    {(hasCanvas || viewedCanvasContent) && displayCanvasContent && !browserView ? (
                        <div className="h-full flex flex-col">
                            {viewedCanvasContent && (
                                <div className="bg-gray-800 dark:bg-gray-900 border-b border-gray-700 dark:border-gray-800 px-6 py-3 shadow-md">
                                    <div className="flex items-center justify-between max-w-4xl mx-auto">
                                        <div className="flex items-center gap-3">
                                            <span className="text-2xl">📌</span>
                                            <div>
                                                <div className="text-sm font-semibold text-gray-200 dark:text-gray-300">Viewing Previous Canvas</div>
                                                <div className="text-xs text-gray-400 dark:text-gray-500">From an earlier message in the conversation</div>
                                            </div>
                                        </div>
                                        <button 
                                            onClick={() => {
                                                setViewedCanvasContent(undefined);
                                                setViewedCanvasType(undefined);
                                            }}
                                            className="px-4 py-2 text-sm font-medium text-gray-200 dark:text-gray-300 bg-gray-700 dark:bg-gray-800 hover:bg-gray-600 dark:hover:bg-gray-700 rounded-lg transition-all duration-200 border border-gray-600 dark:border-gray-700 hover:scale-105 active:scale-95"
                                        >
                                            ← Return to Latest
                                        </button>
                                    </div>
                                </div>
                            )}
                            <div className="flex-1 overflow-auto">
                                {(() => {
                                    // Check if canvas_data has status='preview' which indicates confirmation is needed
                                    const canvasData = (conversationState as any).canvas_data;
                                    const isPreview = canvasData && canvasData.status === 'preview';
                                    const requiresConf = isPreview || ((conversationState as any).canvas_requires_confirmation && (conversationState as any).pending_confirmation);
                                    console.log('🔘 Sidebar confirmation check:', {
                                        canvas_requires_confirmation: (conversationState as any).canvas_requires_confirmation,
                                        pending_confirmation: (conversationState as any).pending_confirmation,
                                        canvasDataStatus: canvasData?.status,
                                        isPreview,
                                        requiresConfirmation: requiresConf,
                                        pending_confirmation_task: (conversationState as any).pending_confirmation_task
                                    });
                                    return (
                                        <CanvasRenderer
                                            canvasType={displayCanvasType as any}
                                            canvasContent={typeof displayCanvasContent === 'string' ? displayCanvasContent : undefined}
                                            canvasData={typeof displayCanvasContent === 'object' ? displayCanvasContent : (conversationState as any).canvas_data}
                                            canvasTitle={conversationState.canvas_title}
                                            canvasMetadata={conversationState.canvas_metadata}
                                            requiresConfirmation={requiresConf}
                                            confirmationMessage={(conversationState as any).canvas_confirmation_message}
                                            onConfirm={async () => {
                                                // User confirmed - send confirmation to continue execution
                                                const taskName = (conversationState as any).pending_confirmation_task?.task_name;
                                                console.log('User confirmed canvas action for task:', taskName);
                                                const { sendCanvasConfirmation } = useConversationStore.getState().actions;
                                                await sendCanvasConfirmation('confirm', taskName);
                                            }}
                                            onCancel={async () => {
                                                // User cancelled - abort the action
                                                const taskName = (conversationState as any).pending_confirmation_task?.task_name;
                                                console.log('User cancelled canvas action for task:', taskName);
                                                const { sendCanvasConfirmation } = useConversationStore.getState().actions;
                                                await sendCanvasConfirmation('cancel', taskName);
                                            }}
                                            onUndo={async () => {
                                                // User clicked undo - send undo command
                                                const canvasData = typeof displayCanvasContent === 'object' ? displayCanvasContent : (conversationState as any).canvas_data;
                                                const filePath = canvasData?.file_path;
                                                if (filePath) {
                                                    console.log('User requested undo for document:', filePath);
                                                    const { sendMessage } = useConversationStore.getState().actions;
                                                    await sendMessage(`Undo the last edit to ${filePath}`, []);
                                                }
                                            }}
                                            onRedo={async () => {
                                                // User clicked redo - send redo command
                                                const canvasData = typeof displayCanvasContent === 'object' ? displayCanvasContent : (conversationState as any).canvas_data;
                                                const filePath = canvasData?.file_path;
                                                if (filePath) {
                                                    console.log('User requested redo for document:', filePath);
                                                    const { sendMessage } = useConversationStore.getState().actions;
                                                    await sendMessage(`Redo the last undone edit to ${filePath}`, []);
                                                }
                                            }}
                                            onShowHistory={async () => {
                                                // User clicked history - show version history
                                                const canvasData = typeof displayCanvasContent === 'object' ? displayCanvasContent : (conversationState as any).canvas_data;
                                                const filePath = canvasData?.file_path;
                                                if (filePath) {
                                                    console.log('User requested version history for document:', filePath);
                                                    const { sendMessage } = useConversationStore.getState().actions;
                                                    await sendMessage(`Show version history for ${filePath}`, []);
                                                }
                                            }}
                                        />
                                    );
                                })()}
                            </div>
                        </div>
                    ) : (
                        <div className="text-center text-gray-500 py-8">
                            <p className="font-semibold">No Canvas Content</p>
                            <p className="text-sm mt-2">Interactive content from responses will appear here.</p>
                            <p className="text-xs mt-1 text-gray-400">Browser live view is shown in the chat area.</p>
                        </div>
                    )}
                </TabsContent>
            </Tabs>
        </aside>
    )
});

export default OrchestrationDetailsSidebar;
