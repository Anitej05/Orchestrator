// components/orchestration-details-sidebar.tsx
"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { DollarSign, Clock, FileIcon, FileText, Image as ImageIcon } from "lucide-react"
import PlanGraph from "@/components/PlanGraph"
import SaveWorkflowButton from "@/components/save-workflow-button"
import { useEffect, useState } from "react"
import { useConversationStore } from "@/lib/conversation-store"
import Markdown from '@/components/ui/markdown'
import { CanvasRenderer } from '@/components/canvas-renderer'
import type { Agent, Message, TaskAgentPair } from "@/lib/types"
import { cn } from "@/lib/utils"
import { API_BASE_URL } from "@/lib/config"
import DocumentViewer from "@/components/document-viewer"


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
    // Modified to support parallel execution batches (array of arrays)
    pendingTasks: { task: string; description: string; agent: string; }[][] | { task: string; description: string; agent: string; }[];
    completedTasks: { task: string; result: string; }[];
}

import { forwardRef, useImperativeHandle } from 'react';

export interface OrchestrationDetailsSidebarRef {
    refreshPlan: () => void;
    viewCanvas: (canvasContent: string, canvasType: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json') => void;
}

const OrchestrationDetailsSidebar = forwardRef<OrchestrationDetailsSidebarRef, OrchestrationDetailsSidebarProps>(
    ({ executionResults, threadId, className, onThreadIdUpdate, onAcceptPlan, onRejectPlan }, ref) => {
        const [plan, setPlan] = useState<Plan>({ pendingTasks: [], completedTasks: [] });
        const [isLoadingPlan, setIsLoadingPlan] = useState(false);
        const [activeTab, setActiveTab] = useState<string>("plan");
        const [lastCanvasContent, setLastCanvasContent] = useState<string | undefined>(undefined);
        // State for viewing specific canvas content from messages
        const [viewedCanvasContent, setViewedCanvasContent] = useState<string | undefined>(undefined);
        const [viewedCanvasType, setViewedCanvasType] = useState<'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json' | undefined>(undefined);
        // State for viewing specific attachment file
        const [viewingFile, setViewingFile] = useState<any | null>(null);

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
            // We want to PRESERVE the batch structure (list of lists) for valid parallel visualization
            if (planData && planData.length > 0) {
                planData.forEach((batch: any) => {
                    if (Array.isArray(batch)) {
                        // It's a batch of tasks
                        const batchTasks = batch.map((task: any) => ({
                            task: task.task_name || 'Unknown Task',
                            description: task.task_description || 'No description',
                            agent: task.primary?.id || task.primary?.name || 'Unknown Agent',
                            short_description: task.short_description,  // AI-generated summary
                            agent_image_url: task.primary?.image_url    // Agent avatar URL
                        }));
                        // @ts-ignore - We are changing the type of pendingTasks to allow nested arrays
                        pendingTasks.push(batchTasks);
                    } else if (typeof batch === 'object') {
                        // Handle edge case where it might not be an array (though it should be)
                        // @ts-ignore
                        pendingTasks.push([{
                            task: batch.task_name || 'Unknown Task',
                            description: batch.task_description || batch.short_description || 'No description',
                            agent: batch.primary?.id || batch.primary?.name || 'Unknown Agent'
                        }]);
                    }
                });
            }

            // // FALLBACK: If planData is empty but we have task_agent_pairs, use those
            // // This handles older conversations where task_plan wasn't saved
            // if (pendingTasks.length === 0 && taskAgentPairs && taskAgentPairs.length > 0) {
            //     console.log('Plan data empty, building from task_agent_pairs for visualization');
            //     // task_agent_pairs is flat, so we treat each as a single-item batch
            //     taskAgentPairs.forEach((pair: TaskAgentPair) => {
            //         // @ts-ignore
            //         pendingTasks.push([{
            //             task: pair.task_name || 'Unknown Task',
            //             description: pair.task_description || 'No description',
            //             agent: pair.primary?.name || pair.primary?.id || 'Unknown Agent'
            //         }]);
            //     });
            // }

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
                if (plan.pendingTasks.flat().length > 0 || plan.completedTasks.length > 0) {
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

        // Method to view specific canvas content from a message
        const viewCanvas = (canvasContent: string, canvasType: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json') => {
            setViewedCanvasContent(canvasContent);
            setViewedCanvasType(canvasType);
            setActiveTab('canvas');
        };

        // Expose methods via ref
        useImperativeHandle(ref, () => ({
            refreshPlan: async () => {
                setIsLoadingPlan(true);
                setTimeout(() => setIsLoadingPlan(false), 500);
            },
            viewCanvas
        }));


        const totalCost = executionResults.reduce((sum, result) => sum + result.cost, 0)
        const totalTime = executionResults.reduce((sum, result) => sum + result.executionTime, 0)
        const allTasks = [...plan.pendingTasks.flat(), ...plan.completedTasks];
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
                    content = `${API_BASE_URL}/api/files/${encodeURIComponent(file.file_path)}`;
                }
            }

            return {
                name: fileName,
                type: file.file_type || file.type || 'unknown',
                content: content,
                file_path: file.file_path
            };
        });

        const hasResults = executionResults.length > 0 || allTasks.length > 0

        return (
            <aside className={cn("border-l border-border-color bg-bg-card text-text-primary p-4 flex flex-col h-full overflow-hidden", className)}>
                <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col overflow-hidden min-w-0">
                    <TabsList className="grid w-full grid-cols-3 bg-bg-subtle/80 backdrop-blur-xl border border-border-color/50 shadow-orbimesh-panel">
                        <TabsTrigger value="plan" className="relative">
                            Plan
                            {(conversationState.metadata?.currentStage === 'executing' ||
                                conversationState.metadata?.currentStage === 'validating') && (
                                    <span className="ml-1 px-1.5 py-0.5 text-[10px] font-semibold bg-status-pending text-foreground rounded-full animate-pulse">
                                        Running
                                    </span>
                                )}
                        </TabsTrigger>
                        <TabsTrigger value="attachments">Attachments</TabsTrigger>
                        <TabsTrigger value="canvas">Canvas</TabsTrigger>
                    </TabsList>

                    <TabsContent value="plan" className="flex-1 flex flex-col">
                        {/* Plan Tab Header with Save Workflow Button */}
                        <div className="flex items-center justify-between mt-4 px-4 py-3">
                            <div className="flex-1">
                                <div className="flex items-center gap-2">
                                    <h3 className="text-orbimesh-section-header">Workflow Visualization</h3>
                                    {plan.pendingTasks.length > 0 && (() => {
                                        const completedCount = Object.values(taskStatuses).filter((t: any) => t.status === 'completed').length;
                                        const totalTasks = plan.pendingTasks.flat().length;
                                        return (
                                            <span className="text-orbimesh-badge px-2 py-1 bg-bg-subtle text-text-secondary rounded-full font-medium border border-border-color">
                                                {completedCount} / {totalTasks} tasks
                                            </span>
                                        );
                                    })()}
                                </div>
                                <p className="text-orbimesh-section-subtitle text-text-tertiary mt-1">
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
                                    disabled={!threadId || plan.pendingTasks.flat().length === 0}
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
                    <TabsContent value="attachments" className="flex-1 overflow-hidden mt-4 min-w-0 max-w-full">
                        {viewingFile ? (
                            <div className="h-full overflow-hidden max-w-full">
                                <DocumentViewer
                                    file={viewingFile}
                                    onBack={() => setViewingFile(null)}
                                />
                            </div>
                        ) : allAttachments.length > 0 ? (
                            <div className="grid grid-cols-[repeat(auto-fill,minmax(144px,144px))] gap-2 overflow-y-auto overflow-x-hidden h-full justify-start p-2">
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
                                        <div 
                                            key={`${att.name}-${index}`} 
                                            className="flex flex-col items-center p-2 rounded-lg bg-bg-card border border-border-color hover:shadow-orbimesh-card-hover transition-all duration-200 cursor-pointer w-[144px] h-[168px]"
                                            onClick={() => setViewingFile(att)}
                                        >
                                            {isImage && att.content ? (
                                                <div className="w-[112px] h-[112px] rounded-md overflow-hidden bg-bg-subtle border border-border-color mb-2 flex-shrink-0">
                                                    <img src={att.content} alt={att.name} className="w-full h-full object-cover" />
                                                </div>
                                            ) : (
                                                <div className="w-[112px] h-[112px] rounded-md bg-bg-subtle border border-border-color flex items-center justify-center mb-2 flex-shrink-0">
                                                    {isPdf ? (
                                                        <FileText className="w-14 h-14 text-status-error" />
                                                    ) : isDoc ? (
                                                        <FileText className="w-14 h-14 text-status-active" />
                                                    ) : isExcel ? (
                                                        <FileText className="w-14 h-14 text-status-success" />
                                                    ) : isImage ? (
                                                        <ImageIcon className="w-14 h-14 text-status-pending" />
                                                    ) : (
                                                        <FileIcon className="w-14 h-14 text-text-disabled" />
                                                    )}
                                                </div>
                                            )}
                                            <span className="text-[10px] text-text-secondary text-center font-medium line-clamp-2 w-full leading-tight px-1" title={att.name}>
                                                {att.name}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        ) : (
                            <div className="text-center text-text-tertiary py-8">
                                <FileIcon className="w-12 h-12 mx-auto mb-2 text-text-disabled" />
                                <p className="text-orbimesh-section-header font-semibold">No Attachments</p>
                                <p className="text-orbimesh-section-subtitle mt-2">Files you upload will appear here.</p>
                            </div>
                        )}
                    </TabsContent>
                    <TabsContent value="canvas" className="flex-1 overflow-hidden mt-4 flex flex-col">
                        {/* Canvas now only shows non-browser content (HTML/Markdown from LLM responses) */}
                        {/* Browser live stream is shown in the chat interface instead */}
                        {(hasCanvas || viewedCanvasContent) && displayCanvasContent && !browserView ? (
                            <div className="h-full flex flex-col">
                                {viewedCanvasContent && (
                                    <div className="bg-bg-card border-b border-border-color px-6 py-3 shadow-orbimesh-panel">
                                        <div className="flex items-center justify-between max-w-4xl mx-auto">
                                            <div className="flex items-center gap-3">
                                                <span className="text-2xl">📌</span>
                                                <div>
                                                    <div className="text-orbimesh-section-header font-semibold text-text-primary">Viewing Previous Canvas</div>
                                                    <div className="text-orbimesh-section-subtitle text-text-tertiary">From an earlier message in the conversation</div>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => {
                                                    setViewedCanvasContent(undefined);
                                                    setViewedCanvasType(undefined);
                                                }}
                                                className="px-4 py-2 text-orbimesh-tab font-medium text-text-primary bg-bg-subtle hover:bg-bg-hover rounded-lg transition-all duration-200 border border-border-color hover:scale-105 active:scale-95"
                                            >
                                                ← Return to Latest
                                            </button>
                                        </div>
                                    </div>
                                )}
                                <div className="flex-1 overflow-auto">
                                    {(() => {
                                        // Check if canvas_data has status='preview' which indicates confirmation is needed
                                        const isPreview = canvasData && canvasData.status === 'preview';
                                        const requiresConf = isPreview || ((conversationState as any).canvas_requires_confirmation && (conversationState as any).pending_confirmation);

                                        // DEBUG: Reduced Sidebar logging
                                        try {
                                            if (canvasData) {
                                                console.log('🔴 SIDEBAR_CANVAS_DATA_EXISTS:', {
                                                    type: canvasData.type,
                                                    keys: Object.keys(canvasData)
                                                });
                                            }
                                        } catch (e) { }

                                        const data = (conversationState as any).canvas_data;

                                        // Auto-switch to plan logic moved here for visibility
                                        if (data?.status === 'planning' && activeTab !== 'plan') {
                                            console.log('🔄 Auto-switching to plan tab - planning started');
                                            setActiveTab('plan');
                                        }
                                        // Define resolvedCanvasType before return
                                        const resolvedCanvasType = viewedCanvasType || canvasData?.type || 'spreadsheet';
                                        console.log('🎯 RESOLVED_CANVAS_TYPE_PASSED_TO_RENDERER:', resolvedCanvasType);

                                        const effectiveData = typeof displayCanvasContent === 'object' ? displayCanvasContent : (conversationState as any).canvas_data;

                                        return (
                                            <CanvasRenderer
                                                key={effectiveData ? JSON.stringify(effectiveData).substring(0, 100) : 'empty'}
                                                canvasType={resolvedCanvasType as any}
                                                canvasContent={typeof displayCanvasContent === 'string' ? displayCanvasContent : undefined}
                                                canvasData={effectiveData}
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
                                                        const { continueConversation } = useConversationStore.getState().actions;
                                                        await continueConversation(`Undo the last edit to ${filePath}`, []);
                                                    }
                                                }}
                                                onRedo={async () => {
                                                    // User clicked redo - send redo command
                                                    const canvasData = typeof displayCanvasContent === 'object' ? displayCanvasContent : (conversationState as any).canvas_data;
                                                    const filePath = canvasData?.file_path;
                                                    if (filePath) {
                                                        console.log('User requested redo for document:', filePath);
                                                        const { continueConversation } = useConversationStore.getState().actions;
                                                        await continueConversation(`Redo the last undone edit to ${filePath}`, []);
                                                    }
                                                }}
                                                onShowHistory={async () => {
                                                    // User clicked history - show version history
                                                    const canvasData = typeof displayCanvasContent === 'object' ? displayCanvasContent : (conversationState as any).canvas_data;
                                                    const filePath = canvasData?.file_path;
                                                    if (filePath) {
                                                        console.log('User requested version history for document:', filePath);
                                                        const { continueConversation } = useConversationStore.getState().actions;
                                                        await continueConversation(`Show version history for ${filePath}`, []);
                                                    }
                                                }}
                                            />
                                        );
                                    })()}
                                </div>
                            </div>
                        ) : (
                            <div className="text-center text-text-tertiary py-8">
                                <p className="text-orbimesh-section-header font-semibold">No Canvas Content</p>
                                <p className="text-orbimesh-section-subtitle mt-2">Interactive content from responses will appear here.</p>
                                <p className="text-orbimesh-file-meta mt-1 text-text-disabled">Browser live view is shown in the chat area.</p>
                            </div>
                        )}
                    </TabsContent>
                </Tabs>
            </aside>
        )
    });

export default OrchestrationDetailsSidebar;
