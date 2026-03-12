// components/orchestration-details-sidebar.tsx
"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { FileIcon, FileText, Image as ImageIcon } from "lucide-react"
import ActionHistoryTimeline from "@/components/action-history-timeline"
import SaveWorkflowButton from "@/components/save-workflow-button"
import { OrchestrationFlow } from "@/components/orchestration-flow"
import { useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { useConversationStore } from "@/lib/conversation-store"
import { CanvasRenderer } from '@/components/canvas-renderer'
import type { Message } from "@/lib/types"
import { cn } from "@/lib/utils"
import { API_BASE_URL } from "@/lib/config"
import DocumentViewer from "@/components/document-viewer"
import { dismissCanvas } from "@/lib/canvas-api"
import { X } from "lucide-react"


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
}

interface Plan {
    // Modified to support parallel execution batches (array of arrays)
    pendingTasks: { task: string; description: string; agent: string; task_id?: string; status?: string; short_description?: string; icon_name?: string; agent_image_url?: string; }[][] | { task: string; description: string; agent: string; task_id?: string; status?: string; short_description?: string; icon_name?: string; agent_image_url?: string; }[];
    completedTasks: { task: string; result: string; }[];
    todoList?: any[]; // Support for todo_list from new orchestrator system
}

import { forwardRef, useImperativeHandle } from 'react';

export interface OrchestrationDetailsSidebarRef {
    refreshPlan: () => void;
    viewCanvas: (canvasContent: string, canvasType: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json') => void;
}

const OrchestrationDetailsSidebar = forwardRef<OrchestrationDetailsSidebarRef, OrchestrationDetailsSidebarProps>(
    ({ executionResults: _executionResults, threadId, className, onThreadIdUpdate: _onThreadIdUpdate }, ref) => {
        const [plan, setPlan] = useState<Plan>({ pendingTasks: [], completedTasks: [], todoList: undefined });
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
        const todoList = conversationState.todo_list || [];
        const actionHistory = (conversationState as any).action_history || [];
        const messages = conversationState.messages || [];
        const uploadedFiles = conversationState.uploaded_files || [];
        const planData = conversationState.plan || [];
        const hasCanvas = conversationState.has_canvas;
        const canvasContent = conversationState.canvas_content;
        const canvasData = (conversationState as any).canvas_data;
        const canvasType = conversationState.canvas_type;
        const browserView = (conversationState as any).browser_view;

        // Determine which canvas to display - viewed canvas takes precedence
        // Browser view is now shown in chat interface, not in canvas
        // Support both canvas_content (string) and canvas_data (structured object)
        const displayCanvasContent = viewedCanvasContent || canvasContent || canvasData;

        // Process plan data from conversation store
        useEffect(() => {
            // ONLY use planData for the workflow structure
            // Real-time status comes from task_statuses, NOT from completed_tasks
            // This prevents duplicates in the task cards
            const pendingTasks: Plan['pendingTasks'] = [];

            // Try todo_list first (new system) - pass directly to task cards
            if (todoList && todoList.length > 0) {
                // For todo_list, pass it directly and keep legacy pending tasks empty
                setPlan({
                    pendingTasks: [],
                    completedTasks: [],
                    todoList: todoList // Pass through directly
                });
                return;
            }
            // Fallback to process pending tasks from planData (the original plan structure for old system)
            // We want to PRESERVE the batch structure (list of lists) for valid parallel visualization
            else if (planData && planData.length > 0) {
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
        }, [planData, taskAgentPairs, todoList]);

        // Auto-switch to Plan tab when tasks arrive
        useEffect(() => {
            const currentStage = conversationState.metadata?.currentStage;
            const brainModeActive = todoList.length > 0;  // Brain emits todo_list → always switch
            const oldPipelineActive = currentStage === 'validating' || currentStage === 'executing' || currentStage === 'planning';

            if (brainModeActive || (oldPipelineActive && plan.pendingTasks.flat().length > 0)) {
                console.log('Auto-switching to plan tab - tasks visible');
                setActiveTab('plan');
            }
        }, [conversationState.metadata?.currentStage, plan.pendingTasks.length, todoList.length]);

        // Toast when execution completes
        const prevStatusRef = useRef<string | undefined>(undefined);
        useEffect(() => {
            const prev = prevStatusRef.current;
            const curr = conversationState.status;
            if (prev && prev !== 'completed' && curr === 'completed') {
                const taskCount = todoList.length || plan.pendingTasks.flat().length;
                toast.success(
                    taskCount > 0
                        ? `All ${taskCount} task${taskCount !== 1 ? 's' : ''} completed`
                        : 'Workflow completed',
                    { description: 'The orchestrator has finished all tasks.', duration: 5000 }
                );
            }
            prevStatusRef.current = curr;
        }, [conversationState.status]);

        // Helper function to check if canvas type is a document/file viewer type
        const isFileViewerType = (canvasType: string | undefined): boolean => {
            if (!canvasType) return false;
            
            // File types that should show in attachments/document viewer instead of canvas
            const fileViewerTypes = ['document', 'spreadsheet', 'pdf', 'image', 'csv', 'xlsx', 'docx', 'file'];
            return fileViewerTypes.includes(canvasType.toLowerCase());
        };

        // Auto-switch to canvas tab when NEW canvas content is created
        useEffect(() => {
            // Only switch if:
            // 1. We have canvas content OR canvas data
            // 2. The canvas content/data is different from what we've seen before (NEW content)
            // 3. Not currently executing (don't override plan view during execution)
            // 4. Canvas type is NOT a document/file viewer type (those should stay in attachments)
            const currentStage = conversationState.metadata?.currentStage;
            const isExecuting = currentStage === 'executing' || currentStage === 'validating';
            const canvasData = (conversationState as any).canvas_data;

            // Check if we have either canvas content or canvas data
            const hasCanvasData = canvasContent || canvasData;

            // Create a unique identifier for the current canvas state
            // This handles both canvas_content (string) and canvas_data (object) changes
            const currentCanvasIdentifier = canvasContent || (canvasData ? JSON.stringify(canvasData) : undefined);

            // Check if this is a file type that should be shown in document viewer/attachments
            const shouldShowAsFile = isFileViewerType(conversationState.canvas_type);

            if (hasCanvas && hasCanvasData && currentCanvasIdentifier !== lastCanvasContent && !isExecuting && !shouldShowAsFile) {
                console.log('Auto-switching to canvas tab due to NEW canvas content/data', {
                    hasCanvas,
                    hasCanvasContent: !!canvasContent,
                    hasCanvasData: !!canvasData,
                    canvasType: conversationState.canvas_type,
                    shouldShowAsFile
                });
                setActiveTab('canvas');
                setLastCanvasContent(currentCanvasIdentifier);
                // Clear any viewed canvas content to show the latest
                setViewedCanvasContent(undefined);
                setViewedCanvasType(undefined);
            } else if (hasCanvas && hasCanvasData && currentCanvasIdentifier !== lastCanvasContent && !isExecuting && shouldShowAsFile) {
                // For file types, update the identifier but don't switch tab
                console.log('Skipping canvas auto-switch for file type', {
                    canvasType: conversationState.canvas_type,
                    currentTab: activeTab
                });
                setLastCanvasContent(currentCanvasIdentifier);
            }
        }, [hasCanvas, canvasContent, (conversationState as any).canvas_data, lastCanvasContent, conversationState.metadata?.currentStage, activeTab]);

        // Method to view specific canvas content from a message
        const viewCanvas = (canvasContent: string, canvasType: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json') => {
            setViewedCanvasContent(canvasContent);
            setViewedCanvasType(canvasType);
            setActiveTab('canvas');
        };

        // Expose methods via ref
        useImperativeHandle(ref, () => ({
            refreshPlan: async () => {
                // no-op: TaskPlanViewer reads live from store
            },
            viewCanvas
        }));


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


        return (
            <aside className={cn("border-l border-border-color bg-bg-card text-text-primary p-4 flex flex-col h-full overflow-hidden", className)}>
                <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col overflow-hidden min-w-0">
                    <TabsList className="grid w-full grid-cols-4 bg-bg-subtle/80 backdrop-blur-xl border border-border-color/50 shadow-orbimesh-panel">
                        <TabsTrigger value="plan" className="relative text-xs data-[state=active]:text-brand-primary data-[state=active]:font-semibold">
                            Plan
                            {(conversationState.metadata?.currentStage === 'executing' ||
                                conversationState.metadata?.currentStage === 'validating') && (
                                    <span className="ml-1 px-1.5 py-0.5 text-[10px] font-semibold bg-status-active text-white rounded-full animate-pulse">
                                        Running
                                    </span>
                                )}
                        </TabsTrigger>
                        <TabsTrigger value="history" className="text-xs data-[state=active]:text-brand-primary data-[state=active]:font-semibold">History</TabsTrigger>
                        <TabsTrigger value="attachments" className="text-xs data-[state=active]:text-brand-primary data-[state=active]:font-semibold">Files</TabsTrigger>
                        <TabsTrigger value="canvas" className="text-xs data-[state=active]:text-brand-primary data-[state=active]:font-semibold">Canvas</TabsTrigger>
                    </TabsList>

                    <TabsContent value="plan" className="flex-1 flex flex-col overflow-hidden">
                        {/* TaskPlanViewer is fully self-contained — reads from Zustand store */}
                        <div className="flex-1 overflow-hidden min-h-0">
                            <OrchestrationFlow />
                        </div>

                        {/* Save workflow button — shown after completion */}
                        {conversationState.status === 'completed' && (
                            <div className="flex-shrink-0 px-4 py-2 border-t border-border-color/40">
                                <SaveWorkflowButton
                                    threadId={conversationState.thread_id || threadId || ''}
                                    disabled={!conversationState.thread_id && !threadId}
                                />
                            </div>
                        )}
                    </TabsContent>

                    {/* Action History Tab - Shows detailed execution log */}
                    <TabsContent value="history" className="flex-1 overflow-y-auto mt-4">
                        <ActionHistoryTimeline history={actionHistory} />
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
                            <div className="flex flex-col items-center justify-center h-full py-12 px-4 text-center">
                                <div className="w-16 h-16 rounded-2xl bg-brand-primary-light border-2 border-dashed border-brand-primary/30 flex items-center justify-center mb-4">
                                    <FileIcon className="w-7 h-7 text-brand-primary/60" />
                                </div>
                                <p className="text-sm font-semibold text-text-secondary">No Attachments</p>
                                <p className="text-xs text-text-tertiary mt-1.5 max-w-[160px] leading-relaxed">Files you upload will appear here</p>
                            </div>
                        )}
                    </TabsContent>
                    <TabsContent value="canvas" className="flex-1 overflow-hidden mt-4 flex flex-col">
                        {/* Canvas now only shows non-browser content (HTML/Markdown from LLM responses) */}
                        {/* Browser live stream is shown in the chat interface instead */}
                        {(hasCanvas || viewedCanvasContent) && displayCanvasContent && !browserView ? (
                            <div className="h-full flex flex-col">
                                {/* Canvas Header with Dismiss Button */}
                                {!viewedCanvasContent && hasCanvas && (
                                    <div className="bg-bg-card border-b border-border-color px-4 py-2 shadow-sm flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <span className="text-orbimesh-section-header font-semibold text-text-primary">
                                                {conversationState.canvas_title || 'Canvas View'}
                                            </span>
                                            {canvasType && (
                                                <span className="text-xs px-2 py-0.5 bg-bg-subtle text-text-secondary rounded-full border border-border-color">
                                                    {canvasType}
                                                </span>
                                            )}
                                        </div>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={async () => {
                                                if (threadId) {
                                                    try {
                                                        // Use REST API to dismiss canvas
                                                        await dismissCanvas(threadId, 'main');
                                                        // Clear local state
                                                        useConversationStore.setState({
                                                            has_canvas: false,
                                                            canvas_content: undefined,
                                                            canvas_data: undefined,
                                                            canvas_type: undefined
                                                        });
                                                    } catch (error) {
                                                        console.error('Failed to dismiss canvas:', error);
                                                    }
                                                }
                                            }}
                                            className="gap-1 text-text-tertiary hover:text-text-primary"
                                        >
                                            <X className="w-4 h-4" />
                                            Dismiss
                                        </Button>
                                    </div>
                                )}
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
                                        const resolvedCanvasType = viewedCanvasType || canvasType || canvasData?.type || 'spreadsheet';
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
                            <div className="flex flex-col items-center justify-center h-full py-12 px-4 text-center">
                                <div className="w-16 h-16 rounded-2xl bg-brand-primary-light border-2 border-dashed border-brand-primary/30 flex items-center justify-center mb-4">
                                    <FileText className="w-7 h-7 text-brand-primary/60" />
                                </div>
                                <p className="text-sm font-semibold text-text-secondary">No Canvas Content</p>
                                <p className="text-xs text-text-tertiary mt-1.5 max-w-[160px] leading-relaxed">Interactive output from agent responses will appear here</p>
                            </div>
                        )}
                    </TabsContent>
                </Tabs>
            </aside>
        )
    });

export default OrchestrationDetailsSidebar;
