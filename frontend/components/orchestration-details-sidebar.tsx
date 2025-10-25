// components/orchestration-details-sidebar.tsx
"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DollarSign, Clock, FileIcon, FileText, Image as ImageIcon } from "lucide-react"
import CollapsibleSection from "@/components/CollapsibleSection"
import PlanGraph from "@/components/PlanGraph"
import { useEffect, useState } from "react"
import { InteractiveStarRating, StarRating } from "@/components/ui/star-rating"
import { useConversationStore } from "@/lib/conversation-store"
import Markdown from '@/components/ui/markdown'
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
 ({ executionResults, threadId, className, onThreadIdUpdate }, ref) => {
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
    const canvasType = conversationState.canvas_type;
    
    // Determine which canvas to display - viewed canvas takes precedence
    const displayCanvasContent = viewedCanvasContent || canvasContent;
    const displayCanvasType = viewedCanvasType || canvasType;

    // Process plan data from conversation store
    useEffect(() => {
        // Always update the plan when planData or metadata changes
        const pendingTasks: Plan['pendingTasks'] = [];
        const completedTasks: Plan['completedTasks'] = [];
        
        // Process pending tasks from planData
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
        
        // Process completed tasks from conversationState.metadata
        const completedTasksData = conversationState.metadata?.completed_tasks || [];
        if (completedTasksData && completedTasksData.length > 0) {
            completedTasksData.forEach((task: any) => {
                if (task && typeof task === 'object') {
                    completedTasks.push({
                        task: task.task_name || 'Unknown Task',
                        result: typeof task.result === 'string' ? task.result : JSON.stringify(task.result, null, 2)
                    });
                }
            });
        }
        
        setPlan({ 
            pendingTasks: pendingTasks, 
            completedTasks: completedTasks 
        });
        setIsLoadingPlan(false);
    }, [planData, conversationState.metadata]);

    // Auto-switch to canvas tab when NEW canvas content is created
    useEffect(() => {
        // Only switch if:
        // 1. We have canvas content
        // 2. The canvas content is different from what we've seen before (NEW content)
        if (hasCanvas && canvasContent && canvasContent !== lastCanvasContent) {
            console.log('Auto-switching to canvas tab due to NEW canvas content');
            setActiveTab('canvas');
            setLastCanvasContent(canvasContent);
            // Clear any viewed canvas content to show the latest
            setViewedCanvasContent(undefined);
            setViewedCanvasType(undefined);
        }
    }, [hasCanvas, canvasContent, lastCanvasContent]);

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
        <aside className={cn("border-l border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 p-4 flex flex-col h-full", className)}>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
                <TabsList className="grid w-full grid-cols-4">
                    <TabsTrigger value="metadata">Metadata</TabsTrigger>
                    <TabsTrigger value="plan">Plan</TabsTrigger>
                    <TabsTrigger value="attachments">Attachments</TabsTrigger>
                    <TabsTrigger value="canvas">Canvas</TabsTrigger>
                </TabsList>
                <TabsContent value="metadata" className="flex-1 overflow-y-auto mt-4 space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>Execution Summary</CardTitle>
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
                <TabsContent value="plan" className="flex-1 flex items-center justify-center">
                    <PlanGraph key={JSON.stringify(plan)} planData={plan} />
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
                    {(hasCanvas || viewedCanvasContent) && displayCanvasContent ? (
                        <div className="h-full flex flex-col">
                            {viewedCanvasContent && (
                                <div className="bg-blue-50 border-b border-blue-200 px-4 py-2 text-sm text-blue-800">
                                    Viewing canvas from a previous message
                                    <button 
                                        onClick={() => {
                                            setViewedCanvasContent(undefined);
                                            setViewedCanvasType(undefined);
                                        }}
                                        className="ml-2 text-blue-600 hover:text-blue-800 underline"
                                    >
                                        Return to latest
                                    </button>
                                </div>
                            )}
                            <div className="flex-1 overflow-auto">
                                {displayCanvasType === 'html' ? (
                                    <iframe
                                        srcDoc={displayCanvasContent}
                                        className="w-full h-full min-h-[300px] border-0"
                                        title="Canvas HTML Content"
                                        sandbox="allow-scripts allow-same-origin"
                                    />
                                ) : (
                                    <div className="prose prose-sm max-w-none p-4">
                                        <Markdown content={displayCanvasContent} />
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="text-center text-gray-500 py-8">
                            <p className="font-semibold">No Canvas Content</p>
                            <p className="text-sm mt-2">Canvas content will appear here when available.</p>
                        </div>
                    )}
                </TabsContent>
            </Tabs>
        </aside>
    )
});

export default OrchestrationDetailsSidebar;
