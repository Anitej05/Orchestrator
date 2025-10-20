// components/orchestration-details-sidebar.tsx
"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DollarSign, Clock, FileIcon } from "lucide-react"
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
}

const OrchestrationDetailsSidebar = forwardRef<OrchestrationDetailsSidebarRef, OrchestrationDetailsSidebarProps>(
 ({ executionResults, threadId, className, onThreadIdUpdate }, ref) => {
    const [plan, setPlan] = useState<Plan>({ pendingTasks: [], completedTasks: [] });
    const [isLoadingPlan, setIsLoadingPlan] = useState(false);
    const [agents, setAgents] = useState<Agent[]>([]);
    
    // Get conversation state from Zustand store
    const conversationState = useConversationStore();
    const taskAgentPairs = conversationState.task_agent_pairs || [];
    const messages = conversationState.messages || [];
    const uploadedFiles = conversationState.uploaded_files || [];
    const planData = conversationState.plan || [];

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


    const totalCost = executionResults.reduce((sum, result) => sum + result.cost, 0)
    const totalTime = executionResults.reduce((sum, result) => sum + result.executionTime, 0)
    const allTasks = [...plan.pendingTasks, ...plan.completedTasks];
    // Collect attachments from messages and uploaded files
    const messageAttachments = messages.flatMap((m: Message) => m.attachments || []);
    const fileAttachments = (uploadedFiles || []).map((file: any) => ({
      name: file.file_name || file.name || 'Unknown File',
      type: file.file_type || file.type || 'unknown',
      content: file.content || ''
    }));
    
    // Combine attachments and deduplicate based on name, type, and content
    // This prevents duplicates while preserving unique files with the same name
    const allAttachments = [...messageAttachments, ...fileAttachments].filter(
      (att, index, self) => 
        index === self.findIndex(a => 
          a.name === att.name && 
          a.type === att.type && 
          a.content === att.content
        )
    );

    const hasResults = executionResults.length > 0 || allTasks.length > 0

    return (
        <aside className={cn("border-l bg-gray-50/50 p-4 flex flex-col h-full", className)}>
            <Tabs defaultValue="metadata" className="h-full flex flex-col">
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
                        <ul className="space-y-2">
                        {allAttachments.map((att: any, index: number) => (
                            <li key={`${att.name}-${index}`} className="flex items-center gap-2 text-sm p-2 rounded-md bg-white border">
                            {att.type.startsWith('image/') ? (
                                <>
                                    <FileIcon className="w-4 h-4 text-gray-500" />
                                    <div className="flex flex-col">
                                        <span className="truncate" title={att.name}>{att.name}</span>
                                        {att.content && (
                                            <img src={att.content} alt={att.name} className="max-w-xs max-h-32 rounded mt-1" />
                                        )}
                                    </div>
                                </>
                            ) : (
                                <>
                                    <FileIcon className="w-4 h-4 text-gray-500" />
                                    <span className="truncate" title={att.name}>{att.name}</span>
                                </>
                            )}
                            </li>
                        ))}
                        </ul>
                    ) : (
                        <div className="text-center text-gray-500 py-8">
                            <p className="font-semibold">No Attachments</p>
                            <p className="text-sm mt-2">Files you upload will appear here.</p>
                        </div>
                    )}
                </TabsContent>
                <TabsContent value="canvas" className="flex-1 overflow-y-auto mt-4">
                    {conversationState.has_canvas && conversationState.canvas_content ? (
                        <div className="h-full flex flex-col">
                            <div className="flex-1 overflow-auto">
                                {conversationState.canvas_type === 'html' ? (
                                    <iframe
                                        srcDoc={conversationState.canvas_content}
                                        className="w-full h-full min-h-[300px] border-0"
                                        title="Canvas HTML Content"
                                        sandbox="allow-scripts allow-same-origin"
                                    />
                                ) : (
                                    <div className="prose prose-sm max-w-none p-4">
                                        <Markdown content={conversationState.canvas_content} />
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
