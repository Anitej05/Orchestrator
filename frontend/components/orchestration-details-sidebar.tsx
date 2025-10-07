// components/orchestration-details-sidebar.tsx
"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DollarSign, Clock, FileIcon } from "lucide-react"
import CollapsibleSection from "@/components/CollapsibleSection"
import PlanGraph from "@/components/PlanGraph"
import { useEffect, useState } from "react"
import { fetchPlanFile } from "@/lib/api-client"
import { InteractiveStarRating, StarRating } from "@/components/ui/star-rating"
import type { Agent, Message } from "@/lib/types"
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
    taskAgentPairs: { primary: Agent }[];
    messages: Message[];
    className?: string;
    onThreadIdUpdate?: (threadId: string) => void;
}

interface Plan {
    pendingTasks: { task: string; description: string; agent: string; }[];
    completedTasks: { task: string; result: string; }[];
}

export default function OrchestrationDetailsSidebar({ executionResults, threadId, taskAgentPairs, messages, className }: OrchestrationDetailsSidebarProps) {
    const [plan, setPlan] = useState<Plan>({ pendingTasks: [], completedTasks: [] });
    const [isLoadingPlan, setIsLoadingPlan] = useState(false);
    const [agents, setAgents] = useState<Agent[]>([]);

    useEffect(() => {
        const uniqueAgentsFromPairs = Array.from(new Set(taskAgentPairs.map(pair => pair.primary.id)))
            .map(id => taskAgentPairs.find(pair => pair.primary.id === id)!.primary);
        setAgents(uniqueAgentsFromPairs);
    }, [taskAgentPairs]);

    const handleRatingUpdate = (agentId: string, newRating: number) => {
        setAgents(prevAgents =>
            prevAgents.map(agent =>
                agent.id === agentId ? { ...agent, rating: newRating, rating_count: (agent.rating_count ?? 0) + 1 } : agent
            )
        );
    };


    useEffect(() => {
        const loadPlan = async () => {
            if (threadId) {
                setIsLoadingPlan(true);
                try {
                    const content = await fetchPlanFile(threadId);
                    if (content) {
                        const pendingTasks: Plan['pendingTasks'] = [];
                        const completedTasks: Plan['completedTasks'] = [];

                        const pendingSectionMatch = content.match(/## Pending Tasks([\s\S]*?)## Completed Tasks/);
                        const pendingSection = pendingSectionMatch ? pendingSectionMatch[1] : '';

                        const completedSectionMatch = content.match(/## Completed Tasks([\s\S]*)/);
                        const completedSection = completedSectionMatch ? completedSectionMatch[1] : '';

                        if (pendingSection) {
                            const taskRegex = /- \*\*Task\*\*: \`(.+?)\`\s*-\s+\*\*Description\*\*:\s(.*?)\s*-\s+\*\*Agent\*\*:\s(.*?)(?=\n- \*\*Task\*\*|\n##|$)/gs;
                            let match;
                            while ((match = taskRegex.exec(pendingSection)) !== null) {
                                pendingTasks.push({
                                    task: match[1].trim(),
                                    description: match[2].trim(),
                                    agent: match[3].trim()
                                });
                            }
                        }

                        if (completedSection) {
                            const taskRegex = /- \*\*Task\*\*: \`(.+?)\`\s*-\s+\*\*Result\*\*:\s*```json\s*([\s\S]*?)\s*```/g;
                            let match;
                            while ((match = taskRegex.exec(completedSection)) !== null) {
                                let formattedResult = match[2].trim();
                                try {
                                    const parsedJson = JSON.parse(formattedResult);
                                    formattedResult = JSON.stringify(parsedJson, null, 2);
                                } catch (e) {
                                    // Keep as is if not valid JSON
                                }
                                completedTasks.push({ task: match[1].trim(), result: formattedResult });
                            }
                        }

                        setPlan({ pendingTasks, completedTasks });
                    } else {
                        setPlan({ pendingTasks: [], completedTasks: [] });
                    }
                } catch (error) {
                    console.error("Failed to load plan:", error);
                    setPlan({ pendingTasks: [], completedTasks: [] });
                } finally {
                    setIsLoadingPlan(false);
                }
            } else {
                setPlan({ pendingTasks: [], completedTasks: [] });
            }
        };

        loadPlan();

    }, [threadId]);

    const totalCost = executionResults.reduce((sum, result) => sum + result.cost, 0)
    const totalTime = executionResults.reduce((sum, result) => sum + result.executionTime, 0)
    const allTasks = [...plan.pendingTasks, ...plan.completedTasks];
    const allAttachments = messages.flatMap(m => m.attachments || []);

    const hasResults = executionResults.length > 0 || allTasks.length > 0

    return (
        <aside className={cn("border-l bg-gray-50/50 p-4 flex flex-col h-full", className)}>
            <Tabs defaultValue="metadata" className="h-full flex flex-col">
                <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="metadata">Metadata</TabsTrigger>
                    <TabsTrigger value="plan">Plan</TabsTrigger>
                    <TabsTrigger value="attachments">Attachments</TabsTrigger>
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
                                                    <TableRow key={index}>
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
                    <PlanGraph planData={plan} />
                </TabsContent>
                <TabsContent value="attachments" className="flex-1 overflow-y-auto mt-4">
                    {allAttachments.length > 0 ? (
                        <ul className="space-y-2">
                        {allAttachments.map((att, index) => (
                            <li key={index} className="flex items-center gap-2 text-sm p-2 rounded-md bg-white border">
                            <FileIcon className="w-4 h-4 text-gray-500" />
                            <span className="truncate" title={att.name}>{att.name}</span>
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
            </Tabs>
        </aside>
    )
}
