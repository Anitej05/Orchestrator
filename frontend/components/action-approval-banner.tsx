"use client"

import React, { useState } from 'react';
import { AlertCircle, Check, X, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useConversationStore } from '@/lib/conversation-store';
import { approveAction, rejectAction } from '@/lib/api/orchestrator';
import { useToast } from '@/hooks/use-toast';

export function ActionApprovalBanner() {
    const {
        pending_action_approval,
        pending_action,
        thread_id,
        actions: { _setConversationState }
    } = useConversationStore();

    const [isProcessing, setIsProcessing] = useState(false);
    const { toast } = useToast();

    if (!pending_action_approval || !pending_action) return null;

    const handleApprove = async () => {
        if (!thread_id) return;

        setIsProcessing(true);
        try {
            await approveAction(thread_id);

            // Optimistic update
            _setConversationState({
                pending_action_approval: false,
                pending_action: undefined,
                metadata: {
                    ...useConversationStore.getState().metadata,
                    currentStage: 'executing',
                    stageMessage: 'Action approved. Resuming...',
                }
            });

            toast({
                title: "Action Approved",
                description: "Orchestrator is resuming execution.",
                variant: "default",
            });
        } catch (error) {
            console.error("Failed to approve action:", error);
            toast({
                title: "Error",
                description: "Failed to approve action. Please try again.",
                variant: "destructive",
            });
        } finally {
            setIsProcessing(false);
        }
    };

    const handleReject = async () => {
        if (!thread_id) return;

        setIsProcessing(true);
        try {
            await rejectAction(thread_id, "User rejected via UI");

            // Optimistic update
            _setConversationState({
                pending_action_approval: false,
                pending_action: undefined,
                metadata: {
                    ...useConversationStore.getState().metadata,
                    currentStage: 'executing',
                    stageMessage: 'Action rejected. Resuming...',
                }
            });

            toast({
                title: "Action Rejected",
                description: "Orchestrator will skip this action.",
                variant: "default",
            });
        } catch (error) {
            console.error("Failed to reject action:", error);
            toast({
                title: "Error",
                description: "Failed to reject action. Please try again.",
                variant: "destructive",
            });
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className="fixed bottom-20 left-1/2 transform -translate-x-1/2 z-50 w-full max-w-2xl px-4 animate-in slide-in-from-bottom-2 duration-300">
            <Card className="p-4 border-l-4 border-l-amber-500 shadow-xl bg-white dark:bg-gray-900 ring-1 ring-black/5">
                <div className="flex items-start gap-4">
                    <div className="p-2 bg-amber-100 dark:bg-amber-900/30 rounded-full shrink-0">
                        <AlertCircle className="w-6 h-6 text-amber-600 dark:text-amber-500" />
                    </div>

                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                                Action Approval Required
                            </h3>
                            <span className="px-2 py-0.5 text-xs font-medium bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300 rounded-full">
                                Paused
                            </span>
                        </div>

                        <p className="text-sm text-gray-600 dark:text-gray-300 mb-3 leading-relaxed">
                            {pending_action.approval_reason || "Sensitive action requires your approval to proceed."}
                        </p>

                        <div className="rounded bg-gray-50 dark:bg-gray-800/50 p-2 text-xs font-mono text-gray-500 mb-4 overflow-hidden text-ellipsis whitespace-nowrap">
                            {pending_action.action_type.toUpperCase()}: {pending_action.resource_id}
                        </div>

                        <div className="flex gap-3">
                            <Button
                                onClick={handleApprove}
                                disabled={isProcessing}
                                className="bg-green-600 hover:bg-green-700 text-white gap-2 flex-1"
                            >
                                {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                                Approve & Execute
                            </Button>

                            <Button
                                onClick={handleReject}
                                variant="outline"
                                disabled={isProcessing}
                                className="border-red-200 hover:bg-red-50 hover:text-red-600 dark:border-red-900/30 dark:hover:bg-red-900/10 gap-2 flex-1"
                            >
                                <X className="w-4 h-4" />
                                Reject & Skip
                            </Button>
                        </div>
                    </div>
                </div>
            </Card>
        </div>
    );
}
