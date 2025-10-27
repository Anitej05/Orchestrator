"use client"

import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { DollarSign, Clock, CheckCircle, AlertCircle } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"

interface PlanTask {
  task_name: string;
  task_description: string;
  primary: {
    name: string;
    price_per_call_usd: number;
    rating: number;
  };
}

interface PlanApprovalModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApprove: () => void;
  onModify: () => void;
  onCancel: () => void;
  taskPlan: any[];
  estimatedCost: number;
  taskCount: number;
}

export function PlanApprovalModal({
  isOpen,
  onClose,
  onApprove,
  onModify,
  onCancel,
  taskPlan,
  estimatedCost,
  taskCount
}: PlanApprovalModalProps) {
  
  // Flatten the task plan (it's an array of batches)
  const allTasks: PlanTask[] = [];
  if (taskPlan && Array.isArray(taskPlan)) {
    taskPlan.forEach((batch: any) => {
      if (Array.isArray(batch)) {
        batch.forEach((task: any) => {
          allTasks.push(task);
        });
      }
    });
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-blue-500" />
            Review Execution Plan
          </DialogTitle>
          <DialogDescription>
            Please review the execution plan before proceeding. You can approve, modify, or cancel.
          </DialogDescription>
        </DialogHeader>

        {/* Cost Summary */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Total Tasks</p>
              <p className="text-2xl font-bold">{taskCount}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Estimated Cost</p>
              <p className="text-2xl font-bold">${estimatedCost.toFixed(4)}</p>
            </div>
          </div>
        </div>

        {/* Task List */}
        <ScrollArea className="flex-1 pr-4">
          <div className="space-y-3">
            {allTasks.map((task, index) => (
              <div
                key={index}
                className="p-4 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="outline" className="text-xs">
                        Task {index + 1}
                      </Badge>
                      <h4 className="font-semibold text-sm">{task.task_name}</h4>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {task.task_description}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-4">
                    <span className="flex items-center gap-1">
                      <span className="font-medium">Agent:</span> {task.primary?.name || 'Unknown'}
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="font-medium">Rating:</span> {task.primary?.rating?.toFixed(1) || 'N/A'} ⭐
                    </span>
                  </div>
                  <span className="flex items-center gap-1 font-medium text-blue-600 dark:text-blue-400">
                    <DollarSign className="w-3 h-3" />
                    {task.primary?.price_per_call_usd?.toFixed(4) || '0.0000'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        <DialogFooter className="flex gap-2 sm:gap-2">
          <Button
            variant="outline"
            onClick={onCancel}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            variant="secondary"
            onClick={onModify}
            className="flex-1"
          >
            Modify Plan
          </Button>
          <Button
            onClick={onApprove}
            className="flex-1 bg-green-600 hover:bg-green-700 text-white"
          >
            Approve & Execute
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
