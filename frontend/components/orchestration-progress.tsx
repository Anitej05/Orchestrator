"use client"

import { useConversationStore } from "@/lib/conversation-store"
import { motion, AnimatePresence } from "framer-motion"
import { Loader2, Search, Brain, CheckCircle, AlertCircle, Users, FileText, BarChart3, Zap, Settings, Play, Shield } from "lucide-react"
import { cn } from "@/lib/utils"
import { useEffect, useState } from "react";

interface OrchestrationProgressProps {
  className?: string;
}

const stageConfig = {
  initializing: {
    icon: Loader2,
    color: "text-blue-500",
    bgColor: "bg-blue-50",
    borderColor: "border-blue-200",
    message: "Starting agent orchestration..."
  },
  parsing: {
    icon: Brain,
    color: "text-purple-500",
    bgColor: "bg-purple-50",
    borderColor: "border-purple-200",
    message: "Analyzing your request..."
  },
  searching: {
    icon: Search,
    color: "text-green-500",
    bgColor: "bg-green-50",
    borderColor: "border-green-200",
    message: "Searching agent directory..."
  },
  ranking: {
    icon: Users,
    color: "text-orange-500",
    bgColor: "bg-orange-50",
    borderColor: "border-orange-200",
    message: "Ranking agents for your tasks..."
  },
  planning: {
    icon: FileText,
    color: "text-indigo-500",
    bgColor: "bg-indigo-50",
    borderColor: "border-indigo-200",
    message: "Creating execution plan..."
  },
  validating: {
    icon: CheckCircle,
    color: "text-teal-500",
    bgColor: "bg-teal-50",
    borderColor: "border-teal-200",
    message: "Validating execution plan..."
  },
  executing: {
    icon: Play,
    color: "text-red-500",
    bgColor: "bg-red-50",
    borderColor: "border-red-200",
    message: "Executing tasks..."
  },
  aggregating: {
    icon: BarChart3,
    color: "text-cyan-500",
    bgColor: "bg-cyan-50",
    borderColor: "border-cyan-200",
    message: "Aggregating results..."
  },
  waiting_for_user: {
    icon: AlertCircle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-50",
    borderColor: "border-yellow-200",
    message: "Waiting for your response..."
  },
  completed: {
    icon: CheckCircle,
    color: "text-green-500",
    bgColor: "bg-green-50",
    borderColor: "border-green-200",
    message: "Orchestration completed successfully!"
  },
  error: {
    icon: AlertCircle,
    color: "text-red-500",
    bgColor: "bg-red-50",
    borderColor: "border-red-200",
    message: "An error occurred during orchestration"
  }
};

export function OrchestrationProgress({ className }: OrchestrationProgressProps) {
  const { status, metadata } = useConversationStore();
  const currentStage = metadata?.currentStage || 'initializing';
  const stageMessage = metadata?.stageMessage || 'Processing...';
  const progress = metadata?.progress || 0;

  // Don't show progress bar if not processing or if completed with error
  if (status === 'idle' || (status === 'error' && currentStage === 'error')) {
    return null;
  }

  const config = stageConfig[currentStage as keyof typeof stageConfig] || stageConfig.initializing;
  const Icon = config.icon;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
        className={cn(
          "w-full p-4 rounded-lg border shadow-sm",
          config.bgColor,
          config.borderColor,
          className
        )}
      >
        <div className="flex items-center space-x-3">
          <motion.div
            animate={{ rotate: currentStage === 'executing' || currentStage === 'initializing' ? 360 : 0 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          >
            <Icon className={cn("w-5 h-5", config.color)} />
          </motion.div>
          
          <div className="flex-1">
            <motion.p
              key={stageMessage}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
              className={cn("text-sm font-medium", config.color)}
            >
              {stageMessage}
            </motion.p>
            
            {progress > 0 && currentStage !== 'completed' && (
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <motion.div
                    className={cn("h-2 rounded-full", config.color.replace('text-', 'bg-'))}
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">{progress}% complete</p>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
