"use client"

import { useConversationStore } from "@/lib/conversation-store"
import { motion, AnimatePresence } from "framer-motion"
import { Loader2, Search, Brain, CheckCircle, AlertCircle, Users, FileText, BarChart3, Zap, Settings, Shield, Database, Link2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface OrchestrationProgressProps {
  className?: string;
}

const stageConfig = {
  initializing: {
    icon: Loader2,
    color: "text-blue-500",
    bgColor: "bg-blue-50 dark:bg-blue-950/30",
    borderColor: "border-blue-200 dark:border-blue-800",
    message: "Starting agent orchestration...",
    spin: true
  },
  analyzing: {
    icon: Brain,
    color: "text-purple-500",
    bgColor: "bg-purple-50 dark:bg-purple-950/30",
    borderColor: "border-purple-200 dark:border-purple-800",
    message: "Analyzing your request...",
    spin: false
  },
  parsing: {
    icon: FileText,
    color: "text-purple-500",
    bgColor: "bg-purple-50 dark:bg-purple-950/30",
    borderColor: "border-purple-200 dark:border-purple-800",
    message: "Breaking down your request...",
    spin: false
  },
  searching: {
    icon: Search,
    color: "text-green-500",
    bgColor: "bg-green-50 dark:bg-green-950/30",
    borderColor: "border-green-200 dark:border-green-800",
    message: "Searching for agents (REST & MCP)...",
    spin: false
  },
  ranking: {
    icon: Users,
    color: "text-orange-500",
    bgColor: "bg-orange-50 dark:bg-orange-950/30",
    borderColor: "border-orange-200 dark:border-orange-800",
    message: "Ranking agents for your tasks...",
    spin: false
  },
  planning: {
    icon: Settings,
    color: "text-indigo-500",
    bgColor: "bg-indigo-50 dark:bg-indigo-950/30",
    borderColor: "border-indigo-200 dark:border-indigo-800",
    message: "Creating execution plan...",
    spin: false
  },
  validating: {
    icon: Shield,
    color: "text-teal-500",
    bgColor: "bg-teal-50 dark:bg-teal-950/30",
    borderColor: "border-teal-200 dark:border-teal-800",
    message: "Validating execution plan...",
    spin: false
  },
  approval_required: {
    icon: AlertCircle,
    color: "text-amber-500",
    bgColor: "bg-amber-50 dark:bg-amber-950/30",
    borderColor: "border-amber-200 dark:border-amber-800",
    message: "Waiting for plan approval...",
    spin: false
  },
  executing: {
    icon: Zap,
    color: "text-red-500",
    bgColor: "bg-red-50 dark:bg-red-950/30",
    borderColor: "border-red-200 dark:border-red-800",
    message: "Executing tasks with agents...",
    spin: true
  },
  evaluating: {
    icon: BarChart3,
    color: "text-pink-500",
    bgColor: "bg-pink-50 dark:bg-pink-950/30",
    borderColor: "border-pink-200 dark:border-pink-800",
    message: "Evaluating agent responses...",
    spin: false
  },
  aggregating: {
    icon: BarChart3,
    color: "text-cyan-500",
    bgColor: "bg-cyan-50 dark:bg-cyan-950/30",
    borderColor: "border-cyan-200 dark:border-cyan-800",
    message: "Generating final response...",
    spin: false
  },
  finalizing: {
    icon: CheckCircle,
    color: "text-blue-500",
    bgColor: "bg-blue-50 dark:bg-blue-950/30",
    borderColor: "border-blue-200 dark:border-blue-800",
    message: "Finalizing...",
    spin: false
  },
  saving: {
    icon: Database,
    color: "text-slate-500",
    bgColor: "bg-slate-50 dark:bg-slate-950/30",
    borderColor: "border-slate-200 dark:border-slate-800",
    message: "Saving conversation...",
    spin: false
  },
  loading: {
    icon: Loader2,
    color: "text-gray-500",
    bgColor: "bg-gray-50 dark:bg-gray-950/30",
    borderColor: "border-gray-200 dark:border-gray-800",
    message: "Loading conversation history...",
    spin: true
  },
  waiting: {
    icon: AlertCircle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-50 dark:bg-yellow-950/30",
    borderColor: "border-yellow-200 dark:border-yellow-800",
    message: "Waiting for your input...",
    spin: false
  },
  waiting_for_user: {
    icon: AlertCircle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-50 dark:bg-yellow-950/30",
    borderColor: "border-yellow-200 dark:border-yellow-800",
    message: "Waiting for your response...",
    spin: false
  },
  connecting_mcp: {
    icon: Link2,
    color: "text-violet-500",
    bgColor: "bg-violet-50 dark:bg-violet-950/30",
    borderColor: "border-violet-200 dark:border-violet-800",
    message: "Connecting to MCP servers...",
    spin: true
  },
  completed: {
    icon: CheckCircle,
    color: "text-green-500",
    bgColor: "bg-green-50 dark:bg-green-950/30",
    borderColor: "border-green-200 dark:border-green-800",
    message: "Orchestration completed successfully!",
    spin: false
  },
  error: {
    icon: AlertCircle,
    color: "text-red-500",
    bgColor: "bg-red-50 dark:bg-red-950/30",
    borderColor: "border-red-200 dark:border-red-800",
    message: "An error occurred during orchestration",
    spin: false
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
        key={currentStage}
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -20, scale: 0.95 }}
        transition={{ 
          duration: 0.4,
          ease: [0.4, 0, 0.2, 1]
        }}
        className={cn(
          "w-full p-4 rounded-lg border shadow-sm backdrop-blur-sm transition-all duration-300",
          config.bgColor,
          config.borderColor,
          className
        )}
      >
        <div className="flex items-center space-x-3">
          <motion.div
            animate={{ 
              rotate: config.spin ? 360 : 0,
              scale: config.spin ? [1, 1.15, 1] : [1, 1.05, 1]
            }}
            transition={{ 
              rotate: { duration: 2, repeat: Infinity, ease: "linear" },
              scale: { duration: 2, repeat: Infinity, ease: "easeInOut" }
            }}
            className="relative"
          >
            <Icon className={cn("w-5 h-5", config.color)} />
            {config.spin && (
              <motion.div
                className={cn("absolute inset-0 rounded-full", config.color.replace('text-', 'bg-'), "opacity-20")}
                animate={{ scale: [1, 1.5, 1], opacity: [0.2, 0, 0.2] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
              />
            )}
          </motion.div>
          
          <div className="flex-1 min-w-0">
            <motion.p
              key={stageMessage}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className={cn("text-sm font-medium truncate", config.color)}
            >
              {stageMessage}
            </motion.p>
            
            {progress > 0 && currentStage !== 'completed' && (
              <motion.div 
                className="mt-2"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                transition={{ duration: 0.3 }}
              >
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                  <motion.div
                    className={cn("h-2 rounded-full relative", config.color.replace('text-', 'bg-'))}
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
                  >
                    <motion.div
                      className="absolute inset-0 bg-white/30"
                      animate={{ x: ['-100%', '100%'] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                      style={{ width: '50%' }}
                    />
                  </motion.div>
                </div>
                <motion.p 
                  className="text-xs text-gray-500 dark:text-gray-400 mt-1"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  {Math.round(progress)}% complete
                </motion.p>
              </motion.div>
            )}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
