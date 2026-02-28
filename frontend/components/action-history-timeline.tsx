// components/action-history-timeline.tsx
/**
 * ActionHistoryTimeline - Shows detailed execution log
 * 
 * Displays agent/tool execution with results - THIS IS WHERE YOU SEE WHAT ACTUALLY HAPPENED!
 * Uses the same icon system as PlanGraph for visual consistency.
 */

"use client"

import React from 'react';
import { CheckCircle, XCircle, Loader2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

// Reuse the same icon mapping from PlanGraph for consistency!
const AGENT_ICONS: Record<string, string> = {
    // Email & Communication
    'gmail': '📧', 'gmail_agent': '📧', 'mail': '📬', 'mail_agent': '📬',
    // Documents & Files
    'document': '📄', 'document_agent': '📄', 'spreadsheet': '📊', 'spreadsheet_agent': '📊',
    // Web & Browser
    'browser': '🌐', 'browser_agent': '🌐', 'web': '🕸️',
    // Code & Development
    'coding': '💻', 'coding_agent': '💻', 'python': '🐍', 'code': '⚡',
    // System & Terminal
    'terminal': '⌨️', 'system': '⚙️',
    // Business & Finance
    'zoho_books': '💼', 'zoho_books_agent': '💼', 'finance': '💰',
    // General & Planning
    'general': '🤖', 'general_agent': '🤖', 'universal': '🌟', 'universal_agent': '🌟',
    'plan': '📋', 'planning': '🗓️',
    // Default fallback
    'default': '🔷'
};

interface ActionHistoryEntry {
    iteration: number;
    action_type: string;
    resource_id: string;     // Agent/tool name - THIS IS THE KEY FIELD!
    instruction: string;
    success: boolean;
    result_summary: string;
    execution_time_ms: number;
    error?: string;
}

interface ActionHistoryTimelineProps {
    history: ActionHistoryEntry[];
}

export default function ActionHistoryTimeline({ history }: ActionHistoryTimelineProps) {
    if (!history || history.length === 0) {
        return (
            <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                <div className="text-4xl mb-3">📝</div>
                <div className="text-sm">No actions executed yet</div>
                <div className="text-xs mt-1">Actions will appear here as they execute</div>
            </div>
        );
    }

    return (
        <div className="p-4 space-y-4">
            {history.map((action, index) => {
                const agentIcon = getAgentIcon(action.resource_id);
                const agentName = formatResourceName(action.resource_id);
                const actionIcon = getActionIcon(action.action_type);
                
                return (
                    <div 
                        key={`action-${action.iteration}-${index}`}
                        className={cn(
                            "relative pl-8 pb-4 border-l-2",
                            action.success 
                                ? "border-green-500 dark:border-green-600" 
                                : "border-red-500 dark:border-red-600"
                        )}
                    >
                        {/* Timeline Dot with Agent Icon */}
                        <div className="absolute left-0 top-0 -translate-x-1/2 w-10 h-10 rounded-full bg-white dark:bg-gray-900 border-2 flex items-center justify-center text-lg border-gray-200 dark:border-gray-700">
                            {agentIcon}
                        </div>
                        
                        {/* Action Card */}
                        <div className={cn(
                            "ml-4 p-4 rounded-lg border shadow-sm",
                            action.success 
                                ? "bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-900" 
                                : "bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900"
                        )}>
                            {/* Header: Agent Name + Action Type + Status */}
                            <div className="flex items-start justify-between gap-2 mb-2">
                                <div className="flex-1">
                                    <div className="flex items-center gap-2">
                                        <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">
                                            {agentName}
                                        </span>
                                        <span className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
                                            {action.action_type}
                                        </span>
                                    </div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                        Iteration {action.iteration}
                                    </div>
                                </div>
                                
                                {/* Status Icon */}
                                <div className="flex-shrink-0">
                                    {action.success ? (
                                        <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                                    ) : (
                                        <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                                    )}
                                </div>
                            </div>
                            
                            {/* Instruction */}
                            <div className="mb-2">
                                <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                                    Instruction:
                                </div>
                                <div className="text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-900 p-2 rounded border border-gray-200 dark:border-gray-800">
                                    {parseInstruction(action.instruction)}
                                </div>
                            </div>
                            
                            {/* Result */}
                            <div className="mb-2">
                                <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                                    Result:
                                </div>
                                <div className={cn(
                                    "text-sm p-2 rounded border",
                                    action.success 
                                        ? "bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800 text-gray-900 dark:text-gray-100"
                                        : "bg-red-100 dark:bg-red-950/50 border-red-300 dark:border-red-900 text-red-900 dark:text-red-100"
                                )}>
                                    {parseResultSummary(action.result_summary, action.success)}
                                </div>
                            </div>
                            
                            {/* Error Details (if failed) */}
                            {!action.success && action.error && (
                                <div className="mb-2">
                                    <div className="text-xs font-medium text-red-700 dark:text-red-300 mb-1">
                                        Error Details:
                                    </div>
                                    <div className="text-xs p-2 rounded bg-red-100 dark:bg-red-950/70 border border-red-300 dark:border-red-900 text-red-900 dark:text-red-100">
                                        {action.error}
                                    </div>
                                </div>
                            )}
                            
                            {/* Execution Time */}
                            <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                                <span>⏱️</span>
                                <span>{(action.execution_time_ms / 1000).toFixed(2)}s</span>
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

// ============================================================================
// HELPER FUNCTIONS (Reused from PlanGraph for consistency)
// ============================================================================

/**
 * Parse instruction field to extract clean text
 * Handles formats like: {"instruction": "Please read the file..."} or plain text
 */
function parseInstruction(instruction: string): string {
    if (!instruction) return 'No instruction provided';
    
    // Try to parse as JSON first
    try {
        // Check if it's a JSON object string
        if (instruction.trim().startsWith('{')) {
            const parsed = JSON.parse(instruction);
            if (parsed.instruction) {
                return parsed.instruction;
            }
        }
    } catch (e) {
        // Not valid JSON, continue with regex extraction
    }
    
    // Try to extract from {"instruction": "text"} format
    const instructionMatch = instruction.match(/["']instruction["']\s*:\s*["']([^"']+)["']/i);
    if (instructionMatch && instructionMatch[1]) {
        return instructionMatch[1];
    }
    
    // Return as-is if it's already clean
    return instruction;
}

/**
 * Parse result_summary to extract clean, user-friendly text
 * Handles formats like:
 * - result: {'task_summary': 'The spreadsheet was successfully...'}
 * - {"task_summary": "Text here"}
 * - Plain text
 */
function parseResultSummary(resultSummary: string, isSuccess: boolean): string {
    if (!resultSummary) {
        return isSuccess ? 'Completed successfully' : 'Operation failed';
    }
    
    // Try to extract task_summary field (most common format)
    const taskSummaryMatch = resultSummary.match(/["']task_summary["']\s*:\s*["']([^"']+)["']/i);
    if (taskSummaryMatch && taskSummaryMatch[1]) {
        return taskSummaryMatch[1];
    }
    
    // Try to parse as JSON
    try {
        // Remove "result: " prefix if present
        let cleanResult = resultSummary.replace(/^result:\s*/i, '').trim();
        
        // Replace Python-style single quotes with double quotes for JSON parsing
        cleanResult = cleanResult.replace(/'/g, '"');
        
        const parsed = JSON.parse(cleanResult);
        
        if (parsed.task_summary) {
            return parsed.task_summary;
        }
        if (parsed.result) {
            return parsed.result;
        }
        if (parsed.message) {
            return parsed.message;
        }
        
        // If it's a simple object, stringify it nicely
        return JSON.stringify(parsed, null, 2);
    } catch (e) {
        // Not valid JSON, continue with text extraction
    }
    
    // Clean up common technical prefixes and artifacts
    let cleaned = resultSummary
        .replace(/^result:\s*/i, '')
        .replace(/^{[^:]+:\s*["']?/i, '')  // Remove opening dict-like patterns
        .replace(/["']?}$/i, '')            // Remove closing braces
        .replace(/^["']/, '')                // Remove leading quotes
        .replace(/["']$/, '')                // Remove trailing quotes
        .trim();
    
    // If we still have technical artifacts, try one more extraction
    if (cleaned.includes('task_summary')) {
        const simpleMatch = cleaned.match(/task_summary["']?\s*:\s*["']?([^"']+)/i);
        if (simpleMatch && simpleMatch[1]) {
            return simpleMatch[1];
        }
    }
    
    // Return cleaned version
    return cleaned || (isSuccess ? 'Completed successfully' : 'Operation failed');
}

/**
 * Get appropriate icon for an agent/resource
 */
function getAgentIcon(resourceId: string): string {
    if (!resourceId) return AGENT_ICONS.default;
    
    const lowerResource = resourceId.toLowerCase();
    
    // Direct match
    if (AGENT_ICONS[lowerResource]) {
        return AGENT_ICONS[lowerResource];
    }
    
    // Keyword matching for flexibility
    if (lowerResource.includes('mail') || lowerResource.includes('gmail')) return '📧';
    if (lowerResource.includes('doc') || lowerResource.includes('document')) return '📄';
    if (lowerResource.includes('sheet') || lowerResource.includes('spreadsheet')) return '📊';
    if (lowerResource.includes('browse') || lowerResource.includes('web')) return '🌐';
    if (lowerResource.includes('code') || lowerResource.includes('coding')) return '💻';
    if (lowerResource.includes('python')) return '🐍';
    if (lowerResource.includes('terminal') || lowerResource.includes('shell')) return '⌨️';
    if (lowerResource.includes('book') || lowerResource.includes('finance') || lowerResource.includes('zoho')) return '💼';
    if (lowerResource.includes('plan')) return '📋';
    
    // Default fallback
    return AGENT_ICONS.default;
}

/**
 * Format resource names for display
 * "gmail_agent" → "Gmail Agent"
 */
function formatResourceName(resourceId: string): string {
    if (!resourceId) return 'Unknown Agent';
    
    return resourceId
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

/**
 * Get icon based on action type
 */
function getActionIcon(type: string): string {
    const icons: Record<string, string> = {
        agent: '🤖',
        tool: '🔧',
        python: '🐍',
        terminal: '⌨️',
        plan: '📋',
        replan: '🔄',
        parallel: '⚡',
        finish: '✨'
    };
    return icons[type] || '❓';
}
