'use client'

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { MessageCircle, Clock, CheckCircle, Paperclip, X, File as FileIcon, AlertCircle, Loader2, Brain, Search, Users, FileText, Play, BarChart3, ChevronDown, ChevronUp, Globe } from 'lucide-react';
import Markdown from '@/components/ui/markdown';
import { type ProcessResponse, type ConversationState, type Message, type Attachment } from '@/lib/types';
import { PlanApprovalModal } from '@/components/plan-approval-modal';
import { useConversationStore } from '@/lib/conversation-store';

interface InteractiveChatInterfaceProps {
  onWorkflowComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
  className?: string;
  state: ConversationState;
  isLoading: boolean;
  startConversation: (input: string, files?: File[], planningMode?: boolean, owner?: string) => Promise<void>;
  continueConversation: (input: string, files?: File[], planningMode?: boolean, owner?: string) => Promise<void>;
  resetConversation: () => void;
  onViewCanvas?: (canvasContent: string, canvasType: 'html' | 'markdown') => void;
  owner?: string;
  onAcceptPlan?: (modifiedPrompt?: string) => Promise<void>;
}

export function InteractiveChatInterface({
  onWorkflowComplete,
  onError,
  className = "",
  state = {
    thread_id: '',
    status: 'idle',
    messages: [],
    isWaitingForUser: false,
    currentQuestion: '',
  },
  isLoading,
  startConversation,
  continueConversation,
  resetConversation,
  onViewCanvas,
  owner,
  onAcceptPlan
}: InteractiveChatInterfaceProps) {
  useEffect(() => {
    if (!state) {
      console.debug('InteractiveChatInterface: no state prop received');
      return;
    }
    if (!Array.isArray(state.messages)) {
      console.warn('InteractiveChatInterface: state.messages is not an array', state.messages);
      return;
    }
    console.debug('InteractiveChatInterface: rendering messages count=', state.messages.length);
    if (state.messages.length > 0) console.debug('InteractiveChatInterface: sample', state.messages.slice(0, 3));
  }, [state, state.messages.length]);
  const [inputValue, setInputValue] = useState('');
  const [userResponse, setUserResponse] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [planningMode, setPlanningMode] = useState(false);
  const [expandedTraces, setExpandedTraces] = useState<Set<string>>(new Set());
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const toggleTrace = (messageId: string) => {
    setExpandedTraces(prev => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  };

  // Plan approval modal handlers
  const handleApprovePlan = async () => {
    console.log('Plan approved by user');
    // DON'T clear isWaitingForUser yet - continueConversation needs to see it's true
    // Only clear approval_required to close the modal
    useConversationStore.setState({ 
      approval_required: false
    });
    // continueConversation will capture isWaitingForUser=true and send user_response
    await continueConversation('approve', [], false, owner);
  };

  const handleCancelPlan = async () => {
    console.log('Plan cancelled by user');
    // DON'T clear isWaitingForUser yet - continueConversation needs to see it's true
    // Only clear approval_required to close the modal
    useConversationStore.setState({ 
      approval_required: false
    });
    // continueConversation will capture isWaitingForUser=true and send user_response
    await continueConversation('cancel', [], false, owner);
  };

  const handleModifyPlan = async () => {
    console.log('User wants to modify plan');
    // Just close the approval modal and let user type modifications
    // The modify logic will be handled in handleSubmit when they click "Modify" button
    useConversationStore.setState({ 
      approval_required: false
    });
  };

  // Handler for Accept & Execute button (uses parent's logic)
  const handleAcceptAndExecute = async () => {
    console.log('User accepts and executes plan');
    
    // Check if this is from planning mode (approval_required) or saved workflow
    if (state.approval_required) {
      // Planning mode - send 'approve' to backend to continue execution
      useConversationStore.setState({ 
        approval_required: false
      });
      await continueConversation('approve', [], false, owner);
    } else if (onAcceptPlan) {
      // Saved workflow - use parent's logic
      useConversationStore.setState({ 
        approval_required: false
      });
      await onAcceptPlan();
    }
  };

  useEffect(() => {
    const urls = attachedFiles
      .filter(file => file.type.startsWith('image/'))
      .map(file => URL.createObjectURL(file));
    setPreviewUrls(urls);

    return () => {
      urls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [attachedFiles]);

  // Listen for auto-execute workflow events from URL parameters
  useEffect(() => {
    const handleAutoExecute = (event: CustomEvent) => {
      const { prompt } = event.detail;
      if (prompt && !state.thread_id) {
        console.log('Auto-executing workflow with prompt:', prompt);
        setInputValue(prompt);
        // Auto-submit after a short delay to ensure state is ready
        setTimeout(() => {
          startConversation(prompt, [], planningMode, owner);
        }, 100);
      }
    };

    window.addEventListener('autoExecuteWorkflow' as any, handleAutoExecute as any);
    
    return () => {
      window.removeEventListener('autoExecuteWorkflow' as any, handleAutoExecute as any);
    };
  }, [state.thread_id, startConversation, planningMode, owner]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileList = e.target.files;
      const newFiles: File[] = [];
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList.item(i);
        if (file) {
          newFiles.push(file);
        }
      }
      setAttachedFiles(prev => [...prev, ...newFiles]);
    }
  };

  const removeFile = (fileName: string) => {
    setAttachedFiles(prev => prev.filter(f => f.name !== fileName));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (state.isWaitingForUser) {
      // User is responding to a question from the system
      if (userResponse.trim()) {
        await continueConversation(userResponse, attachedFiles, planningMode, owner);
        setUserResponse('');
        setAttachedFiles([]); // Clear files after submission
      }
    } else if (state.metadata?.currentStage === 'validating' || state.status === 'planning_complete') {
      // User is modifying a saved workflow plan
      if (inputValue.trim() || attachedFiles.length > 0) {
        // Send the modification as a regular user message - backend will handle combining it with original prompt
        await continueConversation(inputValue, attachedFiles, false, owner);
        setInputValue('');
        setAttachedFiles([]);
      }
    } else {
      // Check if this is a continuation of an existing conversation or a new one
      const hasExistingConversation = state.thread_id && state.messages.length > 0;
      
      if (inputValue.trim() || attachedFiles.length > 0) {
        if (hasExistingConversation) {
          // Continue existing conversation with planning mode
          await continueConversation(inputValue, attachedFiles, planningMode, owner);
        } else {
          // Start new conversation with planning mode
          await startConversation(inputValue, attachedFiles, planningMode, owner);
        }
        setInputValue('');
        setAttachedFiles([]);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };


  // Get browser view from conversation store for live streaming
  const browserView = useConversationStore((s) => (s as any).browser_view);
  const currentStage = state.metadata?.currentStage;
  const isBrowserRunning = currentStage === 'executing' && browserView;

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-gray-50 via-white to-gray-50/50 dark:from-gray-950 dark:via-gray-900/95 dark:to-gray-950">
      {/* Chat Messages */}
      <div className="flex-1 space-y-6 overflow-y-auto p-6">
        {state.messages.length === 0 && !isBrowserRunning && (
          <div className="text-center text-gray-400 dark:text-gray-500 py-8 h-full flex flex-col justify-center items-center">
            <div className="p-6 rounded-full bg-white/40 dark:bg-gray-800/40 backdrop-blur-sm mb-6 border border-gray-200/50 dark:border-gray-700/50">
              <MessageCircle className="w-16 h-16 text-gray-300 dark:text-gray-600" />
            </div>
            <p className="text-lg font-medium">Start a conversation to orchestrate your workflow</p>
          </div>
        )}
        
        {state.messages
          .filter((message: Message) => {
            // Filter out empty assistant messages to prevent empty bubbles
            if (message.type === 'assistant') {
              return message.content && message.content.trim() !== '';
            }
            return true;
          })
          .map((message: Message, index: number) => {
            // Ensure message.id is a valid string
            const messageId = message.id || `message-${index}-${Date.now()}`;
            
            return (
              <div key={messageId} className={`message message-${message.type} w-full flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`p-4 rounded-2xl shadow-lg ${
                  message.type === 'user' 
                    ? 'bg-gradient-to-br from-blue-600 to-blue-700 dark:from-blue-600 dark:to-blue-800 text-white max-w-[85%] shadow-lg shadow-blue-500/30'
                    : message.type === 'system'
                    ? 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/40 text-amber-900 dark:text-amber-200 max-w-[90%] shadow-md'
                    : 'bg-white dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700/40 text-gray-900 dark:text-gray-100 max-w-[95%] shadow-md'
                }`}>
                  <div className="message-content space-y-2">
                    {message.content && (message.type === 'assistant' ? <Markdown content={message.content} /> : <p>{message.content}</p>)}
                    
                    {/* Collapsible browsing trace */}
                    {message.browsing_trace && message.browsing_trace.length > 0 && (
                      <div className="browsing-trace mt-3">
                        <button
                          onClick={() => toggleTrace(messageId)}
                          className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors"
                        >
                          {expandedTraces.has(messageId) ? (
                            <ChevronUp className="w-4 h-4" />
                          ) : (
                            <ChevronDown className="w-4 h-4" />
                          )}
                          <span>
                            {expandedTraces.has(messageId) ? 'Hide' : 'View'} browsing trace 
                            ({message.browsing_trace.length} {message.browsing_trace.length === 1 ? 'action' : 'actions'})
                          </span>
                        </button>
                        
                        {expandedTraces.has(messageId) && (
                          <div className="mt-2 space-y-2 p-3 bg-gray-50 dark:bg-gray-800/30 rounded-lg border border-gray-200 dark:border-gray-800/30">
                            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Browsing Trace:</h4>
                            {message.browsing_trace.map((step, i) => (
                              <div key={i} className="flex items-start gap-3 p-2 bg-white dark:bg-gray-800/50 rounded border border-gray-100 dark:border-gray-800/30">
                                <div className="flex-shrink-0 mt-0.5">
                                  {step.status === 'success' && <CheckCircle className="w-4 h-4 text-green-500" />}
                                  {step.status === 'error' && <AlertCircle className="w-4 h-4 text-red-500" />}
                                  {step.status === 'pending' && <Loader2 className="w-4 h-4 animate-spin text-blue-500" />}
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                                      {step.step_number}. {step.action}
                                    </span>
                                    {step.duration && (
                                      <span className="text-xs text-gray-500 dark:text-gray-400">
                                        {step.duration.toFixed(1)}s
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 truncate">
                                    {step.description}
                                  </p>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {message.attachments && message.attachments.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-2">
                        {message.attachments.map((att: Attachment, attIndex: number) => (
                          <div key={`${messageId}-attachment-${attIndex}`}>
                            {att.type.startsWith('image/') && att.content ? (
                              <img src={att.content} alt={att.name} className="max-w-xs max-h-48 rounded-lg" />
                            ) : (
                              <div className="flex items-center gap-2 p-2 rounded-md bg-gray-20 text-sm">
                                <FileIcon className="w-4 h-4" />
                                <span>{att.name}</span>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    {/* View in Canvas button for messages with canvas content */}
                    {message.has_canvas && message.canvas_content && message.canvas_type && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-2 text-xs"
                        onClick={() => onViewCanvas?.(message.canvas_content!, message.canvas_type!)}
                      >
                        <FileText className="w-3 h-3 mr-1" />
                        View in Canvas
                      </Button>
                    )}
                  </div>
                  <div className={`text-xs opacity-60 mt-1.5 font-medium ${message.type === 'user' ? 'text-blue-100 dark:text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}>
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            );
          })}
        
        {/* Live Browser Stream - shown AFTER messages while browser is running */}
        {isBrowserRunning && (
          <div className="browser-live-stream w-full flex justify-start">
            <div className="w-full max-w-[95%] rounded-2xl overflow-hidden shadow-lg border border-gray-200 dark:border-gray-700 bg-gray-900">
              {/* Header */}
              <div className="bg-gradient-to-r from-gray-800 to-gray-900 px-4 py-3 flex items-center justify-between border-b border-gray-700">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    <div className="absolute inset-0 w-3 h-3 bg-green-500 rounded-full animate-ping opacity-75"></div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-semibold text-white">Live Browser View</span>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span>Automating...</span>
                </div>
              </div>
              {/* Browser Content */}
              <div className="relative w-full" style={{ aspectRatio: '16/10' }}>
                <iframe
                  srcDoc={browserView}
                  className="w-full h-full border-0"
                  sandbox="allow-scripts allow-same-origin"
                  title="Live Browser View"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Form */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-800/30 bg-gray-50/80 dark:bg-gray-800/70 backdrop-blur-lg rounded-b-lg">
        {/* Consolidated Status Indicator - Shows orchestration progress above input */}
        {(isLoading || state.status === 'processing') && !state.isWaitingForUser && (
          <div className={`status-indicator p-3 rounded-lg mb-4 ${
            state.isWaitingForUser
              ? 'bg-yellow-50 border border-yellow-200' 
              : state.metadata?.currentStage === 'completed'
              ? 'bg-green-50 border border-green-200'
              : state.metadata?.currentStage === 'error'
              ? 'bg-red-50 border border-red-200'
              : state.metadata?.currentStage === 'parsing'
              ? 'bg-purple-50 border border-purple-200'
              : state.metadata?.currentStage === 'searching'
              ? 'bg-green-50 border border-green-200'
              : state.metadata?.currentStage === 'ranking'
              ? 'bg-orange-50 border border-orange-200'
              : state.metadata?.currentStage === 'planning'
              ? 'bg-indigo-50 border border-indigo-200'
              : state.metadata?.currentStage === 'validating'
              ? 'bg-teal-50 border border-teal-200'
              : state.metadata?.currentStage === 'executing'
              ? 'bg-red-50 border border-red-200'
              : state.metadata?.currentStage === 'aggregating'
              ? 'bg-cyan-50 border border-cyan-200'
              : 'bg-blue-50 border border-blue-200'  // default initializing state
          }`}>
            <div className="flex items-center space-x-2">
              {state.isWaitingForUser ? (
                <AlertCircle className="w-4 h-4 text-yellow-600" />
              ) : state.metadata?.currentStage === 'completed' ? (
                <CheckCircle className="w-4 h-4 text-green-600" />
              ) : state.metadata?.currentStage === 'error' ? (
                <AlertCircle className="w-4 h-4 text-red-600" />
              ) : state.metadata?.currentStage === 'parsing' ? (
                <Brain className="w-4 h-4 text-purple-600" />
              ) : state.metadata?.currentStage === 'searching' ? (
                <Search className="w-4 h-4 text-green-600" />
              ) : state.metadata?.currentStage === 'ranking' ? (
                <Users className="w-4 h-4 text-orange-600" />
              ) : state.metadata?.currentStage === 'planning' ? (
                <FileText className="w-4 h-4 text-indigo-600" />
              ) : state.metadata?.currentStage === 'validating' ? (
                <CheckCircle className="w-4 h-4 text-teal-600" />
              ) : state.metadata?.currentStage === 'executing' ? (
                <Play className="w-4 h-4 text-red-600" />
              ) : state.metadata?.currentStage === 'aggregating' ? (
                <BarChart3 className="w-4 h-4 text-cyan-600" />
              ) : (
                <Loader2 className={`w-4 h-4 animate-spin text-blue-600`} />
              )}
              <span className={`font-medium text-sm ${
                state.isWaitingForUser
                  ? 'text-yellow-800' 
                  : state.metadata?.currentStage === 'completed'
                  ? 'text-green-800'
                  : state.metadata?.currentStage === 'error'
                  ? 'text-red-800'
                  : state.metadata?.currentStage === 'parsing'
                  ? 'text-purple-800'
                  : state.metadata?.currentStage === 'searching'
                  ? 'text-green-800'
                  : state.metadata?.currentStage === 'ranking'
                  ? 'text-orange-800'
                  : state.metadata?.currentStage === 'planning'
                  ? 'text-indigo-800'
                  : state.metadata?.currentStage === 'validating'
                  ? 'text-teal-800'
                  : state.metadata?.currentStage === 'executing'
                  ? 'text-red-800'
                  : state.metadata?.currentStage === 'aggregating'
                  ? 'text-cyan-800'
                  : 'text-blue-800'  // default initializing state
              }`}>
                 {state.metadata?.stageMessage || (state.isWaitingForUser ? 'Waiting for your response...' : 'Processing your request...')}
              </span>
              {state.metadata?.progress && !state.isWaitingForUser && state.metadata?.currentStage !== 'completed' && state.metadata?.currentStage !== 'error' && (
                <div className="flex-1 bg-gray-200 dark:bg-gray-700/50 rounded-full h-2 ml-4 overflow-hidden">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      state.metadata?.currentStage === 'initializing' ? 'bg-blue-500' :
                      state.metadata?.currentStage === 'parsing' ? 'bg-purple-500' :
                      state.metadata?.currentStage === 'searching' ? 'bg-green-500' :
                      state.metadata?.currentStage === 'ranking' ? 'bg-orange-500' :
                      state.metadata?.currentStage === 'planning' ? 'bg-indigo-500' :
                      state.metadata?.currentStage === 'validating' ? 'bg-teal-500' :
                      state.metadata?.currentStage === 'executing' ? 'bg-red-500' :
                      state.metadata?.currentStage === 'aggregating' ? 'bg-cyan-500' :
                      'bg-blue-500'
                    }`} 
                    style={{ width: `${state.metadata.progress}%` }}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Hide input form when browser is running - only show progress bar */}
        {!isBrowserRunning && (
        <form onSubmit={handleSubmit} className="space-y-4">
          {state.isWaitingForUser ? (
            <div className="space-y-3">
              <Textarea
                value={userResponse}
                onChange={(e) => setUserResponse(e.target.value)}
                placeholder="Type your response here..."
                disabled={isLoading}
                className="min-h-[120px] text-base"
                onKeyDown={handleKeyDown}
                autoFocus
              />
            </div>
          ) : (
            <>
              {/* Attached Files Preview */}
              <div className="flex flex-wrap gap-2">
                {attachedFiles.filter(f => f.type.startsWith('image/')).map((file, index) => (
                  <div key={file.name} className="relative">
                    <img src={previewUrls[index]} alt={file.name} className="h-20 w-20 object-cover rounded-md" />
                    <button 
                      type="button"
                      onClick={() => removeFile(file.name)} 
                      className="absolute top-0 right-0 bg-red-500 text-white rounded-full p-1 text-xs"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
                {attachedFiles.filter(f => !f.type.startsWith('image/')).map(file => (
                  <Badge key={file.name} variant="secondary" className="flex items-center gap-1">
                    <FileIcon className="w-4 h-4" />
                    {file.name}
                    <X className="w-3 h-3 cursor-pointer" onClick={() => removeFile(file.name)} />
                  </Badge>
                ))}
              </div>

              {/* Planning Mode Toggle */}
              <div className="flex items-center justify-between px-1 py-2">
                <div className="flex items-center gap-2">
                  <Switch
                    id="planning-mode"
                    checked={planningMode}
                    onCheckedChange={setPlanningMode}
                  />
                  <label 
                    htmlFor="planning-mode" 
                    className="text-sm font-medium cursor-pointer select-none"
                  >
                    Planning Mode
                  </label>
                </div>
                {planningMode && (
                  <Badge variant="secondary" className="text-xs">
                    Will pause for approval
                  </Badge>
                )}
              </div>

              <div className="space-y-2">
                <Textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Describe what you want to accomplish..."
                  disabled={isLoading}
                  className="min-h-[60px] text-base"
                  onKeyDown={handleKeyDown}
                />
              </div>
            </>
          )}
          
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <Button 
                type="button" 
                variant="outline" 
                size="sm" 
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
              >
                <Paperclip className="w-4 h-4" />
              </Button>
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                onChange={handleFileChange} 
                multiple 
              />
            </div>
            <div className="flex space-x-2">
              {/* Plan Approval Buttons - Show for both planning mode and saved workflows */}
              {(state.approval_required || ((state.metadata?.currentStage === 'validating' || state.status === 'planning_complete') && onAcceptPlan)) ? (
                <>
                  {/* Only show Modify button for non-saved workflows and not in planning mode */}
                  {!state.metadata?.from_workflow && !state.approval_required && (
                    <Button 
                      type="button"
                      variant="outline" 
                      size="sm" 
                      onClick={handleModifyPlan}
                      className="border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800"
                    >
                      Modify Plan
                    </Button>
                  )}
                  
                  {/* Always show Accept button when approval is needed */}
                  <Button 
                    type="button"
                    size="sm" 
                    onClick={handleAcceptAndExecute}
                    className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md"
                  >
                    Accept & Execute
                  </Button>
                </>
              ) : null}
              
              <Button 
                type="submit" 
                disabled={
                  isLoading || 
                  (state.isWaitingForUser ? !userResponse.trim() : (!inputValue.trim() && attachedFiles.length === 0))
                }
              >
                {isLoading 
                  ? 'Processing...' 
                  : (state.metadata?.currentStage === 'validating' || state.status === 'planning_complete')
                  ? 'Modify'
                  : state.isWaitingForUser 
                  ? 'Send Response' 
                  : 'Start Workflow'}
              </Button>
              
              {state.messages.length > 0 && (
                <Button 
                  type="button" 
                  variant="outline"
                  onClick={resetConversation}
                  disabled={isLoading}
                >
                  Reset
                </Button>
              )}
            </div>
          </div>
        </form>
        )}
      </div>

      {/* Plan Approval Modal - Disabled, now using chat interface buttons */}
      <PlanApprovalModal
        isOpen={false}
        onClose={() => {}}
        onApprove={handleApprovePlan}
        onModify={handleModifyPlan}
        onCancel={handleCancelPlan}
        taskPlan={state.task_plan || []}
        taskAgentPairs={state.task_agent_pairs || []}
        estimatedCost={state.estimated_cost || 0}
        taskCount={state.task_count || 0}
      />
    </div>
  );
}
