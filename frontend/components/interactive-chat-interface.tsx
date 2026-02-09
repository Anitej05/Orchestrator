'use client'

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { MessageCircle, Clock, CheckCircle, Paperclip, X, File as FileIcon, AlertCircle, Loader2, Brain, Search, Users, FileText, Play, BarChart3, ChevronDown, ChevronUp, Globe, Copy, Check, Volume2, VolumeX, Mic, MicOff, Square } from 'lucide-react';
import Markdown from '@/components/ui/markdown';
import { type ProcessResponse, type ConversationState, type Message, type Attachment } from '@/lib/types';
import { PlanApprovalModal } from '@/components/plan-approval-modal';
import { useConversationStore } from '@/lib/conversation-store';
import { useTTS } from '@/hooks/useTTS';
import { useSpeechToText } from '@/hooks/useSpeechToText';
import { AudioWaveSVG } from '@/components/ui/audio-wave-animation';

interface InteractiveChatInterfaceProps {
  onWorkflowComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
  className?: string;
  state: ConversationState;
  isLoading: boolean;
  startConversation: (input: string, files?: File[], planningMode?: boolean, owner?: string) => Promise<void>;
  continueConversation: (input: string, files?: File[], planningMode?: boolean, owner?: string) => Promise<void>;
  resetConversation: () => void;
  onViewCanvas?: (canvasContent: string, canvasType: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'email_preview' | 'document' | 'image' | 'json') => void;
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
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);

  // TTS Hook for Read Aloud functionality
  const { speak, stop: stopSpeaking, isSpeaking, isGenerating: isTTSGenerating, isLoading: isTTSLoading } = useTTS();

  // STT Hook for Voice Input functionality
  const {
    startListening,
    stopListening,
    isListening,
    transcript,
    interimTranscript,
    audioLevel,
    isSupported: isSTTSupported,
    resetTranscript,
  } = useSpeechToText({
    continuous: false,
    onResult: (text) => {
      // Append recognized text to input
      setInputValue(prev => prev + (prev ? ' ' : '') + text);
    },
  });

  // Handle read aloud for a message
  const handleReadAloud = async (messageId: string, content: string) => {
    if (speakingMessageId === messageId && (isSpeaking || isTTSGenerating)) {
      // Stop if already speaking or generating this message
      stopSpeaking();
      setSpeakingMessageId(null);
    } else {
      // Stop any current speech and start new
      stopSpeaking();
      setSpeakingMessageId(messageId);
      await speak(content);
    }
  };

  // Sync speaking state with UI
  useEffect(() => {
    if (!isSpeaking && !isTTSGenerating && speakingMessageId) {
      setSpeakingMessageId(null);
    }
  }, [isSpeaking, isTTSGenerating, speakingMessageId]);

  // Handle microphone toggle
  const handleMicrophoneToggle = () => {
    if (isListening) {
      stopListening();
    } else {
      resetTranscript();
      startListening();
    }
  };

  // Copy message content to clipboard with animation feedback
  const copyToClipboard = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      // Reset after 2 seconds
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

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
        if (fileInputRef.current) fileInputRef.current.value = ''; // Reset file input
      }
    } else if (state.metadata?.currentStage === 'validating' || state.status === 'planning_complete') {
      // User is modifying a saved workflow plan
      if (inputValue.trim() || attachedFiles.length > 0) {
        // Send the modification as a regular user message - backend will handle combining it with original prompt
        await continueConversation(inputValue, attachedFiles, false, owner);
        setInputValue('');
        setAttachedFiles([]);
        if (fileInputRef.current) fileInputRef.current.value = ''; // Reset file input
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
        if (fileInputRef.current) fileInputRef.current.value = ''; // Reset file input
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
    <div className={`flex flex-col h-full overflow-hidden ${className}`}>
      {/* Chat Messages - Full Width Edge-to-Edge */}
      <div className="flex-1 overflow-y-auto bg-bg-subtle w-full">
        {state.messages.length === 0 && !isBrowserRunning && (
          <div className="text-center py-8 h-full flex flex-col justify-center items-center">
            <div className="p-6 rounded-full bg-bg-card/40 backdrop-blur-sm mb-6 border border-border-color-light">
              <MessageCircle className="w-16 h-16 text-text-tertiary" />
            </div>
            <p className="ui-section-header">Start a conversation to orchestrate your workflow</p>
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
              <div key={messageId} className={`message message-${message.type} w-full flex px-6 py-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`${message.type === 'user' 
                  ? 'ui-user-bubble' 
                  : message.type === 'system' 
                    ? 'ui-system-bubble' 
                    : 'ui-agent-bubble'
                  }`}>
                  <div className="message-content space-y-2">
                    {message.content && (message.type === 'assistant' ? <Markdown content={message.content} /> : <p>{message.content}</p>)}

                    {/* Collapsible browsing trace */}
                    {message.browsing_trace && message.browsing_trace.length > 0 && (
                      <div className="browsing-trace mt-3">
                        <button
                          onClick={() => toggleTrace(messageId)}
                          className="flex items-center gap-2 ui-metadata-label hover:text-text-secondary transition-colors"
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
                          <div className="mt-2 space-y-2 ui-metadata-item">
                            <h4 className="ui-metadata-label mb-2">Browsing Trace:</h4>
                            {message.browsing_trace.map((step, i) => (
                              <div key={`${messageId}-trace-${step.step_number || i}-${step.action}`} className="flex items-start gap-3 p-2 ui-card">
                                <div className="flex-shrink-0 mt-0.5">
                                  {step.status === 'success' && <CheckCircle className="w-4 h-4 text-status-success" />}
                                  {step.status === 'error' && <AlertCircle className="w-4 h-4 text-status-error" />}
                                  {step.status === 'pending' && <Loader2 className="w-4 h-4 animate-spin text-status-active" />}
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between">
                                    <span className="ui-task-name">
                                      {step.step_number}. {step.action}
                                    </span>
                                    {step.duration && (
                                      <span className="ui-metadata-mono">
                                        {step.duration.toFixed(1)}s
                                      </span>
                                    )}
                                  </div>
                                  <p className="ui-file-meta mt-1 truncate">
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
                              <img src={att.content} alt={att.name} className="max-w-xs max-h-48 rounded-orbimesh-lg" />
                            ) : (
                              <div className="flex items-center gap-2 ui-metadata-item">
                                <FileIcon className="w-4 h-4" />
                                <span className="ui-file-name">{att.name}</span>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    {/* View in Canvas button for messages with canvas content or data */}
                    {message.has_canvas && (message.canvas_content || (message as any).canvas_data) && message.canvas_type && (
                      <Button
                        variant="ui-secondary"
                        size="sm"
                        className="mt-2 text-xs"
                        onClick={() => onViewCanvas?.(message.canvas_content || JSON.stringify((message as any).canvas_data || {}), message.canvas_type!)}
                      >
                        <FileText className="w-3 h-3 mr-1" />
                        View in Canvas
                      </Button>
                    )}
                  </div>
                  {/* Footer with timestamp and copy button */}
                  <div className={`flex items-center justify-between mt-1.5 ${message.type === 'user' ? 'text-text-tertiary' : 'text-text-tertiary'}`}>
                    <div className="ui-file-meta opacity-60">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                    {/* Action buttons for assistant messages */}
                    {message.type === 'assistant' && message.content && (
                      <div className="flex items-center gap-1">
                        {/* Read Aloud button */}
                        <button
                          onClick={() => handleReadAloud(messageId, message.content!)}
                          className={`p-1.5 rounded-orbimesh-lg transition-all duration-300 ease-out hover:bg-bg-hover ${speakingMessageId === messageId && isSpeaking
                            ? 'text-brand-teal scale-110'
                            : (speakingMessageId === messageId && isTTSGenerating) || (isTTSLoading && speakingMessageId === messageId)
                              ? 'text-text-tertiary hover:text-text-secondary'
                              : 'text-text-tertiary hover:text-text-secondary'
                            }`}
                          title={
                            speakingMessageId === messageId && isSpeaking
                              ? 'Stop reading'
                              : speakingMessageId === messageId && isTTSGenerating
                                ? 'Generating audio... (click to cancel)'
                                : isTTSLoading && speakingMessageId === messageId
                                  ? 'Loading TTS model...'
                                  : 'Read aloud'
                          }
                          disabled={isTTSLoading && speakingMessageId !== messageId}
                        >
                          {speakingMessageId === messageId && isSpeaking ? (
                            <Square className="w-4 h-4" />
                          ) : speakingMessageId === messageId && isTTSGenerating ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : isTTSLoading && speakingMessageId === messageId ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Volume2 className="w-4 h-4" />
                          )}
                        </button>
                        {/* Copy button */}
                        <button
                          onClick={() => copyToClipboard(messageId, message.content!)}
                          className={`p-1.5 rounded-orbimesh-lg transition-all duration-300 ease-out hover:bg-bg-hover ${copiedMessageId === messageId
                            ? 'text-status-success scale-110'
                            : 'text-text-tertiary hover:text-text-secondary'
                            }`}
                          title={copiedMessageId === messageId ? 'Copied!' : 'Copy message'}
                        >
                          {copiedMessageId === messageId ? (
                            <Check className="w-4 h-4 animate-bounce" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}

        {/* Live Browser Stream - shown AFTER messages while browser is running */}
        {isBrowserRunning && (
          <div className="browser-live-stream w-full flex justify-start">
            <div className="w-full max-w-[95%] rounded-orbimesh-xl overflow-hidden shadow-lg ui-card bg-bg-card">
              {/* Header */}
              <div className="bg-bg-subtle px-4 py-3 flex items-center justify-between border-b border-border-color">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="w-3 h-3 bg-status-success rounded-full animate-pulse"></div>
                    <div className="absolute inset-0 w-3 h-3 bg-status-success rounded-full animate-ping opacity-75"></div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-brand-teal" />
                    <span className="ui-task-name text-text-primary">Live Browser View</span>
                  </div>
                </div>
                <div className="flex items-center gap-2 ui-file-meta text-text-tertiary">
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
      <div className="p-4 bg-bg-subtle/95 backdrop-blur-lg border-t border-border-color">
        {/* Consolidated Status Indicator - Shows orchestration progress above input */}
        {(isLoading || state.status === 'processing') && !state.isWaitingForUser && (
          <div className={`status-indicator p-3 rounded-orbimesh-lg mb-4 ${state.isWaitingForUser
            ? 'bg-status-pending-light border border-status-pending'
            : state.metadata?.currentStage === 'completed'
              ? 'bg-status-success-light border border-status-success'
              : state.metadata?.currentStage === 'error'
                ? 'bg-status-error-light border border-status-error'
                : state.metadata?.currentStage === 'parsing'
                  ? 'bg-status-active-light border border-status-active'
                  : state.metadata?.currentStage === 'searching'
                    ? 'bg-status-success-light border border-status-success'
                    : state.metadata?.currentStage === 'ranking'
                      ? 'bg-status-pending-light border border-status-pending'
                      : state.metadata?.currentStage === 'planning'
                        ? 'bg-status-active-light border border-status-active'
                        : state.metadata?.currentStage === 'validating'
                          ? 'bg-brand-teal-light border border-brand-teal'
                          : state.metadata?.currentStage === 'executing'
                            ? 'bg-status-error-light border border-status-error'
                            : state.metadata?.currentStage === 'aggregating'
                              ? 'bg-status-active-light border border-status-active'
                              : 'bg-brand-teal-light border border-brand-teal'  // default initializing state
            }`}>
            <div className="flex items-center space-x-2">
              {state.isWaitingForUser ? (
                <AlertCircle className="w-4 h-4 text-status-pending" />
              ) : state.metadata?.currentStage === 'completed' ? (
                <CheckCircle className="w-4 h-4 text-status-success" />
              ) : state.metadata?.currentStage === 'error' ? (
                <AlertCircle className="w-4 h-4 text-status-error" />
              ) : state.metadata?.currentStage === 'parsing' ? (
                <Brain className="w-4 h-4 text-status-active" />
              ) : state.metadata?.currentStage === 'searching' ? (
                <Search className="w-4 h-4 text-status-success" />
              ) : state.metadata?.currentStage === 'ranking' ? (
                <Users className="w-4 h-4 text-status-pending" />
              ) : state.metadata?.currentStage === 'planning' ? (
                <FileText className="w-4 h-4 text-status-active" />
              ) : state.metadata?.currentStage === 'validating' ? (
                <CheckCircle className="w-4 h-4 text-brand-teal" />
              ) : state.metadata?.currentStage === 'executing' ? (
                <Play className="w-4 h-4 text-status-error" />
              ) : state.metadata?.currentStage === 'aggregating' ? (
                <BarChart3 className="w-4 h-4 text-status-active" />
              ) : (
                <Loader2 className={`w-4 h-4 animate-spin text-brand-teal`} />
              )}
              <span className={`ui-metadata-label ${state.isWaitingForUser
                ? 'text-status-pending-dark'
                : state.metadata?.currentStage === 'completed'
                  ? 'text-status-success-dark'
                  : state.metadata?.currentStage === 'error'
                    ? 'text-status-error'
                    : state.metadata?.currentStage === 'parsing'
                      ? 'text-status-active-dark'
                      : state.metadata?.currentStage === 'searching'
                        ? 'text-status-success-dark'
                        : state.metadata?.currentStage === 'ranking'
                          ? 'text-status-pending-dark'
                          : state.metadata?.currentStage === 'planning'
                            ? 'text-status-active-dark'
                            : state.metadata?.currentStage === 'validating'
                              ? 'text-brand-teal'
                              : state.metadata?.currentStage === 'executing'
                                ? 'text-status-error'
                                : state.metadata?.currentStage === 'aggregating'
                                  ? 'text-status-active-dark'
                                  : 'text-brand-teal'  // default initializing state
                }`}>
                {state.metadata?.stageMessage || (state.isWaitingForUser ? 'Waiting for your response...' : 'Processing your request...')}
              </span>
              {state.metadata?.progress && !state.isWaitingForUser && state.metadata?.currentStage !== 'completed' && state.metadata?.currentStage !== 'error' && (
                <div className="flex-1 bg-border-DEFAULT rounded-full h-2 ml-4 overflow-hidden">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${state.metadata?.currentStage === 'initializing' ? 'bg-brand-teal' :
                      state.metadata?.currentStage === 'parsing' ? 'bg-status-active' :
                        state.metadata?.currentStage === 'searching' ? 'bg-status-success' :
                          state.metadata?.currentStage === 'ranking' ? 'bg-status-pending' :
                            state.metadata?.currentStage === 'planning' ? 'bg-status-active' :
                              state.metadata?.currentStage === 'validating' ? 'bg-brand-teal' :
                                state.metadata?.currentStage === 'executing' ? 'bg-status-error' :
                                  state.metadata?.currentStage === 'aggregating' ? 'bg-status-active' :
                                    'bg-brand-teal'
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
                  className="ui-textarea min-h-[120px] text-base"
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
                      <img src={previewUrls[index]} alt={file.name} className="h-20 w-20 object-cover rounded-orbimesh-md" />
                      <button
                        type="button"
                        onClick={() => removeFile(file.name)}
                        className="absolute top-0 right-0 bg-status-error text-white rounded-full p-1 text-xs hover:bg-red-600 transition-colors"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                  {attachedFiles.filter(f => !f.type.startsWith('image/')).map(file => (
                    <Badge key={file.name} variant="ui-pending" className="flex items-center gap-1 pr-1">
                      <FileIcon className="w-4 h-4" />
                      <span className="max-w-[200px] truncate">{file.name}</span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeFile(file.name);
                        }}
                        className="ml-1 hover:bg-red-500/20 rounded-full p-0.5 transition-colors"
                        title="Remove attachment"
                      >
                        <X className="w-3 h-3 cursor-pointer hover:text-status-error" />
                      </button>
                    </Badge>
                  ))}
                </div>

                {/* Planning Mode Toggle - Above Textarea */}
                <div className="flex items-center gap-2">
                  <Switch
                    id="planning-mode"
                    checked={planningMode}
                    onCheckedChange={setPlanningMode}
                    className="border-2 border-brand-teal rounded-full [&>span]:bg-slate-700 [&>span]:dark:bg-white [&[data-state=checked]>span]:bg-white"
                  />
                  <label
                    htmlFor="planning-mode"
                    className="ui-metadata-label cursor-pointer select-none"
                  >
                    Planning Mode
                  </label>
                  {planningMode && (
                    <Badge variant="ui-pending" className="ui-file-meta">
                      Will pause for approval
                    </Badge>
                  )}
                </div>

                <div className="space-y-2">
                  <div className="relative flex-1">
                    {/* Audio Recording Overlay */}
                    {isListening && (
                      <div className="absolute inset-0 bg-brand-teal-light rounded-orbimesh-lg border-2 border-brand-teal flex flex-col items-center justify-center z-10">
                        <AudioWaveSVG
                          isActive={isListening}
                          audioLevel={audioLevel}
                          color="#0D9488"
                          width={160}
                          height={50}
                        />
                        <div className="flex items-center gap-2 mt-3">
                          <div className="w-2 h-2 bg-status-error rounded-full animate-pulse"></div>
                          <span className="ui-metadata-label text-brand-teal">
                            Listening...
                          </span>
                        </div>
                        {(transcript || interimTranscript) && (
                          <p className="ui-file-meta mt-2 px-4 text-center max-w-full truncate">
                            {transcript}{interimTranscript && <span className="opacity-60">{interimTranscript}</span>}
                          </p>
                        )}
                      </div>
                    )}
                    <Textarea
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      placeholder="Describe what you want to accomplish..."
                      disabled={isLoading || isListening}
                      className={`ui-textarea min-h-[60px] text-base ${isListening ? 'opacity-0' : ''}`}
                      onKeyDown={handleKeyDown}
                    />
                  </div>
                </div>
              </>
            )}

            <div className="flex items-center justify-between gap-2">
              {/* Attachment and Audio Buttons - Left Side */}
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="ui-secondary"
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
                {/* Microphone button for voice input */}
                {isSTTSupported && (
                  <Button
                    type="button"
                    variant={isListening ? "default" : "ui-secondary"}
                    size="sm"
                    onClick={handleMicrophoneToggle}
                    disabled={isLoading}
                    className={isListening
                      ? "bg-status-error hover:bg-status-error text-foreground border-status-error"
                      : ""
                    }
                    title={isListening ? "Stop recording" : "Start voice input"}
                  >
                    {isListening ? (
                      <MicOff className="w-4 h-4" />
                    ) : (
                      <Mic className="w-4 h-4" />
                    )}
                  </Button>
                )}
              </div>

              {/* Action Buttons - Right Side */}
              <div className="flex items-center gap-2">
                {/* Plan Approval Buttons - Only show when in planning mode with approval required, or executing saved workflows */}
                {((planningMode && state.approval_required) || ((state.metadata?.currentStage === 'validating' || state.status === 'planning_complete') && onAcceptPlan && state.metadata?.from_workflow)) ? (
                  <>
                    {/* Only show Modify button for non-saved workflows when in planning mode */}
                    {planningMode && !state.metadata?.from_workflow && (
                      <Button
                        type="button"
                        variant="ui-secondary"
                        size="sm"
                        onClick={handleModifyPlan}
                      >
                        Modify Plan
                      </Button>
                    )}

                    {/* Show Accept button when approval is needed in planning mode or for saved workflows */}
                    <Button
                      type="button"
                      size="sm"
                      onClick={handleAcceptAndExecute}
                      className="bg-gradient-to-r from-brand-teal to-status-active hover:from-brand-teal-hover hover:to-status-active text-foreground shadow-md"
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
                  variant="ui-primary"
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
                    variant="ui-secondary"
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
        onClose={() => { }}
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

