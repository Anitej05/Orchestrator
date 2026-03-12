'use client'

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { MessageCircle, CheckCircle, Paperclip, X, File as FileIcon, AlertCircle, Loader2, FileText, ChevronDown, ChevronUp, Globe, Copy, Check, Volume2, Mic, MicOff, Square, ArrowUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import Markdown from '@/components/ui/markdown';
import { type ProcessResponse, type ConversationState, type Message, type Attachment, type CanvasType } from '@/lib/types';
import { useConversationStore } from '@/lib/conversation-store';
import { authFetch } from '@/lib/auth-fetch';
import { API_BASE_URL } from '@/lib/config';
import { useTTS } from '@/hooks/useTTS';
import { useSpeechToText } from '@/hooks/useSpeechToText';
import { AudioWaveSVG } from '@/components/ui/audio-wave-animation';
import { EmailResultCard } from '@/components/email-result-card';
import { ExposedFilesPanel } from '@/components/exposed-files-panel';

interface InteractiveChatInterfaceProps {
  onWorkflowComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
  className?: string;
  state: ConversationState;
  isLoading: boolean;
  startConversation: (input: string, files?: File[], planningMode?: boolean, owner?: string) => Promise<void>;
  continueConversation: (input: string, files?: File[], planningMode?: boolean, owner?: string) => Promise<void>;
  resetConversation: () => void;
  onViewCanvas?: (canvasContent: string, canvasType: CanvasType) => void;
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
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or loading state change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.messages.length, isLoading, state.status]);
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

  const handleModifyPlan = async () => {
    console.log('User wants to modify plan');
    // Just close the approval modal and let user type modifications
    // The modify logic will be handled in handleSubmit when they click "Modify" button
    useConversationStore.setState({
      approval_required: false
    });
  };

  const handleApproveAction = async () => {
    if (!state.thread_id) return;

    try {
      await authFetch(`${API_BASE_URL}/api/orchestrator/action/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: state.thread_id })
      });
    } catch (error) {
      console.error('Failed to approve action:', error);
      return;
    }

    useConversationStore.setState({
      pending_action_approval: false,
      pending_action: undefined
    });

    await continueConversation('approve', [], false, owner);
  };

  const handleRejectAction = async () => {
    if (!state.thread_id) return;

    try {
      await authFetch(`${API_BASE_URL}/api/orchestrator/action/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: state.thread_id, reason: 'User rejected' })
      });
    } catch (error) {
      console.error('Failed to reject action:', error);
      return;
    }

    useConversationStore.setState({
      pending_action_approval: false,
      pending_action: undefined
    });

    await continueConversation('reject', [], false, owner);
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
      <div className="flex-1 overflow-y-auto bg-bg-card w-full">
        {state.messages.length === 0 && !isBrowserRunning && (
          <div className="text-center py-8 h-full flex flex-col justify-center items-center">
            {state.metadata?.status === 'empty' ? (
              <>
                <div className="p-4 rounded-full bg-yellow-50/10 backdrop-blur-sm mb-4 border border-yellow-500/20">
                  <AlertCircle className="w-10 h-10 text-yellow-600" />
                </div>
                <p className="ui-section-header mb-2">Conversation history unavailable</p>
                <p className="text-text-tertiary max-w-md mb-6">
                  This conversation was created before our save system improvements. The message history could not be recovered. Start a new conversation to keep your messages safe.
                </p>
                <Button onClick={resetConversation} variant="secondary" size="sm">
                  Start a New Conversation
                </Button>
              </>
            ) : state.metadata?.status === 'recovered_from_database' ? (
              <>
                <div className="p-4 rounded-full bg-blue-50/10 backdrop-blur-sm mb-4 border border-blue-500/20">
                  <AlertCircle className="w-10 h-10 text-blue-600" />
                </div>
                <p className="ui-section-header mb-2">Conversation recovered</p>
                <p className="text-text-tertiary max-w-md">
                  Some messages were recovered from our database. This history may be incomplete.
                </p>
              </>
            ) : (
              <>
                <div className="relative mb-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-brand-primary/10 to-status-active/10 border border-brand-primary/20 flex items-center justify-center shadow-md">
                    <MessageCircle className="w-8 h-8 text-brand-primary" />
                  </div>
                  <span className="absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full bg-status-active/30 border-2 border-bg-subtle animate-ping" />
                </div>
                <p className="text-base font-semibold text-text-primary">Start a conversation</p>
                <p className="text-sm text-text-tertiary mt-1 max-w-[200px] leading-relaxed text-center">Describe a task and your agents will handle it</p>
              </>
            )}
          </div>
        )}

        {state.messages
          .filter((message: Message) => {
            // Filter out empty messages - must have content, attachments, canvas, or browsing trace
            const hasContent = message.content && message.content.trim() !== '';
            const hasAttachments = message.attachments && message.attachments.length > 0;
            const hasCanvas = message.has_canvas && (message.canvas_content || (message as any).canvas_data);
            const hasBrowsingTrace = message.browsing_trace && message.browsing_trace.length > 0;

            return hasContent || hasAttachments || hasCanvas || hasBrowsingTrace;
          })
          .map((message: Message, index: number) => {
            // Ensure message.id is a valid string
            const messageId = message.id || `message-${index}-${Date.now()}`;

            const isUser = message.type === 'user';
            const isAssistant = message.type === 'assistant';

            return (
              <div key={messageId} className={cn(
                "message w-full flex items-start px-6 py-3",
                isUser ? "justify-end" : "justify-start",
                isAssistant && "gap-3"
              )}>
                {/* Avatar for assistant messages */}
                {isAssistant && (
                  <div className="flex-shrink-0 mt-[3px]">
                    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <circle cx="13" cy="13" r="7.5" stroke="#2C4BA8" strokeWidth="6" fill="none" />
                    </svg>
                  </div>
                )}

                {/* For user messages: wrapper to put timestamp below bubble */}
                {isUser ? (
                  <div className="flex flex-col items-end gap-1">
                    <div className="ui-user-bubble">
                      <div className="message-content space-y-2">
                        {message.content && message.content.trim() !== '' && (
                          <p>{message.content}</p>
                        )}
                        {message.attachments && message.attachments.length > 0 && (
                          <div className="flex flex-wrap gap-2 mt-2">
                            {message.attachments.map((att: Attachment, attIndex: number) => (
                              <div key={`${messageId}-attachment-${attIndex}`}>
                                {att.type.startsWith('image/') && att.content ? (
                                  <img src={att.content} alt={att.name} className="max-w-xs max-h-48 rounded-orbimesh-lg" />
                                ) : (
                                  <div className="flex items-center gap-2 px-3 py-2 rounded-orbimesh-md bg-bg-card border border-border-color">
                                    <FileIcon className="w-4 h-4 text-text-tertiary" />
                                    <span className="text-sm text-text-primary">{att.name}</span>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                        {/* Inline email results for email canvas type */}
                        {message.has_canvas && (message as any).canvas_type === 'email' && (message as any).canvas_data && (
                          <EmailResultCard
                            messages={((message as any).canvas_data as any)?.messages || []}
                            totalCount={((message as any).canvas_data as any)?.total_count}
                            query={((message as any).canvas_data as any)?.query}
                            className="mt-3"
                          />
                        )}
                        {/* View in Canvas button for messages with canvas content or data (non-email) */}
                        {message.has_canvas && (message.canvas_content || (message as any).canvas_data) && message.canvas_type && message.canvas_type !== 'email' && (
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

                        {/* Render exposed files for user message if any */}
                        {message.exposed_files && message.exposed_files.length > 0 && (
                          <ExposedFilesPanel files={message.exposed_files} />
                        )}
                      </div>
                    </div>
                    {/* Footer with timestamp and copy button */}
                    <div className={`flex items-center justify-between mt-1.5 ${message.type === 'user' ? 'text-text-tertiary' : 'text-text-tertiary'}`}>
                      <div className="ui-file-meta opacity-60">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                      {/* Timestamp below bubble, outside */}
                      <span className="text-[10px] text-text-disabled px-1">{message.timestamp.toLocaleTimeString()}</span>
                    </div>
                  </div>
                ) : (
                  <div className={message.type === 'system' ? 'ui-system-bubble' : 'ui-agent-bubble'}>
                    <div className="message-content space-y-2">
                      {message.content && message.content.trim() !== '' && (
                        isAssistant ? <Markdown content={message.content} /> : <p>{message.content}</p>
                      )}

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
                                <div className="flex items-center gap-2 px-3 py-2 rounded-orbimesh-md bg-bg-card border border-border-color">
                                  <FileIcon className="w-4 h-4 text-text-tertiary" />
                                  <span className="text-sm text-text-primary">{att.name}</span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      {/* View in Canvas button */}
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

                      {/* Render exposed files for assistant message */}
                      {message.exposed_files && message.exposed_files.length > 0 && (
                        <ExposedFilesPanel files={message.exposed_files} />
                      )}
                    </div>
                    {/* Footer: timestamp + action buttons */}
                    <div className="flex items-center justify-between mt-1.5 text-text-tertiary">
                      <div className="ui-file-meta opacity-60">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                      {isAssistant && message.content && (
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
                )}
              </div>
            );
          })}

        {/* Thinking dots while loading */}
        {(isLoading || state.status === 'processing') && !state.isWaitingForUser && !isBrowserRunning && (
          <div className="w-full flex justify-start px-6 py-3 gap-3">
            <div className="flex-shrink-0 mt-1">
              <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="13" cy="13" r="7.5" stroke="#2C4BA8" strokeWidth="6" fill="none" />
              </svg>
            </div>
            <div className="flex items-center gap-1.5 pt-1">
              <span className="w-1.5 h-1.5 rounded-full bg-text-tertiary animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-text-tertiary animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-text-tertiary animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />

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
      <div className="px-4 pb-4 pt-2 bg-bg-card border-t border-border-color">

        {/* Hide input form when browser is running */}
        {!isBrowserRunning && (
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            {state.isWaitingForUser ? (
              <div className="space-y-3">
                <Textarea
                  value={userResponse}
                  onChange={(e) => setUserResponse(e.target.value)}
                  placeholder="Type your response here..."
                  disabled={isLoading}
                  className="ui-textarea min-h-[80px] text-base"
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
              </div>
            ) : (
              <>
                {/* Attached Files Preview */}
                {attachedFiles.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-2">
                    {attachedFiles.filter(f => f.type.startsWith('image/')).map((file, index) => (
                      <div key={file.name} className="relative">
                        <img src={previewUrls[index]} alt={file.name} className="h-16 w-16 object-cover rounded-md" />
                        <button
                          type="button"
                          onClick={() => removeFile(file.name)}
                          className="absolute top-0 right-0 bg-status-error text-white rounded-full p-0.5 hover:bg-red-600 transition-colors"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                    {attachedFiles.filter(f => !f.type.startsWith('image/')).map(file => (
                      <Badge key={file.name} variant="ui-pending" className="flex items-center gap-1 pr-1">
                        <FileIcon className="w-3.5 h-3.5" />
                        <span className="max-w-[200px] truncate text-xs">{file.name}</span>
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); removeFile(file.name); }}
                          className="ml-1 hover:bg-red-500/20 rounded-full p-0.5 transition-colors"
                        >
                          <X className="w-3 h-3 cursor-pointer hover:text-status-error" />
                        </button>
                      </Badge>
                    ))}
                  </div>
                )}

                {/* Textarea — subtle card with focus highlight */}
                <div className="relative rounded-xl border border-border-color bg-bg-subtle px-3 py-2 transition-all duration-150 focus-within:border-brand-primary/50 focus-within:bg-bg-card focus-within:shadow-[0_0_0_3px_rgba(44,75,168,0.08)]">
                  {isListening && (
                    <div className="absolute inset-0 bg-brand-teal-light rounded-xl border-2 border-brand-teal flex flex-col items-center justify-center z-10">
                      <AudioWaveSVG isActive={isListening} audioLevel={audioLevel} color="#0D9488" width={160} height={50} />
                      <div className="flex items-center gap-2 mt-3">
                        <div className="w-2 h-2 bg-status-error rounded-full animate-pulse"></div>
                        <span className="ui-metadata-label text-brand-teal">Listening...</span>
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
                    className={`w-full resize-none bg-transparent border-none shadow-none text-sm text-text-primary placeholder:text-text-tertiary focus-visible:ring-0 focus-visible:ring-offset-0 focus:outline-none min-h-[60px] py-1 px-0 ${isListening ? 'opacity-0' : ''}`}
                    onKeyDown={handleKeyDown}
                  />
                </div>

                {/* Toolbar row: planning mode | attachment | mic | send */}
                <div className="flex items-center justify-between gap-2 mt-1 pt-2 border-t border-border-color/60">
                  {/* Left: planning mode + attachment + mic */}
                  <div className="flex items-center gap-1.5">
                    {/* Planning Mode toggle */}
                    <label className="planning-mode-switch">
                      <input
                        type="checkbox"
                        checked={planningMode}
                        onChange={(e) => setPlanningMode(e.target.checked)}
                      />
                      <div className="slider">
                        <div className="circle">
                          <svg className="cross" viewBox="0 0 365.696 365.696" height="6" width="6" xmlns="http://www.w3.org/2000/svg">
                            <g><path fill="currentColor" d="M243.188 182.86 356.32 69.726c12.5-12.5 12.5-32.766 0-45.247L341.238 9.398c-12.504-12.503-32.77-12.503-45.25 0L182.86 122.528 69.727 9.374c-12.5-12.5-32.766-12.5-45.247 0L9.375 24.457c-12.5 12.504-12.5 32.77 0 45.25l113.152 113.152L9.398 295.99c-12.503 12.503-12.503 32.769 0 45.25L24.48 356.32c12.5 12.5 32.766 12.5 45.247 0l113.132-113.132L295.99 356.32c12.503 12.5 32.769 12.5 45.25 0l15.081-15.082c12.5-12.504 12.5-32.77 0-45.25zm0 0"></path></g>
                          </svg>
                          <svg className="checkmark" viewBox="0 0 24 24" height="10" width="10" xmlns="http://www.w3.org/2000/svg">
                            <g><path fill="currentColor" d="M9.707 19.121a.997.997 0 0 1-1.414 0l-5.646-5.647a1.5 1.5 0 0 1 0-2.121l.707-.707a1.5 1.5 0 0 1 2.121 0L9 14.171l9.525-9.525a1.5 1.5 0 0 1 2.121 0l.707.707a1.5 1.5 0 0 1 0 2.121z"></path></g>
                          </svg>
                        </div>
                      </div>
                    </label>
                    <span className="ui-metadata-label cursor-pointer select-none text-xs">
                      Planning{planningMode && <span className="ml-1 text-brand-primary">·</span>}
                    </span>

                    <div className="h-4 w-px bg-border-color mx-1" />

                    {/* Attachment */}
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isLoading}
                      className="w-7 h-7 p-0 rounded-md text-text-tertiary hover:text-text-primary hover:bg-bg-hover"
                      title="Attach file"
                    >
                      <Paperclip className="w-3.5 h-3.5" />
                    </Button>
                    <input type="file" ref={fileInputRef} className="hidden" onChange={handleFileChange} multiple />

                    {/* Mic */}
                    {isSTTSupported && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={handleMicrophoneToggle}
                        disabled={isLoading}
                        className={cn(
                          "w-7 h-7 p-0 rounded-md",
                          isListening
                            ? "text-status-error bg-status-error/10 hover:bg-status-error/20"
                            : "text-text-tertiary hover:text-text-primary hover:bg-bg-hover"
                        )}
                        title={isListening ? "Stop recording" : "Voice input"}
                      >
                        {isListening ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}
                      </Button>
                    )}
                  </div>

                  {/* Right: approval buttons + send */}
                  <div className="flex items-center gap-2">
                    {state.pending_action_approval && (
                      <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-amber-50 border border-amber-200">
                        <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                        <span className="text-xs text-amber-800">{state.currentQuestion || 'Approval required.'}</span>
                        <Button type="button" size="sm" className="bg-status-active text-foreground h-6 text-xs px-2" onClick={handleApproveAction}>Approve</Button>
                        <Button type="button" size="sm" variant="ui-secondary" className="h-6 text-xs px-2" onClick={handleRejectAction}>Reject</Button>
                      </div>
                    )}

                    {((planningMode && state.approval_required) || ((state.metadata?.currentStage === 'validating' || state.status === 'planning_complete') && onAcceptPlan && state.metadata?.from_workflow)) && (
                      <>
                        {planningMode && !state.metadata?.from_workflow && (
                          <Button type="button" variant="ui-secondary" size="sm" onClick={handleModifyPlan} className="h-7 text-xs">Modify Plan</Button>
                        )}
                        <Button
                          type="button"
                          size="sm"
                          onClick={handleAcceptAndExecute}
                          className="h-7 text-xs bg-gradient-to-r from-brand-teal to-status-active hover:from-brand-teal-hover hover:to-status-active text-foreground"
                        >
                          Accept & Execute
                        </Button>
                      </>
                    )}

                    {(() => {
                      const hasContent = !!(inputValue.trim() || attachedFiles.length > 0);
                      return (
                        <Button
                          type="submit"
                          disabled={isLoading || !hasContent}
                          className={cn(
                            "transition-all duration-200 h-7 w-7 p-0 rounded-md",
                            hasContent
                              ? "bg-brand-primary hover:bg-brand-primary-hover text-white shadow-sm"
                              : "bg-bg-subtle text-text-disabled cursor-not-allowed"
                          )}
                          title="Send"
                        >
                          {isLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ArrowUp className="w-3.5 h-3.5" />}
                        </Button>
                      );
                    })()}

                    {state.messages.length > 0 && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={resetConversation}
                        disabled={isLoading}
                        className="w-7 h-7 p-0 rounded-md text-text-disabled hover:text-status-error hover:bg-status-error/10 transition-colors"
                        title="Reset conversation"
                      >
                        <X className="w-3.5 h-3.5" />
                      </Button>
                    )}
                  </div>
                </div>
              </>
            )}

            {/* Waiting for user: simple toolbar with just send */}
            {state.isWaitingForUser && (
              <div className="flex justify-end mt-2">
                {(() => {
                  const hasContent = !!userResponse.trim();
                  return (
                    <Button
                      type="submit"
                      disabled={isLoading || !hasContent}
                      className={cn(
                        "transition-all duration-200 gap-1.5",
                        hasContent
                          ? "bg-brand-primary hover:bg-brand-primary-hover text-white rounded-full px-4 shadow-sm"
                          : "rounded-full w-8 h-8 p-0 bg-bg-subtle text-text-disabled cursor-not-allowed"
                      )}
                    >
                      {isLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ArrowUp className="w-3.5 h-3.5" />}
                      {hasContent && <span className="text-sm">Send Response</span>}
                    </Button>
                  );
                })()}
              </div>
            )}
          </form>
        )}
      </div>

    </div>
  );
}

