'use client'

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { MessageCircle, Clock, CheckCircle, Paperclip, X, File as FileIcon, AlertCircle, Loader2, Brain, Search, Users, FileText, Play, BarChart3 } from 'lucide-react';
import Markdown from '@/components/ui/markdown';
import { type ProcessResponse, type ConversationState, type Message, type Attachment } from '@/lib/types';

interface InteractiveChatInterfaceProps {
  onWorkflowComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
  className?: string;
  state: ConversationState;
  isLoading: boolean;
  startConversation: (input: string, files?: File[]) => Promise<void>;
  continueConversation: (input: string, files?: File[]) => Promise<void>;
  resetConversation: () => void;
  onViewCanvas?: (canvasContent: string, canvasType: 'html' | 'markdown') => void;
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
  onViewCanvas
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
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const urls = attachedFiles
      .filter(file => file.type.startsWith('image/'))
      .map(file => URL.createObjectURL(file));
    setPreviewUrls(urls);

    return () => {
      urls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [attachedFiles]);

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
      if (userResponse.trim()) {
        await continueConversation(userResponse, attachedFiles);
        setUserResponse('');
        setAttachedFiles([]); // Clear files after submission
      }
    } else {
      if (inputValue.trim() || attachedFiles.length > 0) {
        await startConversation(inputValue, attachedFiles);
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


  return (
    <div className="flex flex-col h-full">
      {/* Chat Messages */}
      <div className="flex-1 space-y-4 overflow-y-auto p-6">
        {state.messages.length === 0 && (
          <div className="text-center text-gray-500 py-8 h-full flex flex-col justify-center items-center">
            <MessageCircle className="w-12 h-12 mx-auto mb-2 text-gray-300" />
            <p>Start a conversation to orchestrate your workflow</p>
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
                <div className={`p-4 rounded-lg ${
                  message.type === 'user' 
                    ? 'bg-blue-600 text-white max-w-[85%]'
                    : message.type === 'system'
                    ? 'bg-yellow-50 border border-yellow-200 text-yellow-800 max-w-[90%]'
                    : 'bg-gray-100 border border-gray-200 max-w-[95%]'
                }`}>
                  <div className="message-content space-y-2">
                    {message.content && (message.type === 'assistant' ? <Markdown content={message.content} /> : <p>{message.content}</p>)}
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
                  <div className={`text-xs opacity-70 mt-1 ${message.type === 'user' ? 'text-blue-200' : 'text-gray-500'}`}>
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            );
          })}

        {/* Orchestration Pause Section - Show Continue Button */}
        {state.status === 'orchestration_paused' && (
          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-blue-900 flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                Orchestration Paused - Review & Continue
              </h3>
            </div>

            {/* Show Parsed Tasks */}
            {state.parsed_tasks && state.parsed_tasks.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-blue-800 mb-2">Parsed Tasks:</h4>
                <div className="space-y-2">
                  {state.parsed_tasks.map((task: any, index: number) => (
                    <div key={index} className="p-3 bg-white rounded-lg border border-blue-100">
                      <p className="text-sm text-gray-700">{task.description || task.task_description}</p>
                      {task.capability && (
                        <Badge variant="outline" className="mt-1 text-xs">
                          {task.capability}
                        </Badge>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Show Selected Agents */}
            {state.task_agent_pairs && state.task_agent_pairs.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-blue-800 mb-2">Selected Agents:</h4>
                <div className="space-y-2">
                  {state.task_agent_pairs.map((pair: any, index: number) => (
                    <div key={index} className="p-3 bg-white rounded-lg border border-blue-100">
                      <div className="flex justify-between items-start">
                        <p className="text-sm text-gray-700 flex-1">{pair.task?.description || pair.task?.task_description}</p>
                        <Badge className="ml-2 bg-blue-600 text-white">
                          {pair.agent?.name || 'Unknown Agent'}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Continue Button */}
            <Button 
              onClick={async () => {
                await continueConversation("continue_orchestration");
              }}
              className="w-full bg-green-600 hover:bg-green-700 text-white font-medium"
            >
              <CheckCircle className="w-4 h-4 mr-2" />
              Continue Orchestration
            </Button>
          </div>
        )}
      </div>

      {/* Input Form */}
      <div className="p-4 border-t bg-white rounded-b-lg">
        {/* Consolidated Status Indicator - Shows orchestration progress above input */}
        {(isLoading || state.status === 'processing' || state.isWaitingForUser) && (
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
                <div className="flex-1 bg-gray-200 rounded-full h-2 ml-4">
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

        <form onSubmit={handleSubmit} className="space-y-4">
          {state.isWaitingForUser ? (
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                {state.currentQuestion}
              </label>
              <Textarea
                value={userResponse}
                onChange={(e) => setUserResponse(e.target.value)}
                placeholder="Your response..."
                disabled={isLoading}
                className="min-h-[120px] text-base"
                onKeyDown={handleKeyDown}
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
              <Button 
                type="submit" 
                disabled={isLoading || (!inputValue.trim() && attachedFiles.length === 0 && !userResponse.trim())}
              >
                {isLoading ? 'Processing...' : state.isWaitingForUser ? 'Send Response' : 'Start Workflow'}
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

        {/* Status Indicator for user input */}
        {state.isWaitingForUser && !isLoading && (
          <div className="status-indicator mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4 text-yellow-600" />
              <span className="text-yellow-800 font-medium text-sm">Waiting for your response...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
