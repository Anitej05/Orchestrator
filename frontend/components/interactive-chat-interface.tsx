'use client'

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { MessageCircle, Clock, CheckCircle, Paperclip, X, File as FileIcon } from 'lucide-react';
import Markdown from '@/components/ui/markdown';
import { type ProcessResponse, type ConversationState, type Message, type Attachment } from '@/lib/types';

interface InteractiveChatInterfaceProps {
  onWorkflowComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
  className?: string;
  state: ConversationState;
  isLoading: boolean;
  startConversation: (input: string, files?: File[]) => Promise<void>;
  continueConversation: (input: string) => Promise<void>;
  resetConversation: () => void;
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
  resetConversation
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
  }, [state]);
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
        await continueConversation(userResponse);
        setUserResponse('');
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
        
        {state.messages.map((message: Message) => (
          <div key={message.id} className={`message message-${message.type} w-full flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-4 rounded-lg ${
              message.type === 'user' 
                ? 'bg-blue-600 text-white max-w-[85%]'
                : message.type === 'system'
                ? 'bg-yellow-50 border border-yellow-200 text-yellow-800 max-w-[90%]'
                : 'bg-gray-100 border border-gray-200 max-w-[95%]'
            }`}>
              <div className="message-content space-y-2">
                {message.content && (message.type === 'assistant' ? <Markdown content={message.content} /> : message.content)}
                {message.attachments && message.attachments.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {message.attachments.map((att: Attachment, index: number) => (
                      <div key={index}>
                        {att.type.startsWith('image/') && att.content ? (
                          <img src={att.content} alt={att.name} className="max-w-xs max-h-48 rounded-lg" />
                        ) : (
                          <div className="flex items-center gap-2 p-2 rounded-md bg-gray-200 text-sm">
                            <FileIcon className="w-4 h-4" />
                            <span>{att.name}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div className={`text-xs opacity-70 mt-1 ${message.type === 'user' ? 'text-blue-200' : 'text-gray-500'}`}>
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Input Form */}
      <div className="p-4 border-t bg-white rounded-b-lg">
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
                {previewUrls.map((url, index) => (
                  <div key={index} className="relative">
                    <img src={url} alt={`preview ${index}`} className="h-20 w-20 object-cover rounded-md" />
                    <button 
                      type="button"
                      onClick={() => removeFile(attachedFiles[index].name)} 
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

        {/* Status Indicator */}
        {state.status === 'pending_user_input' && !isLoading && (
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
