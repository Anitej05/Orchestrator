// components/interactive-chat-interface.tsx
"use client"

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { MessageCircle, Clock, CheckCircle } from 'lucide-react';
import Markdown from '@/components/ui/markdown';
import { type ProcessResponse, type ConversationState } from '@/lib/api-client';

interface InteractiveChatInterfaceProps {
  onWorkflowComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
  className?: string;
  state: ConversationState;
  isLoading: boolean;
  startConversation: (input: string) => Promise<void>;
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
  const [inputValue, setInputValue] = useState('');
  const [userResponse, setUserResponse] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (state.isWaitingForUser) {
      if (userResponse.trim()) {
        await continueConversation(userResponse);
        setUserResponse('');
      }
    } else {
      if (inputValue.trim()) {
        await startConversation(inputValue);
        setInputValue('');
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
        
        {state.messages.map((message) => (
          <div key={message.id} className={`message message-${message.type} w-full flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-4 rounded-lg ${
              message.type === 'user' 
                ? 'bg-blue-600 text-white max-w-[85%]' 
                : message.type === 'system'
                ? 'bg-yellow-50 border border-yellow-200 text-yellow-800 max-w-[90%]'
                : 'bg-gray-100 border border-gray-200 max-w-[95%]'
            }`}>
              <div className="message-content">
                {message.type === 'assistant' ? (
                  <div className="text-gray-800">
                    <Markdown content={message.content} />
                  </div>
                ) : (
                  message.content
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
          )}
          
          <div className="flex justify-between items-center">
             <p className="text-xs text-gray-400">
              Orbimesh Orchestrator
            </p>
            <div className="flex space-x-2">
              <Button 
                type="submit" 
                disabled={isLoading || (!inputValue.trim() && !userResponse.trim())}
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