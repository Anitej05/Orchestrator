"use client"

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { MessageCircle, Wifi, WifiOff, Play, Pause, RotateCcw } from 'lucide-react';
import { useWebSocketManager } from '@/hooks/use-websocket-conversation';
import { type ProcessResponse } from '@/lib/types';
import { useConversationStore } from '@/lib/conversation-store';

interface WebSocketWorkflowProps {
  onWorkflowComplete?: (result: ProcessResponse) => void;
  onError?: (error: string) => void;
  className?: string;
}

export function WebSocketWorkflow({ 
  onWorkflowComplete, 
  onError,
  className = "" 
}: WebSocketWorkflowProps) {
  const [inputPrompt, setInputPrompt] = useState('');
  const [isStarted, setIsStarted] = useState(false);

  // Initialize the WebSocket manager. It will automatically connect
  // and keep the Zustand store in sync with backend updates.
  useWebSocketManager();
  
  // Get conversation state from Zustand store
  const conversationState = useConversationStore();
  const { 
    startConversation, 
    continueConversation, 
    resetConversation, 
    loadConversation 
  } = useConversationStore(state => state.actions);
  const isConversationLoading = useConversationStore((state: any) => state.isLoading);
  
  const messages = conversationState.messages;
  const isConnected = true; // Assuming always connected when using WebSocketManager
  const waitingForUser = conversationState.isWaitingForUser;
  const currentQuestion = conversationState.currentQuestion;
  const progress = conversationState.metadata?.progress || 0;
  const currentThreadId = conversationState.thread_id;
  
  const sendMessage = async (input: string, threadId?: string) => {
    if (threadId) {
      await continueConversation(input, [], false);
    } else {
      await startConversation(input, [], false);
    }
  };
  
  const connect = () => {
    // Connection is handled by useWebSocketManager
    console.log('WebSocket connection managed by useWebSocketManager');
  };
  
  const disconnect = () => {
    // Disconnection is handled by useWebSocketManager
    console.log('WebSocket disconnection managed by useWebSocketManager');
  };
  
  const clearMessages = () => {
    resetConversation();
  };

  const handleStartWorkflow = () => {
    if (!inputPrompt.trim()) return;
    
    sendMessage(inputPrompt);
    setIsStarted(true);
  };

  const handleUserResponse = () => {
    const userInput = (document.getElementById('userResponseInput') as HTMLInputElement)?.value;
    if (!userInput?.trim() || !currentThreadId) return;
    
    sendMessage(userInput, currentThreadId);
    (document.getElementById('userResponseInput') as HTMLInputElement).value = '';
  };

  const handleReset = () => {
    clearMessages();
    setIsStarted(false);
    setInputPrompt('');
  };

  const getConnectionStatus = () => {
    if (isConnected) {
      return <div className="flex items-center space-x-1 text-green-600">
        <Wifi className="w-4 h-4" />
        <span className="text-sm">Connected</span>
      </div>;
    } else {
      return <div className="flex items-center space-x-1 text-red-600">
        <WifiOff className="w-4 h-4" />
        <span className="text-sm">Disconnected</span>
      </div>;
    }
  };

  return (
    <div className={`websocket-workflow ${className}`}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Real-time WebSocket Workflow</span>
            {getConnectionStatus()}
          </CardTitle>
          <p className="text-sm text-gray-600">
            Experience real-time workflow orchestration with live progress updates via WebSocket connection.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Initial Input */}
          {!isStarted && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Describe your workflow
                </label>
                <textarea
                  value={inputPrompt}
                  onChange={(e) => setInputPrompt(e.target.value)}
                  placeholder="E.g., Help me analyze customer data and create a marketing strategy..."
                  className="w-full min-h-[100px] p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  disabled={!isConnected}
                />
              </div>
              <div className="flex space-x-2">
                <Button 
                  onClick={handleStartWorkflow}
                  disabled={!isConnected || !inputPrompt.trim()}
                  className="flex-1"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Real-time Workflow
                </Button>
                <Button 
                  onClick={connect}
                  variant="outline"
                  disabled={isConnected}
                >
                  <Wifi className="w-4 h-4 mr-2" />
                  Connect
                </Button>
              </div>
            </div>
          )}

          {/* Progress Indicator */}
          {isStarted && progress > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Workflow Progress</span>
                <span className="text-sm text-gray-600">{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="w-full" />
            </div>
          )}

          {/* Messages Display */}
          {messages.length > 0 && (
            <div className="space-y-4">
              <h4 className="font-medium text-gray-900">Conversation</h4>
              <div className="max-h-96 overflow-y-auto space-y-3 p-4 bg-gray-50 rounded-lg">
                {messages
                  .filter((message) => {
                    // Filter out empty assistant messages to prevent empty bubbles
                    if (message.type === 'assistant') {
                      return message.content && message.content.trim() !== '';
                    }
                    return true;
                  })
                  .map((message) => (
                    <div key={message.id} className={`message message-${message.type}`}>
                      <div className={`p-3 rounded-lg max-w-[80%] ${
                        message.type === 'user' 
                          ? 'bg-blue-500 text-white ml-auto' 
                          : message.type === 'system'
                          ? 'bg-yellow-50 border border-yellow-200 text-yellow-800'
                          : 'bg-white border border-gray-200'
                      }`}>
                        <div className="text-sm font-medium mb-1 opacity-70">
                          {message.type === 'user' ? 'You' : message.type === 'system' ? 'System' : 'Assistant'}
                        </div>
                        <div>{message.content}</div>
                        <div className="text-xs opacity-70 mt-1">
                          {message.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                      
                      {/* Show task-agent pairs for assistant messages */}
                      {message.type === 'assistant' && message.metadata?.task_agent_pairs && (
                        <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                          <h5 className="font-medium text-blue-800 mb-2">Workflow Tasks:</h5>
                          <div className="space-y-2">
                            {message.metadata.task_agent_pairs.map((pair, index) => (
                              <div key={`${message.id}-task-${index}`} className="flex items-center justify-between p-2 bg-white rounded border">
                                <span className="text-sm text-gray-900">
                                  {pair.task_name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </span>
                                <Badge variant="outline">{pair.primary.name}</Badge>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* User Input for Interactive Questions */}
          {waitingForUser && currentQuestion && (
            <div className="space-y-3 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-center space-x-2">
                <MessageCircle className="w-4 h-4 text-yellow-600" />
                <span className="font-medium text-yellow-800">System needs your input</span>
              </div>
              <p className="text-sm text-yellow-700">{currentQuestion}</p>
              <div className="flex space-x-2">
                <input
                  id="userResponseInput"
                  type="text"
                  placeholder="Your response..."
                  className="flex-1 p-2 border border-yellow-300 rounded focus:ring-2 focus:ring-yellow-500 focus:border-transparent"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleUserResponse();
                    }
                  }}
                />
                <Button onClick={handleUserResponse} size="sm">
                  Send
                </Button>
              </div>
            </div>
          )}

          {/* Control Buttons */}
          {isStarted && (
            <div className="flex space-x-2 pt-4 border-t">
              <Button 
                onClick={handleReset}
                variant="outline"
                size="sm"
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                Reset
              </Button>
              <Button 
                onClick={disconnect}
                variant="outline"
                size="sm"
                disabled={!isConnected}
              >
                <Pause className="w-4 h-4 mr-2" />
                Disconnect
              </Button>
            </div>
          )}

          {/* Connection Status */}
          <div className="text-xs text-gray-500 pt-2 border-t">
            Status: {isConnected ? 'Connected to WebSocket' : 'Not connected'} | 
            {currentThreadId && ` Thread: ${currentThreadId.slice(0, 8)}...`}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
