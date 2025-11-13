// components/conversations-dropdown.tsx
'use client'

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, MessageSquare, Plus } from 'lucide-react';

interface ConversationsDropdownProps {
  onConversationSelect: (threadId: string) => void;
  onNewConversation?: () => void;
  currentThreadId?: string;
}

interface ConversationItem {
  thread_id: string;
  created_at?: string;
  title?: string;
  preview?: string;
}

export default function ConversationsDropdown({
  onConversationSelect,
  onNewConversation,
  currentThreadId
}: ConversationsDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadConversations = async () => {
      setLoading(true);
      try {
        const { authFetch } = await import('@/lib/auth-fetch');
        const response = await authFetch('http://localhost:8000/api/conversations');
        if (!response.ok) {
          throw new Error('Failed to load conversations');
        }
        
        // The new endpoint returns full conversation objects with metadata
        const conversationsData = await response.json();
        
        console.log('Loaded conversations:', conversationsData.length);
        
        // Map to the expected format and filter out invalid entries
        const conversationDetails: ConversationItem[] = conversationsData
          .filter((conv: any) => conv && (conv.id || conv.thread_id)) // Only include conversations with valid IDs
          .map((conv: any) => ({
            thread_id: conv.id || conv.thread_id || '',
            created_at: conv.created_at,
            title: conv.title || 'Untitled',
            preview: conv.last_message || conv.title || 'No preview available'
          }));

        // Sort by created_at (newest first)
        conversationDetails.sort((a, b) => 
          new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime()
        );

        setConversations(conversationDetails);
      } catch (err) {
        console.error('Failed to load conversations:', err);
      } finally {
        setLoading(false);
      }
    };

    if (isOpen) {
      loadConversations();
    }
  }, [isOpen]);

  const handleConversationClick = (threadId: string) => {
    onConversationSelect(threadId);
    setIsOpen(false);
  };

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="w-full">
      <CollapsibleTrigger asChild>
        <Button
          variant="ghost"
          className="w-full justify-between hover:bg-gray-100 dark:hover:bg-gray-800"
        >
          <div className="flex items-center space-x-2">
            <MessageSquare className="w-4 h-4" />
            <span className="text-sm font-medium">Conversations</span>
          </div>
          <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'transform rotate-180' : ''}`} />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 space-y-1">
        {/* New Conversation Button */}
        {onNewConversation && (
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start mb-2 border-dashed hover:bg-gray-50 dark:hover:bg-gray-800"
            onClick={() => {
              onNewConversation();
              setIsOpen(false);
            }}
          >
            <Plus className="w-4 h-4 mr-2" />
            <span className="text-sm">New Conversation</span>
          </Button>
        )}
        
        <div className="max-h-[400px] overflow-y-auto pr-1">
          {loading ? (
            <div className="text-sm text-gray-500 dark:text-gray-400 px-2 py-2">Loading conversations...</div>
          ) : conversations.length === 0 ? (
            <div className="text-sm text-gray-500 dark:text-gray-400 px-2 py-2">No conversations yet</div>
          ) : (
            conversations
              .filter((conversation) => conversation.thread_id) // Filter out invalid conversations
              .map((conversation) => (
                <Button
                  key={conversation.thread_id}
                  variant={currentThreadId === conversation.thread_id ? "secondary" : "ghost"}
                  size="sm"
                  className="w-full justify-start text-left h-auto py-2 px-2 mb-1 hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleConversationClick(conversation.thread_id)}
                  title={conversation.preview || conversation.title || 'Untitled conversation'}
                >
                  <div className="flex flex-col w-full min-w-0 overflow-hidden">
                    <div className="font-medium text-sm truncate w-full">
                      {conversation.title || 'Untitled'}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 truncate w-full mt-0.5">
                      {new Date(conversation.created_at || Date.now()).toLocaleDateString()}
                    </div>
                  </div>
                </Button>
              ))
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
