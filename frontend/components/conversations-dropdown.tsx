// components/conversations-dropdown.tsx
'use client'

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, MessageSquare } from 'lucide-react';

interface ConversationsDropdownProps {
  onConversationSelect: (threadId: string) => void;
  currentThreadId?: string;
}

interface ConversationItem {
  thread_id: string;
  created_at?: string;
}

export default function ConversationsDropdown({
  onConversationSelect,
  currentThreadId
}: ConversationsDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadConversations = async () => {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8000/api/conversations');
        if (!response.ok) {
          throw new Error('Failed to load conversations');
        }
        const conversationIds: string[] = await response.json();

        // Convert to conversation items with thread_id
        const conversationItems = conversationIds.map(thread_id => ({
          thread_id,
          created_at: new Date().toISOString() // Placeholder - could add created_at from somewhere
        }));

        setConversations(conversationItems);
      } catch (error) {
        console.error('Failed to load conversations:', error);
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
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <Button
          variant="ghost"
          className="w-full justify-between hover:bg-gray-100"
        >
          <div className="flex items-center space-x-2">
            <MessageSquare className="w-4 h-4" />
            <span>Conversations</span>
          </div>
          <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'transform rotate-180' : ''}`} />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="pl-4 space-y-1">
        {loading ? (
          <div className="text-sm text-gray-500 px-3 py-2">Loading conversations...</div>
        ) : conversations.length === 0 ? (
          <div className="text-sm text-gray-500 px-3 py-2">No conversations yet</div>
        ) : (
          conversations.map((conversation) => (
            <Button
              key={conversation.thread_id}
              variant={currentThreadId === conversation.thread_id ? "secondary" : "ghost"}
              size="sm"
              className="w-full justify-start text-left truncate"
              onClick={() => handleConversationClick(conversation.thread_id)}
            >
              <div className="truncate">
                {conversation.thread_id}
              </div>
            </Button>
          ))
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
