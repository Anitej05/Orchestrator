// components/conversations-dropdown.tsx
'use client'

import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Plus } from 'lucide-react';

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
  const [isOpen] = useState(true);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [authRetryCount, setAuthRetryCount] = useState(0);
  const pageSize = 9;
  const listRef = useRef<HTMLDivElement | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  const hasClerkSession = () => {
    if (typeof window === 'undefined') return false;
    const anyWin: any = window as any;
    return Boolean(anyWin?.Clerk?.session);
  };

  useEffect(() => {
    const loadConversations = async (nextOffset: number, append: boolean) => {
      if (!hasClerkSession()) {
        if (authRetryCount < 6) {
          setAuthRetryCount((count) => count + 1);
          if (!append) {
            setLoading(false);
          }
          setTimeout(() => loadConversations(nextOffset, append), 400);
          return;
        }
        if (!append) {
          setError('Unable to connect to authentication session. Please sign in.');
        }
        return;
      }

      if (append) {
        setLoadingMore(true);
      } else {
        setLoading(true);
        setError(null);
      }

      try {
        const { authFetch } = await import('@/lib/auth-fetch');
        const response = await authFetch(
          `http://localhost:8000/api/conversations?limit=${pageSize}&offset=${nextOffset}`
        );

        if (!response.ok) {
          if (response.status === 401 && !hasClerkSession()) {
            if (authRetryCount < 6) {
              setAuthRetryCount((count) => count + 1);
              if (!append) {
                setLoading(false);
              }
              setTimeout(() => loadConversations(nextOffset, append), 400);
              return;
            }
          }
          if (response.status === 404) {
            console.warn('Conversations endpoint not found - this might be expected if backend is not fully set up');
            setConversations([]);
            setHasMore(false);
            return;
          }
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const payload = await response.json();
        const items = Array.isArray(payload) ? payload : payload.items || [];
        const total = Array.isArray(payload) ? undefined : payload.total;

        const conversationDetails: ConversationItem[] = items
          .filter((conv: any) => conv && (conv.id || conv.thread_id))
          .map((conv: any) => ({
            thread_id: conv.id || conv.thread_id || '',
            created_at: conv.created_at,
            title: conv.title || 'Untitled',
            preview: conv.last_message || conv.title || 'No preview available'
          }));

        if (append) {
          setConversations((prev) => [...prev, ...conversationDetails]);
        } else {
          setConversations(conversationDetails);
        }

        const newOffset = nextOffset + conversationDetails.length;
        setOffset(newOffset);
        if (typeof total === 'number') {
          setHasMore(newOffset < total);
        } else {
          setHasMore(conversationDetails.length === pageSize);
        }
      } catch (err) {
        console.error('Failed to load conversations:', err);

        if (!append) {
          if (err instanceof Error && (err.message.includes('fetch') || err.message.includes('Network') || err.message.includes('Failed to fetch'))) {
            console.warn('Backend server appears to be offline. Conversations will be empty.');
            setError('Unable to connect to server. Please check if the backend is running.');
          } else {
            setError('Failed to load conversations');
          }
          setConversations([]);
        }
      } finally {
        setLoading(false);
        setLoadingMore(false);
      }
    };

    if (isOpen) {
      setAuthRetryCount(0);
      loadConversations(0, false);
    }
  }, [isOpen, pageSize]);

  useEffect(() => {
    if (!sentinelRef.current) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry.isIntersecting && hasMore && !loadingMore && !loading) {
          const nextOffset = offset;
          if (nextOffset === conversations.length) {
            loadNextPage(nextOffset);
          }
        }
      },
      { root: listRef.current, rootMargin: '80px' }
    );

    observer.observe(sentinelRef.current);
    return () => observer.disconnect();
  }, [conversations.length, hasMore, loadingMore, loading, offset]);

  const handleConversationClick = (threadId: string) => {
    onConversationSelect(threadId);
  };

  const handleNewConversation = () => {
    if (onNewConversation) {
      onNewConversation();
    }
  };

  const handleListScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    const nearBottom = target.scrollTop + target.clientHeight >= target.scrollHeight - 32;
    if (nearBottom && hasMore && !loadingMore && !loading) {
      const nextOffset = offset;
      if (nextOffset === conversations.length) {
        loadNextPage(nextOffset);
      }
    }
  };

  const loadNextPage = async (nextOffset: number) => {
    if (!hasClerkSession()) {
      if (authRetryCount < 6) {
        setAuthRetryCount((count) => count + 1);
        setTimeout(() => loadNextPage(nextOffset), 400);
      }
      return;
    }
    setLoadingMore(true);
    try {
      const { authFetch } = await import('@/lib/auth-fetch');
      const response = await authFetch(
        `http://localhost:8000/api/conversations?limit=${pageSize}&offset=${nextOffset}`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const payload = await response.json();
      const items = Array.isArray(payload) ? payload : payload.items || [];
      const total = Array.isArray(payload) ? undefined : payload.total;

      const conversationDetails: ConversationItem[] = items
        .filter((conv: any) => conv && (conv.id || conv.thread_id))
        .map((conv: any) => ({
          thread_id: conv.id || conv.thread_id || '',
          created_at: conv.created_at,
          title: conv.title || 'Untitled',
          preview: conv.last_message || conv.title || 'No preview available'
        }));

      setConversations((prev) => {
        const merged = [...prev, ...conversationDetails].filter(
          (conv) => conv && conv.thread_id
        );
        const uniqueMap = new Map<string, ConversationItem>();
        for (const conv of merged) {
          if (!uniqueMap.has(conv.thread_id)) {
            uniqueMap.set(conv.thread_id, conv);
          }
        }
        return Array.from(uniqueMap.values());
      });
      const newOffset = nextOffset + conversationDetails.length;
      setOffset(newOffset);
      if (typeof total === 'number') {
        setHasMore(newOffset < total);
      } else {
        setHasMore(conversationDetails.length === pageSize);
      }
    } catch (err) {
      console.error('Failed to load more conversations:', err);
    } finally {
      setLoadingMore(false);
    }
  };

  return (
    <div className="w-full h-full flex flex-col space-y-2">
      {/* New Conversation Button */}
      {onNewConversation && (
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start border-dashed hover:bg-gray-50 dark:hover:bg-gray-800"
          onClick={() => {
            onNewConversation();
          }}
        >
          <Plus className="w-4 h-4 mr-2" />
          <span className="text-sm">New Conversation</span>
        </Button>
      )}

      <div className="flex-1 overflow-y-auto pr-1" onScroll={handleListScroll} ref={listRef}>
        {loading ? (
          <div className="text-sm text-gray-500 dark:text-gray-400 px-2 py-2">Loading conversations...</div>
        ) : error ? (
          <div className="text-sm text-red-500 dark:text-red-400 px-2 py-2">
            {error}
          </div>
        ) : conversations.length === 0 ? (
          <div className="text-sm text-gray-500 dark:text-gray-400 px-2 py-2">No conversations yet</div>
        ) : (
          conversations
            .filter((conversation) => conversation.thread_id)
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
                    {new Date(conversation.created_at || Date.now()).toLocaleDateString()} {' '}
                    {new Date(conversation.created_at || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </Button>
            ))
        )}
        <div ref={sentinelRef} className="h-1" />
        {loadingMore && (
          <div className="text-xs text-text-tertiary px-2 py-2">
            Loading more…
          </div>
        )}
      </div>
    </div>
  );
}

