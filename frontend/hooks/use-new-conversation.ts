import { useRouter } from 'next/navigation';
import { useConversationStore } from '@/lib/conversation-store';
import { useToast } from '@/hooks/use-toast';

/**
 * Hook to handle new conversation creation from anywhere in the app
 * Resets all state and redirects to orchestrator (home) page
 */
export function useNewConversation() {
  const router = useRouter();
  const { toast } = useToast();

  const startNewConversation = () => {
    console.log('Starting new conversation - clearing all state');

    // Clear localStorage to prevent auto-restoration
    if (typeof window !== 'undefined') {
      localStorage.removeItem('thread_id');
    }

    // Force clear the Zustand store completely
    useConversationStore.setState({
      metadata: {},
      plan: [],
      task_agent_pairs: [],
      messages: [],
      final_response: undefined,
      thread_id: undefined,
      status: 'idle',
      canvas_content: undefined,
      has_canvas: false,
      task_statuses: {},
      current_executing_task: null,
      pending_action_approval: false,
      pending_action: undefined,
      isLoading: false,
    });

    // Navigate to home page
    router.push('/');

    // Show success toast
    toast({
      title: 'New conversation started',
      description: 'Ready to start a new orchestration',
    });
  };

  return { startNewConversation };
}
