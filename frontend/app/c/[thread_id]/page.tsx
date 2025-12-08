'use client'

import { useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useConversationStore } from '@/lib/conversation-store'
import { useToast } from '@/hooks/use-toast'

export default function ConversationPage() {
  const params = useParams()
  const router = useRouter()
  const threadId = params?.thread_id as string
  const { toast } = useToast()

  useEffect(() => {
    if (!threadId) return

    // Load conversation into store
    console.log('Loading conversation from URL:', threadId)
    const { loadConversation } = useConversationStore.getState().actions
    
    loadConversation(threadId)
      .then(() => {
        // Redirect to home page - conversation will be loaded in the store
        // This gives us the ChatGPT-like experience where content updates without full reload
        router.replace('/')
      })
      .catch((error) => {
        console.error('Failed to load conversation:', error)
        toast({
          title: "Error",
          description: "Failed to load conversation. Please try again.",
          variant: "destructive"
        })
        // Redirect to home on error
        setTimeout(() => router.replace('/'), 1000)
      })
  }, [threadId, router, toast])

  // Show minimal loading state during redirect
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-white"></div>
    </div>
  )
}
