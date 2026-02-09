'use client'

import { useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import { useConversationStore } from "@/lib/conversation-store"
import { useToast } from "@/hooks/use-toast"

const PAGE_BG = "min-h-screen bg-bg-page dark:bg-background text-text-primary"

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
    <div className={`${PAGE_BG} flex items-center justify-center`}>
      <div className="flex flex-col items-center space-y-3">
        <div className="h-12 w-12 rounded-full border-3 border-border-color border-t-brand-teal animate-spin" />
        <p className="text-text-secondary text-sm">Loading conversation…</p>
      </div>
    </div>
  )
}
