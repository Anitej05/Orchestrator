'use client'

import { useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'

export default function ConversationPage() {
  const params = useParams()
  const router = useRouter()
  const threadId = params?.thread_id as string

  useEffect(() => {
    if (threadId) {
      // Redirect to home page with threadId as query parameter
      // This allows shareable /c/{thread_id} URLs while keeping all logic in one place
      router.replace(`/?threadId=${threadId}`)
    }
  }, [threadId, router])

  // Show minimal loading state during redirect
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-white"></div>
    </div>
  )
}
