'use client'

import { useEffect, useState } from 'react'
import { UserButton } from '@clerk/nextjs'

export function UserButtonWrapper() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return <div className="w-8 h-8" /> // Placeholder to maintain layout
  }

  return <UserButton afterSignOutUrl="/sign-in" />
}
