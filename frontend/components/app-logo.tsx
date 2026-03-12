"use client"

import { BookOpen } from "lucide-react"

interface AppLogoProps {
  slug: string
  name: string
  className?: string
}

export function AppLogo({ slug, name, className = "" }: AppLogoProps) {
  // Get initials for display
  const initials = name
    .split(' ')
    .map(word => word[0])
    .slice(0, 2)
    .join('')
    .toUpperCase()

  return (
    <div className={`w-full flex justify-center ${className}`}>
      <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-brand-teal/20 to-brand-teal/10 dark:from-brand-teal/30 dark:to-brand-teal/10 flex items-center justify-center p-4 shadow-sm border-2 border-brand-teal/30">
        <span className="text-xl font-bold text-brand-teal">{initials}</span>
      </div>
    </div>
  )
}
