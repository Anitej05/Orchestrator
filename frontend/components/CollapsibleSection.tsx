// components/CollapsibleSection.tsx
"use client"

import { useState } from "react"
import { ChevronDown, ChevronRight } from "lucide-react"

interface CollapsibleSectionProps {
  title: string
  count: number
  children: React.ReactNode
}

export default function CollapsibleSection({ title, count, children }: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div>
      <div
        className="flex justify-between items-center text-sm cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="text-gray-600 flex items-center">
          {isOpen ? <ChevronDown className="w-4 h-4 mr-2" /> : <ChevronRight className="w-4 h-4 mr-2" />}
          {title}
        </span>
        <span className="font-semibold">{count}</span>
      </div>
      {isOpen && <div className="mt-2 pl-6">{children}</div>}
    </div>
  )
}