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
        className="flex justify-between items-center cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="ui-metadata-label flex items-center">
          {isOpen ? <ChevronDown className="w-4 h-4 mr-2" /> : <ChevronRight className="w-4 h-4 mr-2" />}
          {title}
        </span>
        <span className="ui-metadata-mono">{count}</span>
      </div>
      {isOpen && <div className="mt-2 pl-6">{children}</div>}
    </div>
  )
}
