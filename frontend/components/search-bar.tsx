"use client"

import { useEffect, useState } from "react"

interface SearchBarProps {
  onSearchChange: (query: string) => void
}

export function SearchBar({ onSearchChange }: SearchBarProps) {
  const [query, setQuery] = useState("")

  useEffect(() => {
    onSearchChange(query)
  }, [query, onSearchChange])

  return (
    <div className="relative w-96" suppressHydrationWarning>
      <input
        type="text"
        placeholder="Search apps..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full px-4 py-2.5 pl-10 rounded-lg border border-border-color bg-bg-page text-text-primary placeholder-text-tertiary focus:outline-none focus:ring-2 focus:ring-brand-teal/50 focus:border-transparent"
        suppressHydrationWarning
      />
      <svg
        className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-tertiary"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>
      {query && (
        <button
          onClick={() => setQuery("")}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-text-tertiary hover:text-text-primary"
          suppressHydrationWarning
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  )
}
