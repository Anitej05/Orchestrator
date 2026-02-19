"use client"

import React, { useState, useEffect } from "react"
import { FileText } from "lucide-react"

interface JsonPreviewProps {
  fileUrl: string
}

export default function JsonPreview({ fileUrl }: JsonPreviewProps) {
  const [jsonData, setJsonData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string>("")

  useEffect(() => {
    const loadJson = async () => {
      try {
        const response = await fetch(fileUrl)
        if (!response.ok) throw new Error("Failed to fetch JSON file")
        
        const text = await response.text()
        const parsed = JSON.parse(text)
        setJsonData(parsed)
      } catch (err) {
        console.error("Error loading JSON:", err)
        setError("Failed to parse JSON file")
      } finally {
        setIsLoading(false)
      }
    }

    loadJson()
  }, [fileUrl])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-text-secondary">Loading JSON...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-text-tertiary">
          <FileText className="w-16 h-16 mx-auto mb-4 text-red-400" />
          <p className="text-lg font-semibold text-red-600">Error Loading JSON</p>
          <p className="text-sm mt-2">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full w-full overflow-auto p-6 bg-gray-900">
      <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap break-words">
        {JSON.stringify(jsonData, null, 2)}
      </pre>
    </div>
  )
}
