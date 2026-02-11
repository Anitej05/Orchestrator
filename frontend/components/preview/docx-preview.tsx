"use client"

import React, { useState, useEffect } from "react"
import mammoth from "mammoth"
import { FileText } from "lucide-react"

interface DocxPreviewProps {
  fileUrl: string
}

export default function DocxPreview({ fileUrl }: DocxPreviewProps) {
  const [docxHtml, setDocxHtml] = useState<string>("")
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string>("")

  useEffect(() => {
    const convertDocx = async () => {
      try {
        const response = await fetch(fileUrl)
        if (!response.ok) throw new Error("Failed to fetch document")
        
        const arrayBuffer = await response.arrayBuffer()
        const result = await mammoth.convertToHtml({ arrayBuffer })
        
        setDocxHtml(result.value)
      } catch (err) {
        console.error("Error converting docx:", err)
        setError("Failed to load document. Please try downloading it instead.")
      } finally {
        setIsLoading(false)
      }
    }

    convertDocx()
  }, [fileUrl])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-text-secondary">Converting document...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-text-tertiary">
          <FileText className="w-16 h-16 mx-auto mb-4 text-red-400" />
          <p className="text-lg font-semibold text-red-600">Error Loading Document</p>
          <p className="text-sm mt-2">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto p-6 bg-white">
      <div 
        className="prose prose-sm max-w-none w-full break-words prose-img:max-w-full prose-table:table-fixed prose-table:w-full prose-td:break-words prose-th:break-words prose-pre:whitespace-pre-wrap prose-pre:break-words"
        dangerouslySetInnerHTML={{ __html: docxHtml }}
      />
    </div>
  )
}
