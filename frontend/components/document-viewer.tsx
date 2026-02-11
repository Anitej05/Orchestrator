"use client"

import React from "react"
import DocViewer, { DocViewerRenderers } from "@cyntler/react-doc-viewer"
import { Button } from "@/components/ui/button"
import { ArrowLeft, FileText } from "lucide-react"
import { API_BASE_URL } from "@/lib/config"

interface DocumentViewerProps {
  file: {
    name: string
    type: string
    content?: string
    file_path?: string
  }
  onBack: () => void
}

export default function DocumentViewer({ file, onBack }: DocumentViewerProps) {
  // Determine the URI for the document viewer
  const getDocumentUri = () => {
    // If content is a URL (starts with http), use it directly
    if (file.content && (file.content.startsWith('http://') || file.content.startsWith('https://'))) {
      return file.content
    }
    
    // If we have a file_path, construct backend API URL
    if (file.file_path) {
      return `${API_BASE_URL}/api/files/${encodeURIComponent(file.file_path)}`
    }
    
    // If content looks like an API path, use it
    if (file.content && file.content.startsWith(API_BASE_URL)) {
      return file.content
    }
    
    // Fallback: return content as-is (might be a relative URL)
    return file.content || ''
  }

  const documentUri = getDocumentUri()
  
  // Check if this is a viewable document type
  const isViewableDocument = () => {
    const fileName = file.name.toLowerCase()
    return (
      fileName.endsWith('.pdf') ||
      fileName.endsWith('.doc') ||
      fileName.endsWith('.docx') ||
      fileName.endsWith('.xls') ||
      fileName.endsWith('.xlsx') ||
      fileName.endsWith('.ppt') ||
      fileName.endsWith('.pptx') ||
      fileName.endsWith('.txt') ||
      fileName.endsWith('.csv')
    )
  }

  // Check if this is an image
  const isImage = () => {
    const fileName = file.name.toLowerCase()
    return (
      file.type.startsWith('image/') ||
      fileName.endsWith('.jpg') ||
      fileName.endsWith('.jpeg') ||
      fileName.endsWith('.png') ||
      fileName.endsWith('.gif') ||
      fileName.endsWith('.webp') ||
      fileName.endsWith('.svg')
    )
  }

  const documents = [
    {
      uri: documentUri,
      fileName: file.name,
    },
  ]

  return (
    <div className="h-full flex flex-col">
      {/* Header with back button */}
      <div className="bg-bg-card border-b border-border-color px-4 py-3 shadow-sm">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="hover:bg-bg-subtle"
          >
            <ArrowLeft className="w-4 h-4 mr-1" />
            Back
          </Button>
          <div className="flex items-center gap-2 flex-1">
            <FileText className="w-4 h-4 text-text-secondary" />
            <span className="text-sm font-medium text-text-primary truncate">
              {file.name}
            </span>
          </div>
        </div>
      </div>

      {/* Document viewer */}
      <div className="flex-1 overflow-hidden bg-bg-subtle">
        {isImage() ? (
          // For images, show directly
          <div className="h-full flex items-center justify-center p-4">
            <img
              src={documentUri}
              alt={file.name}
              className="max-w-full max-h-full object-contain"
            />
          </div>
        ) : isViewableDocument() ? (
          // For documents, use DocViewer
          <DocViewer
            documents={documents}
            pluginRenderers={DocViewerRenderers}
            prefetchMethod="GET"
            config={{
              header: {
                disableHeader: true,
              },
            }}
            style={{ height: "100%" }}
          />
        ) : (
          // Unsupported file type
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-text-tertiary">
              <FileText className="w-16 h-16 mx-auto mb-4 text-text-disabled" />
              <p className="text-lg font-semibold">Preview not available</p>
              <p className="text-sm mt-2">
                This file type cannot be previewed in the browser.
              </p>
              <p className="text-xs mt-1 text-text-disabled">{file.type}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
