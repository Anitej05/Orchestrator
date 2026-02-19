"use client"

import React, { useMemo } from "react"
import dynamic from "next/dynamic"
import { Button } from "@/components/ui/button"
import { ArrowLeft, FileText, Download } from "lucide-react"
import { API_BASE_URL } from "@/lib/config"

// Dynamically import preview components to avoid SSR issues
const PdfPreview = dynamic(() => import("@/components/preview/pdf-preview"), { ssr: false })
const DocxPreview = dynamic(() => import("@/components/preview/docx-preview"), { ssr: false })
const ExcelPreview = dynamic(() => import("@/components/preview/excel-preview"), { ssr: false })
const CsvPreview = dynamic(() => import("@/components/preview/csv-preview"), { ssr: false })
const JsonPreview = dynamic(() => import("@/components/preview/json-preview"), { ssr: false })

interface DocumentViewerProps {
  file: {
    name: string
    type: string
    content?: string
    file_path?: string
  }
  onBack: () => void
}

// File type detection utilities
const getFileExtension = (fileName: string): string => {
  return fileName.toLowerCase().split('.').pop() || ''
}

const getFileType = (file: { name: string; type: string }): string => {
  const ext = getFileExtension(file.name)
  
  // Check by extension first (more reliable than MIME type)
  if (ext === 'pdf') return 'pdf'
  if (ext === 'docx') return 'docx'
  if (ext === 'xlsx' || ext === 'xls') return 'excel'
  if (ext === 'csv') return 'csv'
  if (ext === 'json') return 'json'
  if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'bmp'].includes(ext)) return 'image'
  if (['txt', 'md', 'log'].includes(ext)) return 'text'
  
  // Fallback to MIME type
  if (file.type.startsWith('image/')) return 'image'
  if (file.type === 'application/pdf') return 'pdf'
  if (file.type === 'application/json') return 'json'
  if (file.type === 'text/csv') return 'csv'
  if (file.type.includes('spreadsheet') || file.type.includes('excel')) return 'excel'
  if (file.type.includes('word') || file.type.includes('document')) return 'docx'
  
  // Unsupported Office formats
  if (['doc', 'ppt', 'pptx'].includes(ext)) return 'unsupported-office'
  
  return 'unsupported'
}

export default function DocumentViewer({ file, onBack }: DocumentViewerProps) {
  // Determine the URI for the document viewer
  const documentUri = useMemo(() => {
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
  }, [file.content, file.file_path])

  const fileType = useMemo(() => getFileType(file), [file.name, file.type])
  const showDownloadButton = ['docx', 'excel', 'unsupported-office', 'unsupported'].includes(fileType)

  // Render preview component based on file type
  const renderPreview = () => {
    switch (fileType) {
      case 'pdf':
        return <PdfPreview fileUrl={documentUri} />
      
      case 'docx':
        return <DocxPreview fileUrl={documentUri} />
      
      case 'excel':
        return <ExcelPreview fileUrl={documentUri} />
      
      case 'csv':
        return <CsvPreview fileUrl={documentUri} />
      
      case 'json':
        return <JsonPreview fileUrl={documentUri} />
      
      case 'image':
        return (
          <div className="h-full w-full flex items-center justify-center p-4 overflow-hidden">
            <img
              src={documentUri}
              alt={file.name}
              className="max-w-full max-h-full object-contain"
            />
          </div>
        )
      
      case 'text':
        return (
          <div className="h-full w-full overflow-auto p-6 bg-white">
            <pre className="whitespace-pre-wrap break-words text-sm font-mono">
              {/* Text content will be loaded */}
            </pre>
          </div>
        )
      
      case 'unsupported-office':
        return (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-text-tertiary">
              <FileText className="w-16 h-16 mx-auto mb-4 text-text-disabled" />
              <p className="text-lg font-semibold">Preview not available</p>
              <p className="text-sm mt-2">
                This Office file format cannot be previewed in the browser.
              </p>
              <p className="text-sm mt-2">Please use the download button above.</p>
              <p className="text-xs mt-1 text-text-disabled">{file.type}</p>
            </div>
          </div>
        )
      
      default:
        return (
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
        )
    }
  }

  return (
    <div className="h-full w-full max-w-full flex flex-col overflow-hidden min-w-0">
      {/* Header with back button */}
      <div className="bg-bg-card border-b border-border-color px-4 py-3 shadow-sm flex-shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="hover:bg-bg-subtle flex-shrink-0"
          >
            <ArrowLeft className="w-4 h-4 mr-1" />
            Back
          </Button>
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <FileText className="w-4 h-4 text-text-secondary flex-shrink-0" />
            <span className="text-sm font-medium text-text-primary truncate">
              {file.name}
            </span>
          </div>
          {showDownloadButton && (
            <a href={documentUri} download={file.name} className="flex-shrink-0">
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-1" />
                Download
              </Button>
            </a>
          )}
        </div>
      </div>

      {/* Document preview area */}
      <div className="flex-1 overflow-hidden bg-bg-subtle min-w-0 max-w-full">
        {renderPreview()}
      </div>
    </div>
  )
}