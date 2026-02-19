/**
 * Fallback Document Viewer Adapter
 * 
 * Simple, lightweight fallback for when the primary viewer fails
 * Uses native browser capabilities (iframe, img tags)
 */

'use client'

import React from 'react'
import type { 
  IDocumentViewerAdapter, 
  DocumentViewerConfig 
} from './types'
import { FileText, AlertCircle } from 'lucide-react'

/**
 * Fallback adapter implementation
 * Uses basic HTML elements for document rendering
 */
export class FallbackDocumentViewerAdapter implements IDocumentViewerAdapter {
  name = 'Native Browser Viewer'
  version = '1.0.0'

  supportsFileType(fileType: string): boolean {
    // Supports PDFs via iframe and images
    return (
      fileType === 'application/pdf' ||
      fileType.startsWith('image/') ||
      fileType === 'text/plain'
    )
  }

  render(config: DocumentViewerConfig): React.JSX.Element {
    const document = config.documents[0] // Only handle first document in fallback
    
    if (!document) {
      return (
        <div className="flex items-center justify-center h-full text-text-tertiary">
          <div className="text-center">
            <FileText className="w-16 h-16 mx-auto mb-4" />
            <p>No document to display</p>
          </div>
        </div>
      )
    }

    const fileType = document.fileType || this.guessFileType(document.fileName || '')

    // Handle images
    if (fileType.startsWith('image/')) {
      return (
        <div className="flex items-center justify-center h-full p-4 bg-bg-subtle">
          <img
            src={document.uri}
            alt={document.fileName || 'Document'}
            className="max-w-full max-h-full object-contain"
            style={config.style}
          />
        </div>
      )
    }

    // Handle PDFs
    if (fileType === 'application/pdf') {
      return (
        <iframe
          src={document.uri}
          className={config.className || 'w-full h-full border-0'}
          style={config.style}
          title={document.fileName || 'Document'}
        />
      )
    }

    // Handle text files
    if (fileType === 'text/plain') {
      return (
        <iframe
          src={document.uri}
          className={config.className || 'w-full h-full border-0'}
          style={config.style}
          title={document.fileName || 'Document'}
        />
      )
    }

    // Unsupported type
    return (
      <div className="flex items-center justify-center h-full text-text-tertiary">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 mx-auto mb-4 text-status-error" />
          <p className="text-lg font-semibold">Preview not available</p>
          <p className="text-sm mt-2">
            This file type cannot be previewed in the browser.
          </p>
          <p className="text-xs mt-1 text-text-disabled">{fileType}</p>
          {document.fileName && (
            <p className="text-xs mt-1 text-text-disabled">{document.fileName}</p>
          )}
        </div>
      </div>
    )
  }

  private guessFileType(fileName: string): string {
    const ext = fileName.toLowerCase().split('.').pop()
    const typeMap: Record<string, string> = {
      pdf: 'application/pdf',
      jpg: 'image/jpeg',
      jpeg: 'image/jpeg',
      png: 'image/png',
      gif: 'image/gif',
      txt: 'text/plain'
    }
    return typeMap[ext || ''] || 'application/octet-stream'
  }
}
