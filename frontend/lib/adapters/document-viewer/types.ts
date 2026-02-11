/**
 * Document Viewer Adapter
 * 
 * Abstraction layer for document viewing libraries
 * Allows switching between different document viewer implementations without changing application code
 * 
 * Current implementation: @cyntler/react-doc-viewer
 * Fallback: Basic iframe/img viewer
 */

import type React from 'react'

/**
 * Unified document interface
 * All document viewer implementations must conform to this interface
 */
export interface ViewerDocument {
  uri: string
  fileName?: string
  fileType?: string
}

/**
 * Configuration for document viewer
 */
export interface DocumentViewerConfig {
  documents: ViewerDocument[]
  prefetchMethod?: 'GET' | 'POST'
  style?: React.CSSProperties
  className?: string
  onDocumentLoadSuccess?: (document: ViewerDocument) => void
  onDocumentLoadFail?: (error: Error, document: ViewerDocument) => void
  disableHeader?: boolean
  disableFileName?: boolean
}

/**
 * Document viewer adapter interface
 * All implementations must conform to this interface
 */
export interface IDocumentViewerAdapter {
  render(config: DocumentViewerConfig): React.JSX.Element
  supportsFileType(fileType: string): boolean
  name: string
  version: string
}

/**
 * Check if a file type is viewable
 */
export function isViewableDocument(fileName: string): boolean {
  const lower = fileName.toLowerCase()
  return (
    lower.endsWith('.pdf') ||
    lower.endsWith('.doc') ||
    lower.endsWith('.docx') ||
    lower.endsWith('.xls') ||
    lower.endsWith('.xlsx') ||
    lower.endsWith('.ppt') ||
    lower.endsWith('.pptx') ||
    lower.endsWith('.txt') ||
    lower.endsWith('.csv')
  )
}

/**
 * Check if a file is an image
 */
export function isImageFile(fileName: string, fileType?: string): boolean {
  if (fileType?.startsWith('image/')) return true
  
  const lower = fileName.toLowerCase()
  return (
    lower.endsWith('.jpg') ||
    lower.endsWith('.jpeg') ||
    lower.endsWith('.png') ||
    lower.endsWith('.gif') ||
    lower.endsWith('.webp') ||
    lower.endsWith('.svg')
  )
}

/**
 * Get file type from file name
 */
export function getFileType(fileName: string): string {
  const ext = fileName.toLowerCase().split('.').pop() || ''
  
  const typeMap: Record<string, string> = {
    pdf: 'application/pdf',
    doc: 'application/msword',
    docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    xls: 'application/vnd.ms-excel',
    xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ppt: 'application/vnd.ms-powerpoint',
    pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    txt: 'text/plain',
    csv: 'text/csv',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    png: 'image/png',
    gif: 'image/gif',
    webp: 'image/webp',
    svg: 'image/svg+xml'
  }
  
  return typeMap[ext] || 'application/octet-stream'
}
