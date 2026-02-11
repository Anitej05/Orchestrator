/**
 * Document Viewer Adapter Factory
 * 
 * Central point for accessing document viewer functionality
 * Handles adapter selection, fallbacks, and error recovery
 */

'use client'

import React, { useState, useEffect } from 'react'
import type { 
  IDocumentViewerAdapter, 
  DocumentViewerConfig 
} from './types'
import { CyntlerDocumentViewerAdapter } from './cyntler-adapter'
import { FallbackDocumentViewerAdapter } from './fallback-adapter'
import { isFeatureEnabled } from '@/lib/feature-flags'

/**
 * Singleton adapter instances
 */
let primaryAdapter: IDocumentViewerAdapter | null = null
let fallbackAdapter: IDocumentViewerAdapter | null = null

/**
 * Get the primary document viewer adapter
 */
function getPrimaryAdapter(): IDocumentViewerAdapter {
  if (!primaryAdapter) {
    if (isFeatureEnabled('use_new_document_viewer')) {
      primaryAdapter = new CyntlerDocumentViewerAdapter()
    } else {
      // If feature flag is off, use fallback
      primaryAdapter = new FallbackDocumentViewerAdapter()
    }
  }
  return primaryAdapter
}

/**
 * Get the fallback document viewer adapter
 */
function getFallbackAdapter(): IDocumentViewerAdapter {
  if (!fallbackAdapter) {
    fallbackAdapter = new FallbackDocumentViewerAdapter()
  }
  return fallbackAdapter
}

/**
 * Unified Document Viewer Component
 * 
 * Automatically selects the best adapter and handles fallbacks
 * 
 * @example
 * <DocumentViewerAdapter
 *   documents={[{ uri: '/path/to/doc.pdf', fileName: 'doc.pdf' }]}
 * />
 */
export function DocumentViewerAdapter(config: DocumentViewerConfig): React.JSX.Element {
  const [adapter, setAdapter] = useState<IDocumentViewerAdapter>(getPrimaryAdapter)
  const [error, setError] = useState<Error | null>(null)
  const [useFallback, setUseFallback] = useState(false)

  useEffect(() => {
    // Reset error and fallback on config change
    setError(null)
    setUseFallback(false)
  }, [config.documents])

  // Error boundary for adapter failures
  const handleError = (err: Error) => {
    console.error('Document viewer error:', err)
    setError(err)
    
    // Switch to fallback if primary fails
    if (!useFallback) {
      console.log('Switching to fallback document viewer')
      setAdapter(getFallbackAdapter())
      setUseFallback(true)
    }
  }

  try {
    return (
      <ErrorBoundary onError={handleError}>
        {adapter.render(config)}
      </ErrorBoundary>
    )
  } catch (err) {
    handleError(err as Error)
    return getFallbackAdapter().render(config)
  }
}

/**
 * Error Boundary for catching render errors
 */
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; onError: (error: Error) => void },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error: Error) {
    this.props.onError(error)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-full text-text-tertiary">
          <div className="text-center">
            <p className="text-lg font-semibold">Failed to load document</p>
            <p className="text-sm mt-2">Please try again or contact support</p>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * Export adapter for advanced use cases
 */
export function getDocumentViewerAdapter(): IDocumentViewerAdapter {
  return getPrimaryAdapter()
}

/**
 * Check if a file type is supported by the current adapter
 */
export function isFileTypeSupported(fileType: string): boolean {
  return getPrimaryAdapter().supportsFileType(fileType)
}

/**
 * Get adapter information (useful for debugging)
 */
export function getAdapterInfo(): { name: string; version: string } {
  const adapter = getPrimaryAdapter()
  return {
    name: adapter.name,
    version: adapter.version
  }
}
