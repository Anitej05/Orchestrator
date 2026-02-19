/**
 * Cyntler Document Viewer Adapter
 * 
 * Implementation using @cyntler/react-doc-viewer
 * This is the primary, feature-rich document viewer
 */

'use client'

import React from 'react'
import type { 
  IDocumentViewerAdapter, 
  DocumentViewerConfig 
} from './types'

/**
 * Lazy load the actual library to reduce initial bundle size
 */
const DocViewerComponent = React.lazy(() => 
  import('@cyntler/react-doc-viewer').then(module => ({
    default: module.default
  }))
)

const getDocViewerRenderers = () => {
  // Import renderers separately to avoid type issues
  return import('@cyntler/react-doc-viewer').then(m => m.DocViewerRenderers)
}

/**
 * Cyntler adapter implementation
 */
export class CyntlerDocumentViewerAdapter implements IDocumentViewerAdapter {
  name = '@cyntler/react-doc-viewer'
  version = '1.17.1'

  supportsFileType(fileType: string): boolean {
    const supported = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.ms-powerpoint',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      'text/plain',
      'text/csv',
      'image/jpeg',
      'image/png',
      'image/gif'
    ]
    return supported.includes(fileType)
  }

  render(config: DocumentViewerConfig): React.JSX.Element {
    const [renderers, setRenderers] = React.useState<any>(null)
    
    React.useEffect(() => {
      getDocViewerRenderers().then(setRenderers)
    }, [])
    
    return (
      <React.Suspense fallback={<div className="flex items-center justify-center h-full">Loading viewer...</div>}>
        <DocViewerComponent
          documents={config.documents}
          prefetchMethod={config.prefetchMethod || 'GET'}
          style={config.style}
          className={config.className}
          config={{
            header: {
              disableHeader: config.disableHeader ?? true,
              disableFileName: config.disableFileName ?? false
            }
          }}
          pluginRenderers={renderers}
        />
      </React.Suspense>
    )
  }
}
