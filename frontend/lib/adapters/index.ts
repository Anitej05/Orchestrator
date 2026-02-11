/**
 * Adapters Index
 * 
 * Central export point for all application adapters
 * Use these abstractions instead of importing third-party libraries directly
 */

// Document Viewer
export { 
  DocumentViewerAdapter,
  getDocumentViewerAdapter,
  isFileTypeSupported,
  getAdapterInfo as getDocumentViewerInfo
} from './document-viewer'

export type {
  ViewerDocument,
  DocumentViewerConfig,
  IDocumentViewerAdapter
} from './document-viewer/types'

export {
  isViewableDocument,
  isImageFile,
  getFileType
} from './document-viewer/types'

// State Management
export {
  createStore,
  getStateAdapterInfo
} from './state-management'

export type {
  StoreConfig,
  Store,
  UseStore,
  StoreAdapter
} from './state-management/types'
