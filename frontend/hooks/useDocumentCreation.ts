/**
 * Hook for document creation and preview
 * Handles:
 * - Creating documents via /api/documents/create
 * - Displaying canvas preview immediately
 * - Serving files via /files endpoint with inline preview
 */

import { useState, useCallback } from 'react';

export interface DocumentCreationRequest {
  content: string;
  file_name: string;
  file_type: 'docx' | 'pdf' | 'txt';
  thread_id?: string;
}

export interface CanvasDisplay {
  canvas_type: 'pdf' | 'docx' | 'text' | 'file';
  file_name: string;
  file_path?: string;
  preview_url?: string;
  content?: string;
  full_preview_url?: string;
  note?: string;
}

export interface DocumentCreationResponse {
  success: boolean;
  message: string;
  file_path: string;
  relative_path: string;
  preview_url: string;
  canvas_display?: CanvasDisplay;
}

export const useDocumentCreation = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdDocument, setCreatedDocument] = useState<DocumentCreationResponse | null>(null);

  const createDocument = useCallback(
    async (request: DocumentCreationRequest, authToken: string) => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch('/api/documents/create', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`,
          },
          body: JSON.stringify(request),
        });

        if (!response.ok) {
          throw new Error(`Failed to create document: ${response.statusText}`);
        }

        const data: DocumentCreationResponse = await response.json();
        setCreatedDocument(data);
        return data;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        setError(errorMsg);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const clearDocument = useCallback(() => {
    setCreatedDocument(null);
    setError(null);
  }, []);

  return {
    loading,
    error,
    createdDocument,
    createDocument,
    clearDocument,
  };
};

export default useDocumentCreation;
