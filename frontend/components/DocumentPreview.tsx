/**
 * Document Preview Component
 * Displays created documents with inline canvas preview support
 * Handles:
 * - PDF preview (via iframe)
 * - DOCX/Word preview (via /files endpoint)
 * - Text content preview
 * - File download capability
 */

import React, { useMemo } from 'react';
import { CanvasDisplay } from '@/hooks/useDocumentCreation';

export interface DocumentPreviewProps {
  canvasDisplay: CanvasDisplay;
  previewUrl: string;
  fileName: string;
  onDownload?: () => void;
  className?: string;
}

/**
 * Main Document Preview Component
 * Renders different preview types based on canvas_display.canvas_type
 */
export const DocumentPreview: React.FC<DocumentPreviewProps> = ({
  canvasDisplay,
  previewUrl,
  fileName,
  onDownload,
  className = '',
}) => {
  const canvasType = canvasDisplay.canvas_type;

  return (
    <div className={`border rounded-lg p-6 bg-white shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Document Preview</h3>
        <div className="flex gap-2">
          <a
            href={previewUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Full View
          </a>
          {onDownload && (
            <button
              onClick={onDownload}
              className="px-3 py-1 text-sm bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              Download
            </button>
          )}
        </div>
      </div>

      <div className="border-t pt-4">
        {canvasType === 'pdf' && <PDFPreview previewUrl={previewUrl} fileName={fileName} />}
        {canvasType === 'docx' && <DocxPreview previewUrl={previewUrl} fileName={fileName} />}
        {canvasType === 'text' && <TextPreview canvasDisplay={canvasDisplay} />}
        {canvasType === 'file' && <FilePreview canvasDisplay={canvasDisplay} previewUrl={previewUrl} />}
      </div>
    </div>
  );
};

/**
 * PDF Preview Component
 * Displays PDF inline using iframe
 */
const PDFPreview: React.FC<{ previewUrl: string; fileName: string }> = ({
  previewUrl,
  fileName,
}) => {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-gray-600">PDF: {fileName}</p>
      <iframe
        src={previewUrl}
        title={fileName}
        className="w-full h-96 border rounded"
        style={{ minHeight: '500px' }}
      />
      <p className="text-xs text-gray-500">
        If preview doesn't load, use "Full View" button to open in new tab
      </p>
    </div>
  );
};

/**
 * DOCX Preview Component
 * DOCX files need to be converted to PDF for preview
 * Shows message and preview link
 */
const DocxPreview: React.FC<{ previewUrl: string; fileName: string }> = ({
  previewUrl,
  fileName,
}) => {
  return (
    <div className="flex flex-col gap-3 p-4 bg-gray-50 rounded">
      <p className="text-sm font-medium">Word Document: {fileName}</p>
      <p className="text-sm text-gray-600">
        Preview will be displayed once the document is converted to PDF format.
      </p>
      <a
        href={previewUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-block px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-center"
      >
        Click here to preview the document
      </a>
      <p className="text-xs text-gray-500">
        Or use the "Download" button to save the file to your computer
      </p>
    </div>
  );
};

/**
 * Text Preview Component
 * Displays text content directly in the canvas
 */
const TextPreview: React.FC<{ canvasDisplay: CanvasDisplay }> = ({ canvasDisplay }) => {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm font-medium">{canvasDisplay.file_name}</p>
      <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-96 text-xs font-mono border border-gray-300">
        {canvasDisplay.content || 'No content to display'}
      </pre>
      <p className="text-xs text-gray-500">
        Showing preview (use "Full View" for complete document)
      </p>
    </div>
  );
};

/**
 * Generic File Preview Component
 * For file types without specific preview support
 */
const FilePreview: React.FC<{ canvasDisplay: CanvasDisplay; previewUrl: string }> = ({
  canvasDisplay,
  previewUrl,
}) => {
  return (
    <div className="flex flex-col gap-3 p-4 bg-gray-50 rounded">
      <p className="text-sm font-medium">File Created: {canvasDisplay.file_name}</p>
      <p className="text-sm text-gray-600">
        This file type doesn't have a built-in preview. Use the buttons above to view or download.
      </p>
      <a
        href={previewUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-block px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-center"
      >
        View File
      </a>
    </div>
  );
};

export default DocumentPreview;
