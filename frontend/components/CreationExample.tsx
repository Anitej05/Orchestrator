/**
 * Complete Example: Document and Spreadsheet Creation with Canvas Preview
 * 
 * This example demonstrates:
 * 1. Creating documents via /api/documents/create
 * 2. Creating spreadsheets via /api/spreadsheets/create
 * 3. Displaying canvas preview immediately after creation
 * 4. Serving files via /files endpoint with inline preview
 * 5. Handling auth tokens and error cases
 */

'use client';

import React, { useState } from 'react';
import useDocumentCreation from '@/hooks/useDocumentCreation';
import DocumentPreview from '@/components/DocumentPreview';
import SpreadsheetPreview from '@/components/SpreadsheetPreview';

export interface CreationExampleProps {
  authToken: string;
  threadId?: string;
  onCreatedDocument?: (filePath: string) => void;
}

/**
 * Example: Document Creation with Preview
 */
export const DocumentCreationExample: React.FC<CreationExampleProps> = ({
  authToken,
  threadId,
  onCreatedDocument,
}) => {
  const { loading, error, createdDocument, createDocument, clearDocument } =
    useDocumentCreation();
  const [content, setContent] = useState(
    '# Sample Report\n\nThis is a test document created via the orchestrator.'
  );
  const [fileName, setFileName] = useState('report.docx');
  const [fileType, setFileType] = useState<'docx' | 'pdf' | 'txt'>('docx');

  const handleCreate = async () => {
    try {
      const response = await createDocument(
        {
          content,
          file_name: fileName,
          file_type: fileType,
          thread_id: threadId,
        },
        authToken
      );

      if (onCreatedDocument && response.relative_path) {
        onCreatedDocument(response.relative_path);
      }
    } catch (err) {
      console.error('Creation error:', err);
    }
  };

  const handleDownload = async () => {
    if (!createdDocument?.preview_url) return;

    // Download via the /files endpoint
    window.location.href = createdDocument.preview_url + '?download=true';
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-6">
      <div className="bg-white border rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-6">Create Document</h2>

        {/* Form */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">File Name</label>
            <input
              type="text"
              value={fileName}
              onChange={(e) => setFileName(e.target.value)}
              className="w-full px-3 py-2 border rounded"
              placeholder="e.g., report.docx"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">File Type</label>
            <select
              value={fileType}
              onChange={(e) => setFileType(e.target.value as 'docx' | 'pdf' | 'txt')}
              className="w-full px-3 py-2 border rounded"
            >
              <option value="docx">Word (.docx)</option>
              <option value="pdf">PDF (.pdf)</option>
              <option value="txt">Text (.txt)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Content</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="w-full px-3 py-2 border rounded h-48 font-mono text-sm"
              placeholder="Enter document content..."
            />
          </div>

          <button
            onClick={handleCreate}
            disabled={loading}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Creating...' : 'Create Document'}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
            <p className="font-medium">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}
      </div>

      {/* Preview */}
      {createdDocument && (
        <div className="space-y-4">
          <div className="p-4 bg-green-100 text-green-700 rounded">
            <p className="font-medium">✓ Document Created Successfully</p>
            <p className="text-sm">{createdDocument.message}</p>
            <p className="text-xs mt-2 font-mono">{createdDocument.relative_path}</p>
          </div>

          {createdDocument.canvas_display && (
            <DocumentPreview
              canvasDisplay={createdDocument.canvas_display}
              previewUrl={createdDocument.preview_url}
              fileName={createdDocument.canvas_display.file_name}
              onDownload={handleDownload}
            />
          )}

          <button
            onClick={clearDocument}
            className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            Create Another
          </button>
        </div>
      )}
    </div>
  );
};

/**
 * Example: Spreadsheet Creation with Preview
 */
export const SpreadsheetCreationExample: React.FC<CreationExampleProps> = ({
  authToken,
  threadId,
  onCreatedDocument,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdSpreadsheet, setCreatedSpreadsheet] = useState<any>(null);
  const [fileName, setFileName] = useState('sales_report.xlsx');
  const [fileFormat, setFileFormat] = useState<'xlsx' | 'csv'>('xlsx');
  const [csvData, setCsvData] = useState(
    'Month,Sales,Profit\nJan,10000,2000\nFeb,12000,2500\nMar,15000,3200'
  );

  const handleCreateSpreadsheet = async () => {
    setLoading(true);
    setError(null);

    try {
      // Parse CSV data
      const lines = csvData.split('\n');
      const headers = lines[0].split(',').map((h) => h.trim());
      const rows = lines.slice(1).map((line) =>
        line.split(',').map((cell) => {
          const trimmed = cell.trim();
          return isNaN(Number(trimmed)) ? trimmed : Number(trimmed);
        })
      );

      const response = await fetch('/api/spreadsheets/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          filename: fileName,
          file_format: fileFormat,
          data: {
            columns: headers,
            rows,
          },
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to create spreadsheet: ${response.statusText}`);
      }

      const data = await response.json();
      setCreatedSpreadsheet(data);

      if (onCreatedDocument && data.relative_path) {
        onCreatedDocument(data.relative_path);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!createdSpreadsheet?.preview_url) return;
    window.location.href = createdSpreadsheet.preview_url + '?download=true';
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-6">
      <div className="bg-white border rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-6">Create Spreadsheet</h2>

        {/* Form */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">File Name</label>
            <input
              type="text"
              value={fileName}
              onChange={(e) => setFileName(e.target.value)}
              className="w-full px-3 py-2 border rounded"
              placeholder="e.g., data.xlsx"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">File Format</label>
            <select
              value={fileFormat}
              onChange={(e) => setFileFormat(e.target.value as 'xlsx' | 'csv')}
              className="w-full px-3 py-2 border rounded"
            >
              <option value="xlsx">Excel (.xlsx)</option>
              <option value="csv">CSV (.csv)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">CSV Data</label>
            <p className="text-xs text-gray-600 mb-2">
              First row = headers, subsequent rows = data (comma-separated)
            </p>
            <textarea
              value={csvData}
              onChange={(e) => setCsvData(e.target.value)}
              className="w-full px-3 py-2 border rounded h-40 font-mono text-sm"
              placeholder="Name,Age,Salary&#10;John,30,50000&#10;Jane,28,55000"
            />
          </div>

          <button
            onClick={handleCreateSpreadsheet}
            disabled={loading}
            className="w-full px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
          >
            {loading ? 'Creating...' : 'Create Spreadsheet'}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
            <p className="font-medium">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}
      </div>

      {/* Preview */}
      {createdSpreadsheet && (
        <div className="space-y-4">
          <div className="p-4 bg-green-100 text-green-700 rounded">
            <p className="font-medium">✓ Spreadsheet Created Successfully</p>
            <p className="text-sm">{createdSpreadsheet.message}</p>
            <p className="text-xs mt-2 font-mono">{createdSpreadsheet.relative_path}</p>
          </div>

          {createdSpreadsheet.canvas_display && (
            <SpreadsheetPreview
              canvasDisplay={createdSpreadsheet.canvas_display}
              previewUrl={createdSpreadsheet.preview_url}
              fileName={createdSpreadsheet.canvas_display.file_name}
              onDownload={handleDownload}
            />
          )}

          <button
            onClick={() => setCreatedSpreadsheet(null)}
            className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            Create Another
          </button>
        </div>
      )}
    </div>
  );
};

/**
 * Combined Example Component
 * Shows both document and spreadsheet creation in tabs
 */
export const CreationDemoPage: React.FC<CreationExampleProps> = (props) => {
  const [activeTab, setActiveTab] = useState<'document' | 'spreadsheet'>('document');

  return (
    <div className="w-full">
      {/* Tab Navigation */}
      <div className="flex gap-4 border-b mb-6">
        <button
          onClick={() => setActiveTab('document')}
          className={`px-4 py-2 font-medium border-b-2 ${
            activeTab === 'document'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-gray-600 hover:text-gray-800'
          }`}
        >
          📄 Create Document
        </button>
        <button
          onClick={() => setActiveTab('spreadsheet')}
          className={`px-4 py-2 font-medium border-b-2 ${
            activeTab === 'spreadsheet'
              ? 'border-green-600 text-green-600'
              : 'border-transparent text-gray-600 hover:text-gray-800'
          }`}
        >
          📊 Create Spreadsheet
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'document' && <DocumentCreationExample {...props} />}
      {activeTab === 'spreadsheet' && <SpreadsheetCreationExample {...props} />}
    </div>
  );
};

export default CreationDemoPage;
