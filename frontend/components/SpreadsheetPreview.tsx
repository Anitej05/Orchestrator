/**
 * Spreadsheet Preview Component
 * Displays created spreadsheets with data preview
 * Features:
 * - Data table preview
 * - Download/full view options
 * - Pagination for large datasets
 */

import React, { useState, useMemo } from 'react';

export interface SpreadsheetCanvasDisplay {
  canvas_type: 'spreadsheet';
  file_name: string;
  columns: string[];
  rows: (string | number | boolean)[][];
  total_rows: number;
  preview_url: string;
}

export interface SpreadsheetPreviewProps {
  canvasDisplay: SpreadsheetCanvasDisplay;
  previewUrl: string;
  fileName: string;
  onDownload?: () => void;
  className?: string;
}

/**
 * Spreadsheet Preview Component
 * Shows data table with first N rows
 */
export const SpreadsheetPreview: React.FC<SpreadsheetPreviewProps> = ({
  canvasDisplay,
  previewUrl,
  fileName,
  onDownload,
  className = '',
}) => {
  const [itemsPerPage] = useState(10);
  const [currentPage, setCurrentPage] = useState(0);

  const paginatedRows = useMemo(() => {
    const start = currentPage * itemsPerPage;
    return canvasDisplay.rows.slice(start, start + itemsPerPage);
  }, [currentPage, itemsPerPage, canvasDisplay.rows]);

  const totalPages = Math.ceil(canvasDisplay.total_rows / itemsPerPage);

  return (
    <div className={`border rounded-lg p-6 bg-white shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">Spreadsheet Preview</h3>
          <p className="text-sm text-gray-600">{fileName}</p>
          <p className="text-xs text-gray-500 mt-1">
            Total rows: {canvasDisplay.total_rows} | Showing page {currentPage + 1} of {totalPages}
          </p>
        </div>
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

      <div className="border-t pt-4 overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead className="bg-gray-100 border-b">
            <tr>
              {canvasDisplay.columns.map((col, idx) => (
                <th
                  key={idx}
                  className="px-4 py-2 text-left font-semibold text-gray-700 border-r last:border-r-0"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedRows.map((row, rowIdx) => (
              <tr key={rowIdx} className="border-b hover:bg-gray-50 last:border-b-0">
                {row.map((cell, cellIdx) => (
                  <td
                    key={cellIdx}
                    className="px-4 py-2 text-gray-800 border-r last:border-r-0"
                  >
                    {renderCell(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>

        {paginatedRows.length === 0 && (
          <div className="text-center py-8 text-gray-500">No data to display</div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4 border-t pt-4">
          <button
            onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
            disabled={currentPage === 0}
            className="px-3 py-1 text-sm bg-gray-300 text-gray-700 rounded hover:bg-gray-400 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="text-sm text-gray-600">
            Page {currentPage + 1} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
            disabled={currentPage >= totalPages - 1}
            className="px-3 py-1 text-sm bg-gray-300 text-gray-700 rounded hover:bg-gray-400 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

/**
 * Render cell value with proper formatting
 */
function renderCell(value: string | number | boolean): React.ReactNode {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return String(value);
}

export default SpreadsheetPreview;
