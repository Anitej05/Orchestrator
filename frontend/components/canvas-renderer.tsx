// components/canvas-renderer.tsx
"use client"

import { useState } from 'react'
import Markdown from '@/components/ui/markdown'
import { FileText, Table, Mail, FileCode, Image as ImageIcon, File, Undo2, Redo2, History, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface CanvasRendererProps {
  canvasType: 'html' | 'markdown' | 'pdf' | 'spreadsheet' | 'spreadsheet_plan' | 'email_preview' | 'document' | 'image' | 'json'
  canvasContent?: string  // For legacy HTML/markdown content
  canvasData?: Record<string, any>  // For structured data (preferred)
  canvasTitle?: string
  canvasMetadata?: Record<string, any>
  requiresConfirmation?: boolean
  confirmationMessage?: string
  onConfirm?: () => void
  onCancel?: () => void
  onUndo?: () => void
  onRedo?: () => void
  onShowHistory?: () => void
}

export function CanvasRenderer({
  canvasType,
  canvasContent,
  canvasData,
  canvasTitle,
  canvasMetadata,
  requiresConfirmation,
  confirmationMessage,
  onConfirm,
  onCancel,
  onUndo,
  onRedo,
  onShowHistory
}: CanvasRendererProps) {
  const [imageError, setImageError] = useState(false)

  // Check if undo/redo is available from canvas data
  const canUndo = canvasData?.can_undo || false
  const canRedo = canvasData?.can_redo || false
  const isDocumentEdit = canvasData?.original_type === 'docx' || canvasData?.file_path?.endsWith('.docx')

  // Debug logging for confirmation props
  console.log('🔘 CanvasRenderer confirmation props:', {
    requiresConfirmation,
    hasOnConfirm: !!onConfirm,
    hasOnCancel: !!onCancel,
    canvasType,
    confirmationMessage
  })

  // Render email preview template
  const renderEmailPreview = (data: any) => {
    return (
      <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900">
        {/* Warning Banner */}
        {requiresConfirmation && (
          <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 p-4 m-4">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-amber-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                  {confirmationMessage || "Please review before confirming"}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Email Preview Card */}
        <div className="flex-1 overflow-auto p-4">
          <div className="max-w-3xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
            {/* Email Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
              <h2 className="text-2xl font-bold mb-2">📧 Email Preview</h2>
              <p className="text-blue-100 text-sm">Review the email details below</p>
            </div>

            {/* Email Metadata */}
            <div className="bg-gray-50 dark:bg-gray-700 p-6 border-b border-gray-200 dark:border-gray-600">
              <div className="space-y-3">
                <div className="flex">
                  <span className="font-semibold text-gray-700 dark:text-gray-300 w-20">To:</span>
                  <span className="text-gray-900 dark:text-gray-100">{data.to?.join(', ')}</span>
                </div>
                {data.cc && data.cc.length > 0 && (
                  <div className="flex">
                    <span className="font-semibold text-gray-700 dark:text-gray-300 w-20">CC:</span>
                    <span className="text-gray-900 dark:text-gray-100">{data.cc.join(', ')}</span>
                  </div>
                )}
                {data.bcc && data.bcc.length > 0 && (
                  <div className="flex">
                    <span className="font-semibold text-gray-700 dark:text-gray-300 w-20">BCC:</span>
                    <span className="text-gray-900 dark:text-gray-100">{data.bcc.join(', ')}</span>
                  </div>
                )}
                <div className="flex">
                  <span className="font-semibold text-gray-700 dark:text-gray-300 w-20">Subject:</span>
                  <span className="text-gray-900 dark:text-gray-100 font-semibold">{data.subject}</span>
                </div>
              </div>
            </div>

            {/* Email Body */}
            <div className="p-6">
              {data.is_html ? (
                <div dangerouslySetInnerHTML={{ __html: data.body }} className="prose dark:prose-invert max-w-none" />
              ) : (
                <pre className="whitespace-pre-wrap font-sans text-gray-900 dark:text-gray-100 leading-relaxed">
                  {data.body}
                </pre>
              )}
            </div>

            {/* Attachments */}
            {data.attachments && data.attachments.count > 0 && (
              <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4 m-6">
                <div className="flex items-center">
                  <svg className="h-5 w-5 text-blue-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                  </svg>
                  <span className="font-semibold text-blue-900 dark:text-blue-100">
                    {data.attachments.count} attachment{data.attachments.count > 1 ? 's' : ''}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Confirmation Buttons */}
        {requiresConfirmation && (onConfirm || onCancel) && (
          <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
            <div className="max-w-3xl mx-auto flex justify-end gap-3">
              {onCancel && (
                <button
                  onClick={onCancel}
                  className="px-6 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 font-medium transition-colors"
                >
                  Cancel
                </button>
              )}
              {onConfirm && (
                <button
                  onClick={onConfirm}
                  className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Confirm & Send
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    )
  }

  // Render based on canvas type
  const renderContent = () => {
    // Debug logging
    console.log('🎨 CanvasRenderer renderContent:', {
      canvasType,
      hasCanvasData: !!canvasData,
      hasCanvasContent: !!canvasContent,
      canvasDataKeys: canvasData ? Object.keys(canvasData) : [],
      canvasContentPreview: canvasContent ? canvasContent.substring(0, 100) : null
    })

    // If we have structured data, use it (preferred)
    if (canvasData) {
      switch (canvasType) {
        case 'email_preview':
          return renderEmailPreview(canvasData)

        case 'spreadsheet_plan':
          // Render spreadsheet execution plan for approval
          const planActions = canvasData.rows || []
          const planHeaders = canvasData.headers || ['Step', 'Action', 'Description']
          const planSummary = canvasData.plan_summary || ''
          const estimatedSteps = canvasData.estimated_steps || planActions.length

          return (
            <div className="flex flex-col h-full bg-white dark:bg-gray-900">
              {/* Header */}
              <div className="bg-gradient-to-r from-blue-600 to-blue-500 px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Table className="w-5 h-5 text-white" />
                  <div>
                    <h3 className="text-white font-semibold text-sm">Spreadsheet Execution Plan</h3>
                    <div className="flex items-center gap-3 text-xs text-blue-100">
                      <span>{estimatedSteps} estimated operations</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Plan Summary */}
              {planSummary && (
                <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 px-4 py-3">
                  <p className="text-sm text-blue-900 dark:text-blue-100 font-medium">Plan Summary</p>
                  <p className="text-sm text-blue-800 dark:text-blue-200 mt-1">{planSummary}</p>
                </div>
              )}

              {/* Warning Banner */}
              {requiresConfirmation && (
                <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 px-4 py-3">
                  <div className="flex items-start">
                    <svg className="h-5 w-5 text-amber-500 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                        {confirmationMessage || 'Review the plan and approve to execute changes'}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Actions Table */}
              <div className="flex-1 overflow-auto">
                <table className="w-full border-collapse">
                  <thead className="sticky top-0 z-10">
                    <tr className="bg-gray-100 dark:bg-gray-800">
                      {planHeaders.map((header: string, idx: number) => (
                        <th
                          key={idx}
                          className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-left text-xs font-semibold text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800"
                        >
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {planActions.map((row: string[], rowIdx: number) => (
                      <tr
                        key={rowIdx}
                        className={`${rowIdx % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800/50'}`}
                      >
                        {row.map((cell: string, cellIdx: number) => (
                          <td
                            key={cellIdx}
                            className="border border-gray-300 dark:border-gray-600 px-4 py-3 text-sm text-gray-900 dark:text-gray-100"
                          >
                            {cellIdx === 0 ? (
                              <span className="font-semibold text-blue-600 dark:text-blue-400">{cell}</span>
                            ) : cellIdx === 1 ? (
                              <span className="font-medium text-gray-700 dark:text-gray-300">{cell}</span>
                            ) : (
                              <span className="text-gray-600 dark:text-gray-400">{cell}</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Confirmation Buttons */}
              {requiresConfirmation && (onConfirm || onCancel) && (
                <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
                  <div className="flex justify-end gap-3">
                    {onCancel && (
                      <button
                        onClick={onCancel}
                        className="px-6 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 font-medium transition-colors"
                      >
                        Cancel
                      </button>
                    )}
                    {onConfirm && (
                      <button
                        onClick={onConfirm}
                        className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                      >
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Approve & Execute
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          )

        case 'spreadsheet':
          // Render spreadsheet from structured data - Excel-like styling
          const spreadsheetData = canvasData.data || []
          const spreadsheetHeaders = spreadsheetData.length > 0 ? spreadsheetData[0] : (canvasData.headers || [])
          const spreadsheetRows = spreadsheetData.length > 1 ? spreadsheetData.slice(1) : (canvasData.rows || [])
          const spreadsheetFilename = canvasData.filename || 'Spreadsheet'
          const spreadsheetMetadata = canvasData.metadata || {}
          const fileId = canvasData.file_id

          // Generate Excel-style column letters (A, B, C, ... Z, AA, AB, etc.)
          const getColumnLetter = (index: number): string => {
            let letter = ''
            while (index >= 0) {
              letter = String.fromCharCode((index % 26) + 65) + letter
              index = Math.floor(index / 26) - 1
            }
            return letter
          }

          return (
            <div className="flex flex-col h-full bg-white dark:bg-gray-900">
              {/* Excel-like header bar */}
              <div className="bg-gradient-to-r from-green-600 to-green-500 px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Table className="w-5 h-5 text-white" />
                  <div>
                    <h3 className="text-white font-semibold text-sm">
                      {spreadsheetFilename}
                    </h3>
                    <div className="flex items-center gap-3 text-xs text-green-100">
                      <span>{spreadsheetMetadata.rows_total || spreadsheetRows.length} rows</span>
                      <span>×</span>
                      <span>{spreadsheetMetadata.columns || spreadsheetHeaders.length} columns</span>
                      {spreadsheetMetadata.truncated && (
                        <span className="bg-yellow-500 text-yellow-900 px-2 py-0.5 rounded text-xs font-medium">
                          Showing {spreadsheetMetadata.rows_shown} of {spreadsheetMetadata.rows_total}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                {fileId && (
                  <div className="flex gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => window.open(`/api/spreadsheet/download/${fileId}?format=xlsx`, '_blank')}
                      className="bg-white/20 hover:bg-white/30 text-white border-0 text-xs"
                    >
                      <Download className="w-3 h-3 mr-1" />
                      Excel
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => window.open(`/api/spreadsheet/download/${fileId}?format=csv`, '_blank')}
                      className="bg-white/20 hover:bg-white/30 text-white border-0 text-xs"
                    >
                      <Download className="w-3 h-3 mr-1" />
                      CSV
                    </Button>
                  </div>
                )}
              </div>

              {/* Formula bar placeholder */}
              <div className="bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-600 px-3 py-1.5 flex items-center gap-2">
                <span className="text-xs text-gray-500 dark:text-gray-400 font-mono w-12">A1</span>
                <div className="h-4 border-l border-gray-300 dark:border-gray-600"></div>
                <span className="text-xs text-gray-600 dark:text-gray-300 font-mono">
                  {spreadsheetHeaders[0] || ''}
                </span>
              </div>

              {/* Spreadsheet grid */}
              <div className="flex-1 overflow-auto">
                <div className="inline-block min-w-full">
                  <table className="min-w-full border-collapse">
                    {/* Column headers with letters */}
                    <thead className="sticky top-0 z-10">
                      <tr className="bg-gray-100 dark:bg-gray-800">
                        {/* Row number header cell */}
                        <th className="w-12 min-w-[48px] bg-gray-200 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-center text-xs font-medium text-gray-600 dark:text-gray-400 sticky left-0 z-20"></th>
                        {/* Column letter headers */}
                        {spreadsheetHeaders.map((_: any, idx: number) => (
                          <th
                            key={idx}
                            className="bg-gray-200 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 px-1 py-1 text-center text-xs font-medium text-gray-600 dark:text-gray-400 min-w-[80px]"
                          >
                            {getColumnLetter(idx)}
                          </th>
                        ))}
                      </tr>
                      {/* Actual header row with data */}
                      <tr className="bg-green-50 dark:bg-green-900/30">
                        <td className="w-12 bg-gray-200 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-center text-xs font-bold text-gray-700 dark:text-gray-300 sticky left-0 z-20">1</td>
                        {spreadsheetHeaders.map((header: any, idx: number) => (
                          <td
                            key={idx}
                            className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-xs font-bold text-gray-900 dark:text-gray-100 bg-green-50 dark:bg-green-900/30"
                          >
                            {header}
                          </td>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {spreadsheetRows.map((row: any[], rowIdx: number) => (
                        <tr
                          key={rowIdx}
                          className={`${rowIdx % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-100 dark:bg-gray-800/70'} hover:bg-blue-50 dark:hover:bg-blue-900/30`}
                        >
                          {/* Row number */}
                          <td className="w-12 bg-gray-200 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-center text-xs font-medium text-gray-600 dark:text-gray-400 sticky left-0">
                            {rowIdx + 2}
                          </td>
                          {/* Data cells */}
                          {row.map((cell: any, cellIdx: number) => (
                            <td
                              key={cellIdx}
                              className="border border-gray-300 dark:border-gray-600 px-3 py-1.5 text-xs text-gray-900 dark:text-gray-100 font-mono"
                            >
                              {cell !== null && cell !== undefined ? String(cell) : ''}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Status bar */}
              <div className="bg-gray-100 dark:bg-gray-800 border-t border-gray-300 dark:border-gray-600 px-4 py-1 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>Sheet 1 of 1</span>
                <div className="flex items-center gap-4">
                  <span>Rows: {spreadsheetMetadata.rows_total || spreadsheetRows.length}</span>
                  <span>Columns: {spreadsheetMetadata.columns || spreadsheetHeaders.length}</span>
                </div>
              </div>
            </div>
          )

        case 'document':
          // Render document from structured data
          const docTitle = canvasData.title || 'Document'
          const docContent = canvasData.content || ''
          const docStatus = canvasData.status // 'preview', 'created', 'edited'
          const docFilePath = canvasData.file_path
          const docFileType = canvasData.file_type
          const docMetadata = canvasData.metadata || {}

          return (
            <div className="p-6">
              <div className="mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold flex items-center gap-2 text-gray-900 dark:text-gray-100">
                  <FileText className="w-5 h-5" />
                  {docTitle}
                </h3>
                <div className="flex items-center gap-2 mt-2">
                  {docStatus && (
                    <span className={`inline-block px-2 py-1 text-xs rounded ${docStatus === 'preview' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' :
                      docStatus === 'created' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                        docStatus === 'edited' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                          'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
                      }`}>
                      {docStatus.charAt(0).toUpperCase() + docStatus.slice(1)}
                    </span>
                  )}
                  {docFileType && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {docFileType.toUpperCase()}
                    </span>
                  )}
                </div>
              </div>
              {/* Render markdown content properly */}
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <Markdown content={docContent} />
              </div>
              {(docFilePath || docMetadata.file_path) && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
                  <p><strong>File:</strong> {docFilePath || docMetadata.file_path}</p>
                </div>
              )}
              {docMetadata.original_content && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <details className="text-sm">
                    <summary className="cursor-pointer text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200">
                      View original content
                    </summary>
                    <pre className="mt-2 whitespace-pre-wrap font-sans bg-gray-100 dark:bg-gray-800 p-3 rounded text-xs text-gray-700 dark:text-gray-300">
                      {docMetadata.original_content}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          )

        case 'pdf':
          // Render PDF from structured data (with base64 or URL)
          const pdfData = canvasData.pdf_data || canvasData.content
          if (pdfData) {
            // Add zoom parameter to PDF URL for better quality
            const pdfSrcFromData = pdfData.startsWith('data:') || pdfData.startsWith('http')
              ? pdfData
              : `data:application/pdf;base64,${pdfData}`

            // Add #zoom=125 to display PDF at 125% zoom for better readability
            const pdfSrcWithZoom = pdfSrcFromData + '#zoom=125&toolbar=1&navpanes=0&scrollbar=1'

            return (
              <div className="h-full flex flex-col">
                <div className="mb-4 pb-4 border-b border-gray-200 dark:border-gray-700 px-6 pt-6">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold flex items-center gap-2 text-gray-900 dark:text-gray-100">
                        <FileText className="w-5 h-5" />
                        {canvasData.title || 'PDF Document'}
                      </h3>
                      {canvasData.file_path && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          {canvasData.file_path}
                        </p>
                      )}
                      {canvasData.status === 'edited' && (
                        <p className="text-xs text-green-600 dark:text-green-400 mt-1 font-medium">
                          ✓ Document edited successfully
                        </p>
                      )}
                    </div>

                    {/* Undo/Redo Controls for Document Edits */}
                    {isDocumentEdit && (onUndo || onRedo || onShowHistory) && (
                      <div className="flex items-center gap-2 ml-4">
                        {onUndo && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={onUndo}
                            disabled={!canUndo}
                            title={canUndo ? "Undo last edit" : "No edits to undo"}
                            className="flex items-center gap-1"
                          >
                            <Undo2 className="w-4 h-4" />
                            <span className="hidden sm:inline">Undo</span>
                          </Button>
                        )}
                        {onRedo && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={onRedo}
                            disabled={!canRedo}
                            title={canRedo ? "Redo last undone edit" : "No edits to redo"}
                            className="flex items-center gap-1"
                          >
                            <Redo2 className="w-4 h-4" />
                            <span className="hidden sm:inline">Redo</span>
                          </Button>
                        )}
                        {onShowHistory && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={onShowHistory}
                            title="View version history"
                            className="flex items-center gap-1"
                          >
                            <History className="w-4 h-4" />
                            <span className="hidden sm:inline">History</span>
                          </Button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex-1 px-6 pb-6">
                  <iframe
                    src={pdfSrcWithZoom}
                    className="w-full h-full min-h-[700px] border-0 rounded-lg"
                    title={canvasData.title || "PDF Document"}
                    key={pdfSrcFromData.substring(0, 100)} // Force re-render on content change
                  />
                </div>
              </div>
            )
          }
          break

        case 'json':
          return (
            <div className="p-6">
              <pre className="text-sm bg-gray-50 dark:bg-gray-900 p-4 rounded overflow-auto">
                {JSON.stringify(canvasData, null, 2)}
              </pre>
            </div>
          )

        default:
          // Fallback to content rendering
          break
      }
    }

    // Legacy: render from canvasContent (HTML/markdown strings)
    if (canvasContent) {
      switch (canvasType) {
        case 'html':
        case 'email_preview':
          return (
            <iframe
              key={canvasContent?.substring(0, 100)}
              srcDoc={canvasContent}
              className="w-full h-full min-h-[500px] border-0"
              title={canvasTitle || "Canvas HTML Content"}
              sandbox="allow-scripts allow-same-origin"
            />
          )

        case 'markdown':
          return (
            <div className="prose prose-sm max-w-none p-6 overflow-auto">
              <Markdown content={canvasContent} />
            </div>
          )

        case 'pdf':
          // For PDFs, canvasContent should be a base64 string or URL
          const pdfSrc = canvasContent.startsWith('data:') || canvasContent.startsWith('http')
            ? canvasContent
            : `data:application/pdf;base64,${canvasContent}`

          // Add zoom parameter for better quality
          const pdfSrcWithZoom = pdfSrc + '#zoom=125&toolbar=1&navpanes=0&scrollbar=1'

          return (
            <iframe
              src={pdfSrcWithZoom}
              className="w-full h-full min-h-[700px] border-0"
              title={canvasTitle || "PDF Document"}
              key={pdfSrc.substring(0, 100)} // Force re-render on content change
            />
          )

        case 'image':
          // For images, canvasContent should be a base64 string or URL
          const imageSrc = canvasContent.startsWith('data:') || canvasContent.startsWith('http')
            ? canvasContent
            : `data:image/png;base64,${canvasContent}`

          if (imageError) {
            return (
              <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-gray-500">
                <ImageIcon className="w-16 h-16 mb-4 text-gray-300" />
                <p className="font-semibold">Failed to load image</p>
                <p className="text-sm mt-2">The image could not be displayed</p>
              </div>
            )
          }

          return (
            <div className="flex items-center justify-center p-6 bg-gray-50 dark:bg-gray-900">
              <img
                src={imageSrc}
                alt={canvasTitle || "Canvas Image"}
                className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-lg"
                onError={() => setImageError(true)}
              />
            </div>
          )

        case 'spreadsheet':
          // For spreadsheets, render as a table if it's CSV/JSON data
          try {
            let data: any[][] = []

            if (canvasContent.startsWith('[')) {
              // JSON array format
              data = JSON.parse(canvasContent)
            } else {
              // CSV format
              const lines = canvasContent.split('\n').filter(line => line.trim())
              data = lines.map(line => line.split(',').map(cell => cell.trim()))
            }

            if (data.length === 0) {
              return <div className="p-6 text-gray-500">Empty spreadsheet</div>
            }

            const headers = data[0]
            const rows = data.slice(1)

            return (
              <div className="overflow-auto p-6">
                <div className="inline-block min-w-full align-middle">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 border border-gray-200 dark:border-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-800">
                      <tr>
                        {headers.map((header: any, idx: number) => (
                          <th
                            key={idx}
                            className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider border-r border-gray-200 dark:border-gray-700 last:border-r-0"
                          >
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                      {rows.map((row: any[], rowIdx: number) => (
                        <tr key={rowIdx} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                          {row.map((cell: any, cellIdx: number) => (
                            <td
                              key={cellIdx}
                              className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 border-r border-gray-200 dark:border-gray-700 last:border-r-0"
                            >
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          } catch (error) {
            return (
              <div className="p-6">
                <pre className="text-sm bg-gray-50 dark:bg-gray-900 p-4 rounded overflow-auto">
                  {canvasContent}
                </pre>
              </div>
            )
          }

        case 'json':
          // Pretty print JSON
          try {
            const jsonData = typeof canvasContent === 'string' ? JSON.parse(canvasContent) : canvasContent
            return (
              <div className="p-6">
                <pre className="text-sm bg-gray-50 dark:bg-gray-900 p-4 rounded overflow-auto">
                  {JSON.stringify(jsonData, null, 2)}
                </pre>
              </div>
            )
          } catch (error) {
            return (
              <div className="p-6">
                <pre className="text-sm bg-gray-50 dark:bg-gray-900 p-4 rounded overflow-auto">
                  {canvasContent}
                </pre>
              </div>
            )
          }

        case 'document':
          // For documents, render as formatted text with metadata
          const docData: any = typeof canvasContent === 'string' ? { content: canvasContent } : canvasContent
          const docTitle = docData.title || 'Document'
          const docContent = docData.content || canvasContent
          const docStatus = docData.status // 'preview', 'created', 'edited'
          const docMetadata = docData.metadata || {}

          return (
            <div className="p-6">
              <div className="mb-4 pb-4 border-b">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  {docTitle}
                </h3>
                {docStatus && (
                  <span className={`inline-block mt-2 px-2 py-1 text-xs rounded ${docStatus === 'preview' ? 'bg-yellow-100 text-yellow-800' :
                    docStatus === 'created' ? 'bg-green-100 text-green-800' :
                      docStatus === 'edited' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'
                    }`}>
                    {docStatus.charAt(0).toUpperCase() + docStatus.slice(1)}
                  </span>
                )}
              </div>
              <div className="prose prose-sm max-w-none">
                {/* Render markdown if content looks like markdown, otherwise use pre */}
                {typeof docContent === 'string' && (docContent.includes('##') || docContent.includes('**') || docContent.includes('```')) ? (
                  <Markdown content={docContent} />
                ) : (
                  <pre className="whitespace-pre-wrap font-sans bg-gray-50 dark:bg-gray-900 p-4 rounded">
                    {docContent}
                  </pre>
                )}
              </div>
              {docMetadata.file_path && (
                <div className="mt-4 pt-4 border-t text-xs text-gray-500">
                  <p><strong>File:</strong> {docMetadata.file_path}</p>
                </div>
              )}
            </div>
          )

        default:
          // Fallback: render as plain text
          return (
            <div className="p-6">
              <pre className="text-sm bg-gray-50 dark:bg-gray-900 p-4 rounded overflow-auto whitespace-pre-wrap">
                {canvasContent}
              </pre>
            </div>
          )
      }
    }

    // No content to display
    return (
      <div className="flex items-center justify-center h-full min-h-[400px] text-gray-500">
        <div className="text-center">
          <p className="font-semibold">No content to display</p>
          <p className="text-sm mt-2">Canvas content is empty</p>
        </div>
      </div>
    )
  }

  // Get icon based on canvas type
  const getIcon = () => {
    switch (canvasType) {
      case 'email_preview':
        return <Mail className="w-5 h-5" />
      case 'spreadsheet':
        return <Table className="w-5 h-5" />
      case 'pdf':
      case 'document':
        return <FileText className="w-5 h-5" />
      case 'json':
        return <FileCode className="w-5 h-5" />
      case 'image':
        return <ImageIcon className="w-5 h-5" />
      default:
        return <File className="w-5 h-5" />
    }
  }

  return (
    <div className="h-full flex flex-col">
      {canvasTitle && (
        <div className="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="text-blue-600 dark:text-blue-400">
              {getIcon()}
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                {canvasTitle}
              </h3>
              {canvasMetadata && Object.keys(canvasMetadata).length > 0 && (
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {canvasType.replace('_', ' ').toUpperCase()}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
      <div className="flex-1 overflow-auto">
        {renderContent()}
      </div>

      {/* Global Confirmation Buttons - shown for any canvas type that requires confirmation */}
      {(() => {
        const shouldShow = requiresConfirmation && (onConfirm || onCancel) && canvasType !== 'email_preview';
        console.log('🔘 Button visibility check:', {
          requiresConfirmation,
          hasOnConfirm: !!onConfirm,
          hasOnCancel: !!onCancel,
          canvasType,
          isNotEmailPreview: canvasType !== 'email_preview',
          shouldShowButtons: shouldShow
        });

        if (!shouldShow) return null;

        return (
          <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
            <div className="max-w-3xl mx-auto flex justify-end gap-3">
              {onCancel && (
                <button
                  onClick={onCancel}
                  className="px-6 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 font-medium transition-colors"
                >
                  Cancel
                </button>
              )}
              {onConfirm && (
                <button
                  onClick={onConfirm}
                  className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Confirm
                </button>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  )
}
