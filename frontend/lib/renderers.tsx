import React from 'react';

/**
 * Route-based renderer mapping for agent responses
 * 
 * PHASE 7: Frontend Renderer Mapping
 * 
 * This module provides a unified approach to rendering agent responses based on
 * their route and task_type. Eliminates response shape guessing.
 */

export interface StandardResponse {
  success: boolean;
  route: string;
  task_type: string;
  data: Record<string, any>;
  preview?: Record<string, any>;
  artifact?: {
    id: string;
    filename: string;
    url?: string;
  };
  metrics: {
    rows_processed: number;
    columns_affected: number;
    execution_time_ms: number;
    llm_calls: number;
  };
  confidence: number;
  needs_clarification: boolean;
  message: string;
}

export type RouteRenderer = {
  route: string;
  render: (response: StandardResponse) => React.ReactNode;
};

// ============================================================================
// RENDERER IMPLEMENTATIONS
// ============================================================================

/**
 * Text + Statistics renderer for summaries
 */
function renderTextStats(response: StandardResponse) {
  const { data, message } = response;
  
  return (
    <div className="space-y-4">
      {message && <div className="prose text-gray-700">{message}</div>}
      
      {data.statistics && Array.isArray(data.statistics) && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
          {data.statistics.map((stat: any, idx: number) => (
            <div key={idx} className="p-4 border border-gray-200 rounded-lg bg-white shadow-sm">
              <h4 className="font-semibold text-gray-900">{stat.column}</h4>
              <div className="text-sm text-gray-600 space-y-1 mt-2">
                {stat.mean !== undefined && <p>Mean: {stat.mean.toFixed(2)}</p>}
                {stat.min !== undefined && <p>Min: {stat.min}</p>}
                {stat.max !== undefined && <p>Max: {stat.max}</p>}
              </div>
            </div>
          ))}
        </div>
      )}
      
      {data.row_count !== undefined && (
        <div className="text-sm text-gray-500 mt-4">
          {data.row_count} rows × {data.column_count} columns
        </div>
      )}
    </div>
  );
}

/**
 * Table preview renderer for display operations
 */
function renderTablePreview(response: StandardResponse) {
  const preview = response.preview || response.data.preview;
  
  if (!preview || !Array.isArray(preview)) {
    return <div className="text-gray-500">No preview available</div>;
  }
  
  const columns = Object.keys(preview[0] || {});
  
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {preview.map((row: any, idx: number) => (
            <tr key={idx}>
              {columns.map((col) => (
                <td key={col} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {row[col]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * JSON diff renderer for comparisons
 */
function renderJsonDiff(response: StandardResponse) {
  const { data, artifact, message } = response;
  
  return (
    <div className="space-y-4">
      {message && <div className="text-gray-700 mb-4">{message}</div>}
      
      <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm">
        {JSON.stringify(data, null, 2)}
      </pre>
      
      {artifact && (
        <div className="mt-4">
          <a
            href={artifact.url || `/api/download/${artifact.id}`}
            download={artifact.filename}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            📥 Download {artifact.filename}
          </a>
        </div>
      )}
    </div>
  );
}

/**
 * Plan canvas renderer for plan_operation
 */
function renderPlanCanvas(response: StandardResponse) {
  const { data, message } = response;
  const actions = data.actions || [];
  
  return (
    <div className="space-y-4">
      {message && <div className="text-gray-700 mb-4">{message}</div>}
      
      {actions.length > 0 ? (
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            {actions.map((action: any, idx: number) => (
              <li key={idx} className="px-6 py-4">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <span className="inline-flex items-center justify-center h-8 w-8 rounded-full bg-blue-100 text-blue-800 font-semibold">
                      {idx + 1}
                    </span>
                  </div>
                  <div className="ml-4">
                    <h4 className="text-sm font-medium text-gray-900">
                      {action.action_type?.replace('_', ' ').toUpperCase()}
                    </h4>
                    <p className="text-sm text-gray-500 mt-1">
                      {action.description || `Execute ${action.action_type}`}
                    </p>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="text-gray-500">No actions planned</div>
      )}
    </div>
  );
}

/**
 * Answer with optional table renderer for nl_query
 */
function renderAnswerWithTable(response: StandardResponse) {
  const { data, message, preview } = response;
  
  return (
    <div className="space-y-4">
      {message && (
        <div className="prose text-gray-700 mb-4">
          {message}
        </div>
      )}
      
      {data.answer && (
        <div className="bg-blue-50 p-4 rounded-lg">
          <p className="text-gray-900">{data.answer}</p>
        </div>
      )}
      
      {preview && <div className="mt-4">{renderTablePreview(response)}</div>}
      
      {data.steps_taken && Array.isArray(data.steps_taken) && data.steps_taken.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm text-gray-600">
            View {data.steps_taken.length} analysis steps
          </summary>
          <ul className="list-disc list-inside mt-2 text-sm text-gray-600">
            {data.steps_taken.map((step: string, idx: number) => (
              <li key={idx}>{step}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/**
 * File upload success renderer
 */
function renderFileUploadSuccess(response: StandardResponse) {
  const { data, message } = response;
  
  return (
    <div className="space-y-4">
      <div className="rounded-md bg-green-50 p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-green-800">
              {message || 'File uploaded successfully'}
            </h3>
            <div className="mt-2 text-sm text-green-700">
              <p>File ID: <code className="bg-green-100 px-2 py-1 rounded">{data.file_id}</code></p>
              <p className="mt-1">
                {data.rows} rows × {data.columns} columns
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * File created success renderer
 */
function renderFileCreatedSuccess(response: StandardResponse) {
  const { data, artifact, message } = response;
  
  return (
    <div className="space-y-4">
      <div className="rounded-md bg-green-50 p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-green-800">
              {message || 'File created successfully'}
            </h3>
          </div>
        </div>
      </div>
      
      {artifact && (
        <div className="mt-4">
          <a
            href={artifact.url || `/api/download/${artifact.id}`}
            download={artifact.filename}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700"
          >
            📥 Download {artifact.filename}
          </a>
        </div>
      )}
    </div>
  );
}

/**
 * Default fallback renderer
 */
function renderDefault(response: StandardResponse) {
  const { data, message, artifact } = response;
  
  return (
    <div className="space-y-4">
      {message && <div className="text-gray-700">{message}</div>}
      
      {Object.keys(data).length > 0 && (
        <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
      
      {artifact && (
        <div className="mt-4">
          <a
            href={artifact.url || `/api/download/${artifact.id}`}
            download={artifact.filename}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-800"
          >
            📥 Download {artifact.filename}
          </a>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// ROUTE RENDERER MAPPING
// ============================================================================

export const ROUTE_RENDERERS: RouteRenderer[] = [
  { route: '/get_summary', render: renderTextStats },
  { route: '/display', render: renderTablePreview },
  { route: '/compare', render: renderJsonDiff },
  { route: '/merge', render: renderTablePreview },
  { route: '/plan_operation', render: renderPlanCanvas },
  { route: '/nl_query', render: renderAnswerWithTable },
  { route: '/upload', render: renderFileUploadSuccess },
  { route: '/create', render: renderFileCreatedSuccess },
  { route: '/transform', render: renderTablePreview },
  { route: '/simulate_operation', render: renderPlanCanvas },
];

/**
 * Main renderer function
 * 
 * Usage:
 * ```tsx
 * import { renderAgentResponse } from '@/lib/renderers';
 * 
 * function AgentResponseComponent({ response }) {
 *   return renderAgentResponse(response);
 * }
 * ```
 */
export function renderAgentResponse(response: StandardResponse) {
  const renderer = ROUTE_RENDERERS.find(r => r.route === response.route);
  
  if (!renderer) {
    console.warn(`No renderer for route: ${response.route}`);
    return renderDefault(response);
  }
  
  // Wrap in error boundary
  try {
    return renderer.render(response);
  } catch (error) {
    console.error(`Renderer error for route ${response.route}:`, error);
    return renderDefault(response);
  }
}

/**
 * Render error state
 */
export function renderErrorState(response: StandardResponse) {
  return (
    <div className="rounded-md bg-red-50 p-4">
      <div className="flex">
        <div className="flex-shrink-0">
          <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-red-800">
            {response.message || 'An error occurred'}
          </h3>
          {response.needs_clarification && (
            <p className="mt-2 text-sm text-red-700">
              Please provide more information or try a different approach.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
