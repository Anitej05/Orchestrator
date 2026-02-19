/**
 * Example: Using Document Viewer Adapter
 */

import { DocumentViewerAdapter } from '@/lib/adapters/document-viewer'

export function DocumentViewerExample() {
  return (
    <div className="h-screen">
      <DocumentViewerAdapter
        documents={[
          { 
            uri: 'https://example.com/document.pdf',
            fileName: 'document.pdf' 
          }
        ]}
        style={{ height: '100%' }}
        onDocumentLoadSuccess={(doc) => {
          console.log('Document loaded:', doc.fileName)
        }}
        onDocumentLoadFail={(error, doc) => {
          console.error('Failed to load:', doc.fileName, error)
        }}
      />
    </div>
  )
}

/**
 * Example: Using State Management Adapter
 */

import { createStore } from '@/lib/adapters/state-management'

// Define your state interface
interface TodoState {
  todos: Array<{ id: string; text: string; completed: boolean }>
  addTodo: (text: string) => void
  toggleTodo: (id: string) => void
  removeTodo: (id: string) => void
}

// Create store with adapter
export const useTodoStore = createStore<TodoState>({
  initialState: {
    todos: [],
    addTodo: (text) => {
      const newTodo = { 
        id: Math.random().toString(), 
        text, 
        completed: false 
      }
      useTodoStore.setState(state => ({
        todos: [...state.todos, newTodo]
      }))
    },
    toggleTodo: (id) => {
      useTodoStore.setState(state => ({
        todos: state.todos.map(todo =>
          todo.id === id ? { ...todo, completed: !todo.completed } : todo
        )
      }))
    },
    removeTodo: (id) => {
      useTodoStore.setState(state => ({
        todos: state.todos.filter(todo => todo.id !== id)
      }))
    }
  },
  // Optional: persist to localStorage
  persist: {
    name: 'todo-storage'
  }
})

// Use in component
export function TodoList() {
  const todos = useTodoStore.useStore(s => s.todos)
  const addTodo = useTodoStore.useStore(s => s.addTodo)
  const toggleTodo = useTodoStore.useStore(s => s.toggleTodo)
  const removeTodo = useTodoStore.useStore(s => s.removeTodo)

  return (
    <div>
      <button onClick={() => addTodo('New Todo')}>Add Todo</button>
      {todos.map(todo => (
        <div key={todo.id}>
          <input
            type="checkbox"
            checked={todo.completed}
            onChange={() => toggleTodo(todo.id)}
          />
          <span>{todo.text}</span>
          <button onClick={() => removeTodo(todo.id)}>Remove</button>
        </div>
      ))}
    </div>
  )
}

/**
 * Example: Using Feature Flags
 */

import { isFeatureEnabled, useFeatureFlag } from '@/lib/feature-flags'

// Check feature flag (server-side or client-side)
export function ConditionalFeature() {
  if (isFeatureEnabled('enable_experimental_ui')) {
    return <NewExperimentalUI />
  }
  return <OldStableUI />
}

// Use as React hook
export function FeatureFlaggedComponent() {
  const isEnabled = useFeatureFlag('enable_advanced_analytics')
  
  return (
    <div>
      {isEnabled && <AdvancedAnalytics />}
    </div>
  )
}

/**
 * Example: Creating a Custom Adapter
 */

// 1. Define types
export interface IHttpAdapter {
  name: string
  version: string
  get<T>(url: string, options?: RequestOptions): Promise<T>
  post<T>(url: string, data: any, options?: RequestOptions): Promise<T>
}

export interface RequestOptions {
  headers?: Record<string, string>
  timeout?: number
}

// 2. Implement primary adapter (using fetch)
export class FetchHttpAdapter implements IHttpAdapter {
  name = 'Fetch API'
  version = '1.0.0'

  async get<T>(url: string, options?: RequestOptions): Promise<T> {
    const response = await fetch(url, {
      headers: options?.headers,
      signal: options?.timeout 
        ? AbortSignal.timeout(options.timeout) 
        : undefined
    })
    return response.json()
  }

  async post<T>(url: string, data: any, options?: RequestOptions): Promise<T> {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers
      },
      body: JSON.stringify(data),
      signal: options?.timeout 
        ? AbortSignal.timeout(options.timeout) 
        : undefined
    })
    return response.json()
  }
}

// 3. Implement fallback adapter (using XMLHttpRequest)
export class XhrHttpAdapter implements IHttpAdapter {
  name = 'XMLHttpRequest'
  version = '1.0.0'

  async get<T>(url: string, options?: RequestOptions): Promise<T> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('GET', url)
      
      if (options?.headers) {
        Object.entries(options.headers).forEach(([key, value]) => {
          xhr.setRequestHeader(key, value)
        })
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          reject(new Error(`HTTP ${xhr.status}`))
        }
      }

      xhr.onerror = () => reject(new Error('Network error'))
      
      if (options?.timeout) {
        xhr.timeout = options.timeout
      }

      xhr.send()
    })
  }

  async post<T>(url: string, data: any, options?: RequestOptions): Promise<T> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', url)
      
      xhr.setRequestHeader('Content-Type', 'application/json')
      
      if (options?.headers) {
        Object.entries(options.headers).forEach(([key, value]) => {
          xhr.setRequestHeader(key, value)
        })
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          reject(new Error(`HTTP ${xhr.status}`))
        }
      }

      xhr.onerror = () => reject(new Error('Network error'))
      
      if (options?.timeout) {
        xhr.timeout = options.timeout
      }

      xhr.send(JSON.stringify(data))
    })
  }
}

// 4. Create factory
import { isFeatureEnabled } from '@/lib/feature-flags'

let httpAdapter: IHttpAdapter | null = null

export function getHttpAdapter(): IHttpAdapter {
  if (!httpAdapter) {
    if (isFeatureEnabled('use_fetch_api')) {
      httpAdapter = new FetchHttpAdapter()
    } else {
      httpAdapter = new XhrHttpAdapter()
    }
  }
  return httpAdapter
}

export function createHttpClient() {
  const adapter = getHttpAdapter()
  
  return {
    get: <T>(url: string, options?: RequestOptions) => 
      adapter.get<T>(url, options),
    post: <T>(url: string, data: any, options?: RequestOptions) =>
      adapter.post<T>(url, data, options)
  }
}

// 5. Use in application
export async function fetchUserData(userId: string) {
  const http = createHttpClient()
  
  try {
    const user = await http.get(`/api/users/${userId}`)
    return user
  } catch (error) {
    console.error('Failed to fetch user:', error)
    throw error
  }
}
