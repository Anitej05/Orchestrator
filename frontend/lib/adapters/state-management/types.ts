/**
 * State Management Adapter
 * 
 * Abstraction layer for state management libraries
 * Currently uses Zustand, but can be swapped for Redux, Jotai, or others
 * 
 * Benefits:
 * - Easy migration between state management solutions
 * - Consistent API across the application
 * - Type-safe state management
 * - Feature flag support for gradual rollouts
 */

export interface StoreConfig<T> {
  initialState: T
  persist?: {
    name: string
    storage?: Storage
  }
}

export interface Store<T> {
  getState: () => T
  setState: (partial: Partial<T> | ((state: T) => Partial<T>)) => void
  subscribe: (listener: (state: T, prevState: T) => void) => () => void
  destroy: () => void
}

export interface StoreAdapter {
  create<T>(config: StoreConfig<T>): Store<T>
  name: string
  version: string
}

/**
 * Hook type for React components
 */
export type UseStore<T> = {
  (): T
  <U>(selector: (state: T) => U): U
  <U>(selector: (state: T) => U, equality?: (a: U, b: U) => boolean): U
}
