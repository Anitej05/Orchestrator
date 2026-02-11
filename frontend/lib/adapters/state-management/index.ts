/**
 * State Management Factory
 * 
 * Central point for creating state stores
 * Handles adapter selection based on feature flags
 */

import type { StoreConfig, Store, UseStore } from './types'
import { ZustandAdapter } from './zustand-adapter'
import { isFeatureEnabled } from '@/lib/feature-flags'

/**
 * Get the current state management adapter
 */
function getStateAdapter() {
  // In the future, we can switch adapters based on feature flags
  // For now, always use Zustand
  return new ZustandAdapter()
}

/**
 * Create a new state store
 * 
 * @example
 * interface CounterState {
 *   count: number
 *   increment: () => void
 * }
 * 
 * const counterStore = createStore<CounterState>({
 *   initialState: {
 *     count: 0,
 *     increment: () => counterStore.setState(s => ({ count: s.count + 1 }))
 *   }
 * })
 * 
 * // In component:
 * const count = counterStore.useStore(s => s.count)
 */
export function createStore<T>(config: StoreConfig<T>): Store<T> & { useStore: UseStore<T> } {
  const adapter = getStateAdapter()
  return adapter.create(config)
}

/**
 * Get adapter info for debugging
 */
export function getStateAdapterInfo() {
  const adapter = getStateAdapter()
  return {
    name: adapter.name,
    version: adapter.version
  }
}
