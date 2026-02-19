/**
 * Zustand State Management Adapter
 * 
 * Wrapper around Zustand for easy replacement
 */

import { create as zustandCreate } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { StoreAdapter, StoreConfig, Store, UseStore } from './types'

export class ZustandAdapter implements StoreAdapter {
  name = 'Zustand'
  version = '5.0.8'

  create<T>(config: StoreConfig<T>): Store<T> & { useStore: UseStore<T> } {
    // Create base store
    let useStore: any

    if (config.persist) {
      // Create persisted store
      useStore = zustandCreate<T>()(
        persist(
          () => config.initialState,
          {
            name: config.persist.name,
            storage: config.persist.storage 
              ? createJSONStorage(() => config.persist!.storage!)
              : createJSONStorage(() => localStorage)
          }
        )
      )
    } else {
      // Create non-persisted store
      useStore = zustandCreate<T>(() => config.initialState)
    }

    return {
      getState: () => useStore.getState(),
      setState: (partial) => {
        if (typeof partial === 'function') {
          const currentState = useStore.getState()
          const updates = partial(currentState)
          useStore.setState(updates)
        } else {
          useStore.setState(partial)
        }
      },
      subscribe: (listener) => {
        return useStore.subscribe((state: T, prevState: T) => {
          listener(state, prevState)
        })
      },
      destroy: () => {
        // Zustand doesn't have built-in destroy, but we can reset
        useStore.setState(config.initialState, true)
      },
      useStore
    }
  }
}
