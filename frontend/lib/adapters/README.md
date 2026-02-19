# Adapters - Quick Start Guide

## Overview

This directory contains abstraction layers for all third-party libraries used in the application. Using these adapters instead of direct library imports makes the codebase more maintainable and allows easy library replacements.

## Why Use Adapters?

✅ **Easy Migration** - Switch libraries without changing application code  
✅ **Graceful Fallbacks** - Automatic fallback if primary library fails  
✅ **Feature Flags** - Control rollouts and A/B testing  
✅ **Type Safety** - Consistent TypeScript interfaces  
✅ **Bundle Optimization** - Lazy loading and code splitting  

## Available Adapters

### 📄 Document Viewer

**Use Instead Of**: Direct `@cyntler/react-doc-viewer` imports

**Example**:
```tsx
import { DocumentViewerAdapter } from '@/lib/adapters/document-viewer'

function MyComponent() {
  return (
    <DocumentViewerAdapter
      documents={[
        { uri: '/document.pdf', fileName: 'document.pdf' }
      ]}
      style={{ height: '600px' }}
    />
  )
}
```

**Features**:
- Supports PDF, Office docs, images
- Automatic fallback to native browser viewers
- Lazy loaded for better performance

---

### 🗄️ State Management

**Use Instead Of**: Direct `zustand` imports

**Example**:
```tsx
import { createStore } from '@/lib/adapters/state-management'

interface UserState {
  name: string
  setName: (name: string) => void
}

const userStore = createStore<UserState>({
  initialState: {
    name: '',
    setName: (name) => userStore.setState({ name })
  },
  persist: {
    name: 'user-storage' // Optional: persist to localStorage
  }
})

// In component:
function UserProfile() {
  const name = userStore.useStore(s => s.name)
  const setName = userStore.useStore(s => s.setName)
  
  return <input value={name} onChange={e => setName(e.target.value)} />
}
```

**Features**:
- Can switch between Zustand, Redux, Jotai, etc.
- Optional persistence
- Type-safe selectors

---

## Feature Flags

Control which adapter implementation is used:

```tsx
import { isFeatureEnabled } from '@/lib/feature-flags'

if (isFeatureEnabled('use_new_document_viewer')) {
  // Use primary implementation
} else {
  // Use fallback
}
```

**Configure** in `lib/feature-flags.ts`:
```typescript
use_new_document_viewer: {
  enabled: true,
  rolloutPercentage: 100, // 0-100
  environments: ['development', 'staging', 'production']
}
```

---

## Adding Your Own Adapter

1. **Create adapter directory**:
   ```
   lib/adapters/my-feature/
     ├── types.ts           # Interfaces
     ├── primary-adapter.ts # Main implementation
     ├── fallback-adapter.ts # Backup implementation
     └── index.ts          # Factory
   ```

2. **Define interface** (`types.ts`):
   ```typescript
   export interface IMyAdapter {
     name: string
     version: string
     doSomething(): Promise<Result>
   }
   ```

3. **Implement adapters** (`primary-adapter.ts`):
   ```typescript
   export class PrimaryAdapter implements IMyAdapter {
     name = 'Primary Library'
     version = '1.0.0'
     
     async doSomething() {
       // Use primary library
     }
   }
   ```

4. **Create factory** (`index.ts`):
   ```typescript
   import { isFeatureEnabled } from '@/lib/feature-flags'
   
   function getAdapter(): IMyAdapter {
     if (isFeatureEnabled('use_my_feature')) {
       return new PrimaryAdapter()
     }
     return new FallbackAdapter()
   }
   
   export function useMyFeature() {
     return getAdapter()
   }
   ```

5. **Add feature flag**:
   ```typescript
   // In lib/feature-flags.ts
   type FeatureFlag = 
     | 'existing_flags'
     | 'use_my_feature'
   
   const FLAGS: Record<FeatureFlag, FeatureFlagConfig> = {
     use_my_feature: {
       enabled: false,
       description: 'Enable my feature',
       environments: ['development']
     }
   }
   ```

---

## Best Practices

### ✅ DO

- **Always use adapters** for third-party libraries
- **Provide fallbacks** for critical functionality
- **Lazy load** heavy dependencies
- **Document** adapter interfaces
- **Test** both primary and fallback implementations

### ❌ DON'T

- Import third-party libraries directly in components
- Skip fallback implementations for critical features
- Bundle large libraries without code splitting
- Change adapter interfaces without migration plan

---

## Migration Example

**Scenario**: Replace `@cyntler/react-doc-viewer` with `react-pdf`

1. **Create new adapter**:
   ```typescript
   // lib/adapters/document-viewer/react-pdf-adapter.tsx
   export class ReactPdfAdapter implements IDocumentViewerAdapter {
     name = 'react-pdf'
     version = '7.0.0'
     
     render(config: DocumentViewerConfig) {
       // Implementation using react-pdf
     }
   }
   ```

2. **Update factory**:
   ```typescript
   // lib/adapters/document-viewer/index.tsx
   function getPrimaryAdapter() {
     if (isFeatureEnabled('use_react_pdf')) {
       return new ReactPdfAdapter()
     }
     return new CyntlerDocumentViewerAdapter()
   }
   ```

3. **Test & rollout**:
   ```typescript
   // Start with 10% of users
   use_react_pdf: {
     enabled: true,
     rolloutPercentage: 10
   }
   
   // Gradually increase to 100%
   // Monitor for errors
   // Remove old adapter when stable
   ```

---

## Debugging

**Check which adapter is active**:
```tsx
import { getDocumentViewerInfo } from '@/lib/adapters'

console.log(getDocumentViewerInfo())
// { name: '@cyntler/react-doc-viewer', version: '1.17.1' }
```

**View all feature flags**:
```tsx
import { getAllFeatures } from '@/lib/feature-flags'

console.table(getAllFeatures())
```

---

## Resources

- [Full Documentation](./LIBRARY_ABSTRACTION.md) - Comprehensive guide
- [Feature Flags](../feature-flags.ts) - Flag configuration
- [Adapter Pattern](https://refactoring.guru/design-patterns/adapter) - Design pattern reference

---

**Questions?** Check the full documentation or ask the team!
