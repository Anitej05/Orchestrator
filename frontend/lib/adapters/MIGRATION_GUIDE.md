# Migration Guide: Adopting Adapter Pattern

## Current Status

The adapter pattern has been implemented for:
- ✅ Document Viewer (`@cyntler/react-doc-viewer`)
- ✅ State Management (`zustand`)
- ✅ Feature Flags System

## Recommended Migration Order

### Priority 1: Critical Dependencies (Immediate)

These libraries are essential and should be abstracted first:

#### 1. Document Viewer ✅ DONE
**Status**: Complete  
**Location**: `lib/adapters/document-viewer/`  
**Action**: Already implemented and integrated

#### 2. State Management ✅ DONE
**Status**: Complete  
**Location**: `lib/adapters/state-management/`  
**Action**: Already implemented, need to migrate existing stores

**Migration Steps**:
```bash
# Find all Zustand stores
# Replace direct zustand imports with adapter
```

**Example**:
```typescript
// Before
import { create } from 'zustand'

const useStore = create((set) => ({
  count: 0,
  increment: () => set(state => ({ count: state.count + 1 }))
}))

// After
import { createStore } from '@/lib/adapters/state-management'

const useStore = createStore({
  initialState: {
    count: 0,
    increment: () => useStore.setState(s => ({ count: s.count + 1 }))
  }
})
```

---

### Priority 2: UI Libraries (Short-term)

#### 3. Markdown Renderer
**Current**: `react-markdown`  
**Estimated Effort**: 2-3 hours  
**Priority**: Medium

**Why**: 
- Used extensively in chat interface
- Potential bundle size optimization
- Alternative libraries available (marked, markdown-it)

**Implementation Plan**:
1. Create `lib/adapters/markdown-renderer/`
2. Implement react-markdown adapter
3. Create simple fallback (dangerouslySetInnerHTML with sanitization)
4. Add feature flag `use_advanced_markdown`
5. Migrate chat components

**Benefits**:
- Easy to test different renderers
- Fallback for when full markdown not needed
- Bundle size optimization possible

---

#### 4. Animation Library
**Current**: `framer-motion`  
**Estimated Effort**: 3-4 hours  
**Priority**: Low-Medium

**Why**:
- Large bundle impact
- Alternative: CSS animations, react-spring
- Not critical for functionality

**Implementation Plan**:
1. Create `lib/adapters/animation/`
2. Implement framer-motion adapter
3. Create CSS-only fallback
4. Add feature flag `use_advanced_animations`
5. Gradually migrate animations

**Benefits**:
- Reduce bundle size with CSS fallback
- Easy to switch to react-spring if needed
- Performance optimization options

---

### Priority 3: Data Libraries (Medium-term)

#### 5. Chart Library
**Current**: `recharts`  
**Estimated Effort**: 4-5 hours  
**Priority**: Low

**Why**:
- Large bundle size
- Alternatives: Chart.js, Victory, Nivo
- Only used in metrics/dashboard

**Implementation Plan**:
1. Create `lib/adapters/charts/`
2. Implement Recharts adapter
3. Create simple SVG fallback
4. Add feature flag `use_advanced_charts`
5. Migrate dashboard components

**Benefits**:
- Lazy load only when needed
- Switch to lighter library if needed
- Simple fallback for basic charts

---

#### 6. Flow/Diagram Library
**Current**: `reactflow`  
**Estimated Effort**: 5-6 hours  
**Priority**: Low

**Why**:
- Specialized use case (plan visualization)
- Heavy dependency
- Critical for UX but not frequently used

**Implementation Plan**:
1. Create `lib/adapters/flow-diagram/`
2. Implement reactflow adapter
3. Create simple SVG/Canvas fallback
4. Add feature flag `use_advanced_flow`
5. Lazy load in plan visualization

**Benefits**:
- Only load when viewing plans
- Alternative implementations possible
- Fallback for simple visualizations

---

### Priority 4: Utility Libraries (Long-term)

#### 7. Date Library
**Current**: `date-fns`  
**Estimated Effort**: 2-3 hours  
**Priority**: Very Low

**Why**:
- Tree-shakeable already
- Alternatives: dayjs, luxon
- Small impact

**Implementation Plan**:
1. Create `lib/adapters/date-utils/`
2. Implement date-fns adapter
3. Create native Date fallback
4. Migrate date operations gradually

---

#### 8. Form Library
**Current**: `react-hook-form`  
**Estimated Effort**: 6-8 hours  
**Priority**: Very Low

**Why**:
- Works well
- Large migration effort
- Low priority unless issues arise

**Implementation Plan**:
- Only implement if migrating away from react-hook-form
- Keep as direct dependency for now

---

## Implementation Checklist

For each adapter you create:

- [ ] Create adapter directory structure
- [ ] Define TypeScript interfaces
- [ ] Implement primary adapter
- [ ] Implement fallback adapter
- [ ] Create factory with error handling
- [ ] Add feature flag
- [ ] Write unit tests
- [ ] Document usage
- [ ] Migrate existing code
- [ ] Monitor in production

---

## Testing Strategy

### Unit Tests
```typescript
// Test both adapters work
describe('DocumentViewerAdapter', () => {
  it('should render with primary adapter', () => {
    // Test primary
  })
  
  it('should fallback on error', () => {
    // Test fallback behavior
  })
  
  it('should handle different file types', () => {
    // Test file type support
  })
})
```

### Integration Tests
```typescript
// Test feature flag switching
describe('Feature Flag Integration', () => {
  it('should use primary when flag enabled', () => {
    // Enable flag, verify primary adapter used
  })
  
  it('should use fallback when flag disabled', () => {
    // Disable flag, verify fallback used
  })
})
```

### E2E Tests
```typescript
// Test user workflows
describe('Document Viewing', () => {
  it('should display PDF correctly', () => {
    // Upload PDF, verify it displays
  })
  
  it('should handle viewer errors gracefully', () => {
    // Force error, verify fallback works
  })
})
```

---

## Rollout Strategy

### Phase 1: Development Testing (Week 1)
- Enable all new adapters in development
- Test thoroughly with different file types
- Fix any issues

### Phase 2: Staging Deployment (Week 2)
- Deploy to staging environment
- Run full test suite
- Monitor for errors

### Phase 3: Gradual Production Rollout (Weeks 3-4)
```typescript
// Week 3: 10% of users
use_new_document_viewer: {
  enabled: true,
  rolloutPercentage: 10,
  environments: ['production']
}

// Week 4: 50% of users
rolloutPercentage: 50

// Week 5: 100% of users
rolloutPercentage: 100
```

### Phase 4: Cleanup (Week 5)
- Remove old implementations
- Update documentation
- Archive old code

---

## Monitoring

### Metrics to Track
- Error rates before/after migration
- Performance metrics (load time, render time)
- User feedback
- Bundle size changes

### Error Handling
```typescript
// Log adapter errors for monitoring
const adapter = getAdapter()
try {
  adapter.render(config)
} catch (error) {
  console.error('Adapter error:', {
    adapter: adapter.name,
    version: adapter.version,
    error: error.message
  })
  // Send to error tracking service
}
```

---

## Rollback Plan

If issues arise:

1. **Immediate**: Disable feature flag
   ```typescript
   use_new_document_viewer: {
     enabled: false  // ← Instant rollback
   }
   ```

2. **Short-term**: Reduce rollout percentage
   ```typescript
   rolloutPercentage: 0  // Back to 0%
   ```

3. **Long-term**: Revert code changes if needed
   ```bash
   git revert <commit-hash>
   ```

---

## Success Criteria

Adapter implementation is successful when:

✅ No increase in error rates  
✅ Performance maintained or improved  
✅ Bundle size same or reduced  
✅ User experience unchanged or better  
✅ Code is more maintainable  
✅ Team understands new patterns  

---

## Next Steps

1. ✅ Complete document viewer migration
2. ✅ Complete state management abstraction
3. 🔄 Migrate existing Zustand stores (in progress)
4. ⏭️ Implement markdown renderer adapter (next)
5. ⏭️ Implement animation adapter (future)

---

## Team Training

Schedule training sessions:
- Week 1: Overview of adapter pattern
- Week 2: Hands-on creating adapters
- Week 3: Migration strategies
- Week 4: Monitoring and rollback

---

## Questions?

Contact the architecture team or refer to:
- [Full Documentation](./LIBRARY_ABSTRACTION.md)
- [Quick Start Guide](./README.md)
- [Code Examples](./EXAMPLES.ts)

---

**Last Updated**: February 12, 2026  
**Next Review**: March 12, 2026
