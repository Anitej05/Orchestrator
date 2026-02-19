/**
 * Feature Flags System
 * 
 * Centralized feature flag management for gradual rollouts and A/B testing
 * Allows toggling features without code changes
 */

export type FeatureFlag = 
  | 'use_new_document_viewer'
  | 'use_optimized_state_management'
  | 'enable_advanced_analytics'
  | 'enable_experimental_ui'
  | 'use_alternative_markdown_renderer'
  | 'enable_offline_mode'
  | 'use_enhanced_websocket'

interface FeatureFlagConfig {
  enabled: boolean
  description: string
  rolloutPercentage?: number // 0-100, for gradual rollouts
  environments?: ('development' | 'staging' | 'production')[]
}

/**
 * Feature flag configuration
 * Modify these to enable/disable features across the application
 */
const FLAGS: Record<FeatureFlag, FeatureFlagConfig> = {
  use_new_document_viewer: {
    enabled: true,
    description: 'Use the abstracted document viewer with fallback support',
    environments: ['development', 'staging', 'production']
  },
  use_optimized_state_management: {
    enabled: true,
    description: 'Use the abstracted state management layer',
    environments: ['development', 'staging', 'production']
  },
  enable_advanced_analytics: {
    enabled: false,
    description: 'Enable advanced analytics tracking',
    rolloutPercentage: 10,
    environments: ['production']
  },
  enable_experimental_ui: {
    enabled: false,
    description: 'Enable experimental UI components',
    environments: ['development']
  },
  use_alternative_markdown_renderer: {
    enabled: false,
    description: 'Use alternative markdown rendering library',
    rolloutPercentage: 0,
    environments: ['development']
  },
  enable_offline_mode: {
    enabled: false,
    description: 'Enable offline mode with service worker',
    environments: ['development']
  },
  use_enhanced_websocket: {
    enabled: true,
    description: 'Use enhanced websocket with automatic reconnection',
    environments: ['development', 'staging', 'production']
  }
}

/**
 * Get current environment
 */
function getCurrentEnvironment(): 'development' | 'staging' | 'production' {
  if (typeof window === 'undefined') return 'development'
  
  const hostname = window.location.hostname
  if (hostname === 'localhost' || hostname === '127.0.0.1') return 'development'
  if (hostname.includes('staging')) return 'staging'
  return 'production'
}

/**
 * Check if a feature flag is enabled
 * 
 * @param flag - The feature flag to check
 * @returns true if the flag is enabled for the current environment
 * 
 * @example
 * if (isFeatureEnabled('use_new_document_viewer')) {
 *   // Use new document viewer
 * } else {
 *   // Use legacy viewer
 * }
 */
export function isFeatureEnabled(flag: FeatureFlag): boolean {
  const config = FLAGS[flag]
  
  if (!config.enabled) return false
  
  // Check environment restrictions
  if (config.environments) {
    const currentEnv = getCurrentEnvironment()
    if (!config.environments.includes(currentEnv)) return false
  }
  
  // Check rollout percentage
  if (config.rolloutPercentage !== undefined) {
    // Use deterministic selection based on session/user
    // For now, random selection (in production, you'd use user ID hash)
    const random = Math.random() * 100
    return random < config.rolloutPercentage
  }
  
  return true
}

/**
 * Get feature flag configuration
 * Useful for debugging and admin panels
 */
export function getFeatureConfig(flag: FeatureFlag): FeatureFlagConfig {
  return FLAGS[flag]
}

/**
 * Get all feature flags
 * Useful for debugging and admin panels
 */
export function getAllFeatures(): Record<FeatureFlag, FeatureFlagConfig> {
  return FLAGS
}

/**
 * Hook for using feature flags in React components
 */
export function useFeatureFlag(flag: FeatureFlag): boolean {
  // In a real implementation, this might subscribe to flag changes
  // For now, it's a simple wrapper
  return isFeatureEnabled(flag)
}
