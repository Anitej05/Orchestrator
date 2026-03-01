/**
 * Frontend Configuration
 * 
 * Single source of truth for all runtime configuration.
 */

// Get API base URL from environment variable or default to localhost for development
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

