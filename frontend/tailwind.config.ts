import type { Config } from 'tailwindcss';

const config = {
  darkMode: "class",
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './hooks/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx,mdx}',
    './styles/**/*.{css}',
  ],
  safelist: [
    'ui-card',
    'ui-card-hover',
    'ui-input',
    'ui-button-primary',
    'ui-button-secondary',
    'ui-task-pending',
    'ui-badge-active',
    'ui-conversation-container',
    'ui-panel-container',
  ],
  theme: {
    extend: {
      // Orbimesh Design Token Colors - Royal Blue Globals
      colors: {
        // Brand Colors - Royal Blue Primary
        'brand': {
          'primary': 'var(--color-brand-primary)',        // Primary Blue - Main actions, active tabs, links, citations
          'primary-hover': 'var(--color-brand-primary-hover)',  // Hover Blue - Link hovers, button hovers
          'primary-light': 'var(--color-brand-primary-light)',  // Light Blue - User message backgrounds
          // Legacy alias for gradual migration
          'teal': 'var(--color-brand-primary)',
          'teal-hover': 'var(--color-brand-primary-hover)',
          'teal-light': 'var(--color-brand-primary-light)',
        },
        
        // Semantic Status Colors
        'status': {
          // Success/Green - Completed tasks, success states, cost values
          'success': 'var(--color-status-success)',
          'success-light': 'var(--color-status-success-light)',   // Completed task backgrounds
          'success-border': 'var(--color-status-success-border)',  // Completed task borders
          'success-dark': 'var(--color-status-success-dark)',    // Cost text, darker green for contrast
          
          // Active/Blue - In-progress tasks, active workflows, spinners
          'active': 'var(--color-status-active)',
          'active-light': 'var(--color-status-active-light)',    // In-progress backgrounds
          'active-border': 'var(--color-status-active-border)',   // In-progress borders
          'active-dark': 'var(--color-status-active-dark)',     // Status badge text
          
          // Pending/Amber - Pending tasks, warnings
          'pending': 'var(--color-status-pending)',
          'pending-light': 'var(--color-status-pending-light)',   // Pending backgrounds
          'pending-dark': 'var(--color-status-pending-dark)',    // Status badge text
          
          // Error/Red - Failed states, errors
          'error': 'var(--color-status-error)',
          'error-light': 'var(--color-status-error-light)',     // Error backgrounds
        },
        
        // Text Colors - Brand.md Typography
        'text': {
          'primary': 'var(--color-text-primary)',         // Slate 900 - Headings, user messages, task names
          'secondary': 'var(--color-text-secondary)',       // Slate 700 - Agent responses, descriptions
          'tertiary': 'var(--color-text-tertiary)',        // Slate 500 - Labels, metadata, inactive tabs, list items
          'disabled': 'var(--color-text-disabled)',        // Slate 400 - Disabled states
        },
        
        // Background Colors
        'bg': {
          'page': 'var(--color-bg-page)',            // Slate 50 - Main app background, code blocks
          'card': 'var(--color-bg-card)',            // White - Right panel, task cards, elevated surfaces
          'subtle': 'var(--color-bg-subtle)',          // Slate 50 - Metadata items, file items, summary cards
          'hover': 'var(--color-bg-hover)',           // Light blue - File hover, interactive items
        },
        
        // Border Colors - Brand.md Component Matrix
        'border-color': {
          'light': 'var(--color-border-light)',           // Slate 100 - Subtle dividers
          'DEFAULT': 'var(--color-border-DEFAULT)',         // Slate 200 - Standard borders, panel separator, tab separator
          'medium': 'var(--color-border-medium)',          // Slate 300 - Emphasized borders, pending tasks
        },
        
        // Interactive/Accent
        'citation': {
          'bg': 'rgba(44, 75, 168, 0.1)',         // 10% blue opacity
          'bg-hover': 'rgba(44, 75, 168, 0.15)',  // 15% blue opacity
        },
      },

      // Typography Tokens
      fontFamily: {
        'orbimesh-ui': [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'sans-serif',
        ],
        'orbimesh-mono': [
          'JetBrains Mono',
          'monospace',
        ],
      },

      fontSize: {
        // Conversation Panel Sizes
        'orbimesh-user': ['15px', { lineHeight: '24px', letterSpacing: '-0.01em', fontWeight: '600' }],
        'orbimesh-agent': ['15px', { lineHeight: '24px', letterSpacing: '0', fontWeight: '400' }],
        'orbimesh-h2': ['18px', { lineHeight: '1.3', fontWeight: '600' }],
        'orbimesh-h3': ['16px', { lineHeight: '1.3', fontWeight: '600' }],
        'orbimesh-citation': ['13px', { lineHeight: '1.4', fontWeight: '500' }],
        'orbimesh-code': ['14px', { lineHeight: '1.5' }],
        
        // Right Panel Sizes
        'orbimesh-tab': ['14px', { lineHeight: '1.4', fontWeight: '500' }],
        'orbimesh-section-header': ['16px', { lineHeight: '1.3', fontWeight: '600' }],
        'orbimesh-section-subtitle': ['13px', { lineHeight: '1.4', fontWeight: '400' }],
        'orbimesh-task-name': ['14px', { lineHeight: '1.4', fontWeight: '600' }],
        'orbimesh-task-description': ['13px', { lineHeight: '1.4', fontWeight: '400' }],
        'orbimesh-metadata-label': ['12px', { lineHeight: '1.4', fontWeight: '500' }],
        'orbimesh-metadata-value': ['13px', { lineHeight: '1.4', fontWeight: '400' }],
        'orbimesh-metadata-mono': ['11px', { lineHeight: '1.4', fontWeight: '400', fontFamily: 'JetBrains Mono' }],
        'orbimesh-badge': ['11px', { lineHeight: '1.2', fontWeight: '500', letterSpacing: '0.02em' }],
        'orbimesh-file-name': ['13px', { lineHeight: '1.4', fontWeight: '500' }],
        'orbimesh-file-meta': ['11px', { lineHeight: '1.4', fontWeight: '400' }],
      },

      fontWeight: {
        'orbimesh-regular': '400',
        'orbimesh-medium': '500',
        'orbimesh-semibold': '600',
      },

      lineHeight: {
        'orbimesh-body': '1.6',     // 24px for 15px text
        'orbimesh-heading': '1.3',
        'orbimesh-small': '1.4',
        'orbimesh-compact': '1.2',
      },

      letterSpacing: {
        'orbimesh-tight': '-0.01em',
        'orbimesh-normal': '0',
        'orbimesh-loose': '0.02em',
        'orbimesh-uppercase': '0.05em',
      },

      // Spacing Tokens (based on 12px unit system)
      spacing: {
        'orbimesh-xs': '4px',         // 1/3 unit
        'orbimesh-sm': '8px',         // 2/3 unit
        'orbimesh-md': '12px',        // 1 unit
        'orbimesh-lg': '16px',        // 1.33 units
        'orbimesh-xl': '20px',        // 1.67 units
        'orbimesh-2xl': '24px',       // 2 units
        'orbimesh-3xl': '40px',       // 3.33 units
        // Component-specific semantic tokens
        'ui-message-gap': '12px',        // Between user/agent messages
        'ui-breathing-gap': '24px',      // Between conversation turns
        'ui-card-padding': '16px',       // Standard card padding
        'ui-button-padding-x': '16px',   // Button horizontal padding
        'ui-button-padding-y': '8px',    // Button vertical padding
      },

      // Border Radius Tokens
      borderRadius: {
        'orbimesh-sm': '4px',         // Citations, badges, buttons
        'orbimesh-md': '6px',         // Code blocks
        'orbimesh-lg': '8px',         // Task cards
        'orbimesh-xl': '12px',        // User messages
        'orbimesh-full': '50%',       // Agent icons (circles)
      },

      // Component-Specific Spacing
      gap: {
        'orbimesh-message': '12px',   // User → Agent gap
        'orbimesh-breathing': '24px', // Agent → User gap
        'orbimesh-paragraph': '16px', // Between paragraphs
        'orbimesh-list-item': '8px',  // Between list items
      },

      padding: {
        'orbimesh-message': '12px 16px',        // User message padding
        'orbimesh-badge': '4px 8px',            // Status badge padding
        'orbimesh-tab': '12px 20px',            // Tab padding
        'orbimesh-task-card': '12px 16px',      // Task card padding
        'orbimesh-metadata-item': '12px',       // Metadata item padding
        'orbimesh-file-item': '12px',           // File item padding
        'orbimesh-tab-content': '24px',         // Tab content area
        'orbimesh-container': '40px 20px',      // Container padding (vertical horizontal)
      },

      maxWidth: {
        'orbimesh-conversation': '720px',       // Optimal reading length
        'orbimesh-message': '75%',              // User message max width
      },

      width: {
        'orbimesh-panel': '400px',              // Right panel width
      },

      // Shadow Tokens
      boxShadow: {
        'orbimesh-panel': '-2px 0 8px rgba(0, 0, 0, 0.04)',
        'orbimesh-card-hover': '0 4px 12px rgba(0, 0, 0, 0.08)',
      },

      // Focus Ring Token
      outline: {
        'orbimesh-focus': '3px solid rgba(6, 182, 212, 0.1)',
      },
    },
  },
  plugins: [],
};

export default config;
