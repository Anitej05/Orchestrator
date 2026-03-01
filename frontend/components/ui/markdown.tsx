// Project_Frontend_Copy/components/ui/markdown.tsx
'use client';

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import rehypeRaw from "rehype-raw";
import React, { FC } from "react";

interface MarkdownProps {
  content: string;
}

/**
 * Markdown renderer following Brand.md conversation formatting specs.
 * Renders assistant responses with Orbimesh design system styling.
 * 
 * Brand.md compliance:
 * - Typography: Inter Regular 15px body, Inter Semibold headings
 * - Colors: Teal (#0D9488) for links/citations, Slate palette for text
 * - Spacing: 16px paragraphs, 8px list items, 24px H2 margins
 * - Code: JetBrains Mono 14px, #F8FAFC backgrounds
 * - Citations: [1] format with .ui-citation styling
 */
const Markdown: FC<MarkdownProps> = ({ content }) => {
  // Check if content contains HTML that should not be processed by Markdown
  const containsHtml = content.includes('<!DOCTYPE html>') ||
    content.includes('<html') ||
    content.includes('<button') ||
    content.includes('<script>') ||
    content.includes('onClick=') ||
    content.includes('onclick=');

  if (containsHtml) {
    // For HTML content, render as plain text to avoid React parsing issues
    return (
      <div className="whitespace-pre-wrap font-mono text-[14px] text-text-secondary">
        {content}
      </div>
    );
  }

  const extractText = (child: React.ReactNode): string => {
    if (typeof child === 'string') return child;
    if (Array.isArray(child)) return child.map(extractText).join('');
    if (React.isValidElement(child)) return extractText((child.props as any).children);
    return '';
  };

  const isLikelyCodeBlock = (text: string) => {
    // Don't treat as code block if it contains markdown code blocks (triple backticks)
    // This allows properly formatted markdown to be parsed correctly
    if (text.includes('```')) {
      return false;
    }

    const lines = text.split('\n').filter((line) => line.trim().length > 0);
    if (lines.length < 3) return false;

    const codeIndicators = [/^#include\b/, /;\s*$/, /\{\s*$/, /\}\s*$/, /\bint\b|\bvoid\b|\bclass\b|\breturn\b/];
    const indicatorHits = lines.reduce((count, line) => {
      const trimmed = line.trim();
      return count + (codeIndicators.some((rx) => rx.test(trimmed)) ? 1 : 0);
    }, 0);

    return indicatorHits >= 2;
  };

  if (isLikelyCodeBlock(content)) {
    return (
      <pre className="ui-markdown-code-block whitespace-pre-wrap">
        {content}
      </pre>
    );
  }

  return (
    <div className="ui-markdown-content max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          // Headings
          h1: ({ node, ...props }) => (
            <h1 className="ui-markdown-h1" {...props} />
          ),
          h2: ({ node, ...props }) => (
            <h2 className="ui-markdown-h2" {...props} />
          ),
          h3: ({ node, ...props }) => (
            <h3 className="ui-markdown-h3" {...props} />
          ),
          h4: ({ node, ...props }) => (
            <h4 className="ui-markdown-h4" {...props} />
          ),
          h5: ({ node, ...props }) => (
            <h5 className="ui-markdown-h5" {...props} />
          ),
          h6: ({ node, ...props }) => (
            <h6 className="ui-markdown-h6" {...props} />
          ),

          // Paragraphs - check for block-level children to avoid nesting errors
          p: ({ node, children, ...props }) => {
            // Recursively check if any descendant contains block-level elements
            const hasBlockDescendant = (n: any): boolean => {
              if (!n?.children) return false;
              return n.children.some((child: any) => {
                if (child.type === 'element') {
                  // Check for block-level elements or code elements (which become divs)
                  if (['div', 'pre', 'table', 'code'].includes(child.tagName)) {
                    return true;
                  }
                  // Recursively check children
                  return hasBlockDescendant(child);
                }
                return false;
              });
            };

            // If it has block descendants, render as div to avoid HTML nesting errors
            if (hasBlockDescendant(node)) {
              return <div className="ui-markdown-paragraph" {...props}>{children}</div>;
            }

            return <p className="ui-markdown-paragraph" {...props}>{children}</p>;
          },

          // Lists
          ul: ({ node, ...props }) => (
            <ul className="ui-markdown-ul" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="ui-markdown-ol" {...props} />
          ),
          li: ({ node, ...props }) => (
            <li className="ui-markdown-li" {...props} />
          ),

          // Links
          a: ({ node, children, ...props }) => {
            const text = extractText(children).trim();
            const isCitation = /^\[\d+\]$/.test(text);
            return (
              <a
                className={isCitation ? "ui-citation" : "ui-markdown-link"}
                target="_blank"
                rel="noopener noreferrer"
                {...props}
              >
                {children}
              </a>
            );
          },

          // Blockquotes
          blockquote: ({ node, ...props }) => (
            <blockquote className="ui-markdown-blockquote" {...props} />
          ),

          // Horizontal rule
          hr: ({ node, ...props }) => (
            <hr className="ui-markdown-hr" {...props} />
          ),

          // Strong/Bold
          strong: ({ node, ...props }) => (
            <strong className="ui-markdown-strong" {...props} />
          ),

          // Emphasis/Italic
          em: ({ node, ...props }) => (
            <em className="ui-markdown-em" {...props} />
          ),

          // Tables
          table: ({ node, ...props }) => (
            <div className="ui-markdown-table-wrapper">
              <table className="ui-markdown-table" {...props} />
            </div>
          ),
          thead: ({ node, ...props }) => (
            <thead className="ui-markdown-thead" {...props} />
          ),
          tbody: ({ node, ...props }) => (
            <tbody className="ui-markdown-tbody" {...props} />
          ),
          tr: ({ node, ...props }) => (
            <tr className="ui-markdown-tr" {...props} />
          ),
          th: ({ node, ...props }) => (
            <th className="ui-markdown-th" {...props} />
          ),
          td: ({ node, ...props }) => (
            <td className="ui-markdown-td" {...props} />
          ),

          // Images
          img: ({ node, ...props }) => {
            if (props.src === "") {
              return <img {...props} src="" alt={props.alt || ""} />;
            }
            return (
              <img
                className="ui-markdown-img"
                {...props}
              />
            );
          },

          // Pre tag (handles code blocks)
          pre: ({ children, ...props }) => {
            // Extract language from the code element inside
            const codeElement = React.Children.toArray(children).find(
              (child) => React.isValidElement(child) && (child.type === 'code' || (child.props as any)?.className?.includes('language-'))
            ) as React.ReactElement | undefined;

            const className = (codeElement?.props as any)?.className || '';
            const language = className.replace('language-', '');

            return (
              <div className="ui-markdown-code-block-wrapper">
                {language && (
                  <div className="ui-markdown-code-language-label">
                    <span>{language}</span>
                  </div>
                )}
                <pre
                  className={`ui-markdown-code-block ${language ? 'ui-markdown-code-block-with-label' : ''}`}
                  {...props}
                >
                  {children}
                </pre>
              </div>
            )
          },

          // Code tag (handles inline code, and the inner text of blocks)
          code: ({ node, className, children, ...props }: any) => {
            const match = /language-(\w+)/.exec(className || '');
            const isBlock = !!match; // If it has a language class, it's inside a pre (handled above)

            // If it's a block (inside pre), render plain code (pre handles styling)
            // We use !isBlock to apply inline styles. 
            // Note: Generic blocks (no language) will get inline styles, but because they are inside <pre>, 
            // the pre's layout will behave like a block. 
            // We might double-style (gray background inside gray background).
            // To fix generic blocks, we can check if we are in a pre context, but that's hard.
            // Simple fix: rely on `pre` styles in CSS or just accept pill-style for generic blocks for now (better than full width inline).
            // Actually, we can assume that if it's rendered by react-markdown v9+, blocks are always in pre.
            // Inline code is NOT in pre.
            // But we don't know parent here.

            // However, we can use a trick: standard inline code usually doesn't have className="language-xyz".

            if (isBlock) {
              return <code className={`${className} font-mono text-[14px]`} {...props}>{children}</code>
            }

            // For no-language code (inline OR generic block), apply inline styles.
            // If it IS a generic block, the parent `pre` (defined above) provides the container.
            // The `code` inside will look like an inline pill. To prevent this, we can add a class to `pre` that resets child code styles?
            // See the `pre` implementation: it doesn't pass a clear signal.
            // Let's just return a styled span-like code for inline.

            return (
              <code
                className={`ui-markdown-code-inline ${className || ''}`}
                {...props}
              >
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default Markdown;
