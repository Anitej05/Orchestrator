// Project_Frontend_Copy/components/ui/markdown.tsx

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import rehypeRaw from "rehype-raw";
import { FC } from "react";

interface MarkdownProps {
  content: string;
}

/**
 * A component to render markdown content.
 * It supports GitHub Flavored Markdown (GFM) for tables and allows raw HTML.
 * @param {MarkdownProps} props The props for the component.
 * @param {string} props.content The markdown string to render.
 * @returns {JSX.Element} The rendered markdown content.
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
      <div className="whitespace-pre-wrap font-mono text-sm">
        {content}
      </div>
    );
  }

  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        rehypePlugins={[rehypeRaw]}
        components={{
          // Headings
          h1: ({ node, ...props }) => (
            <h1 className="text-2xl font-bold mt-6 mb-4 text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700 pb-2" {...props} />
          ),
          h2: ({ node, ...props }) => (
            <h2 className="text-xl font-bold mt-5 mb-3 text-gray-900 dark:text-gray-100" {...props} />
          ),
          h3: ({ node, ...props }) => (
            <h3 className="text-lg font-semibold mt-4 mb-2 text-gray-900 dark:text-gray-100" {...props} />
          ),
          h4: ({ node, ...props }) => (
            <h4 className="text-base font-semibold mt-3 mb-2 text-gray-900 dark:text-gray-100" {...props} />
          ),
          h5: ({ node, ...props }) => (
            <h5 className="text-sm font-semibold mt-2 mb-1 text-gray-900 dark:text-gray-100" {...props} />
          ),
          h6: ({ node, ...props }) => (
            <h6 className="text-sm font-semibold mt-2 mb-1 text-gray-700 dark:text-gray-300" {...props} />
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
              return <div className="my-3 leading-relaxed text-gray-800 dark:text-gray-200" {...props}>{children}</div>;
            }

            return <p className="my-3 leading-relaxed text-gray-800 dark:text-gray-200" {...props}>{children}</p>;
          },

          // Lists
          ul: ({ node, ...props }) => (
            <ul className="my-3 ml-6 space-y-2 list-disc marker:text-gray-500 dark:marker:text-gray-400" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="my-3 ml-6 space-y-2 list-decimal marker:text-gray-500 dark:marker:text-gray-400" {...props} />
          ),
          li: ({ node, ...props }) => (
            <li className="text-gray-800 dark:text-gray-200 leading-relaxed" {...props} />
          ),

          // Links
          a: ({ node, ...props }) => (
            <a
              className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline underline-offset-2 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
              {...props}
            />
          ),

          // Blockquotes
          blockquote: ({ node, ...props }) => (
            <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 py-2 my-4 italic text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-800/50 rounded-r" {...props} />
          ),

          // Horizontal rule
          hr: ({ node, ...props }) => (
            <hr className="my-6 border-gray-300 dark:border-gray-700" {...props} />
          ),

          // Strong/Bold
          strong: ({ node, ...props }) => (
            <strong className="font-bold text-gray-900 dark:text-gray-100" {...props} />
          ),

          // Emphasis/Italic
          em: ({ node, ...props }) => (
            <em className="italic text-gray-800 dark:text-gray-200" {...props} />
          ),

          // Tables
          table: ({ node, ...props }) => (
            <div className="overflow-x-auto my-4 rounded-lg border border-gray-200 dark:border-gray-700">
              <table className="table-auto w-full" {...props} />
            </div>
          ),
          thead: ({ node, ...props }) => (
            <thead className="bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700" {...props} />
          ),
          tbody: ({ node, ...props }) => (
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700" {...props} />
          ),
          tr: ({ node, ...props }) => (
            <tr className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors" {...props} />
          ),
          th: ({ node, ...props }) => (
            <th className="px-4 py-3 text-left font-semibold text-gray-900 dark:text-gray-100 text-sm" {...props} />
          ),
          td: ({ node, ...props }) => (
            <td className="px-4 py-3 text-gray-800 dark:text-gray-200 text-sm" {...props} />
          ),

          // Images
          img: ({ node, ...props }) => {
            if (props.src === "") {
              return <img {...props} src="" alt={props.alt || ""} />;
            }
            return (
              <img
                className="rounded-lg my-4 max-w-full h-auto shadow-md"
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

            const className = codeElement?.props?.className || '';
            const language = className.replace('language-', '');

            return (
              <div className="my-4 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                {language && (
                  <div className="bg-gray-100 dark:bg-gray-800 px-4 py-2 text-xs font-mono text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700 font-semibold uppercase tracking-wider flex items-center justify-between">
                    <span>{language}</span>
                  </div>
                )}
                <pre className={`overflow-x-auto p-4 bg-gray-50 dark:bg-gray-950 text-sm leading-relaxed text-gray-800 dark:text-gray-100 ${!language ? 'rounded-lg' : ''}`} {...props}>
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
              return <code className={`${className} font-mono text-sm`} {...props}>{children}</code>
            }

            // For no-language code (inline OR generic block), apply inline styles.
            // If it IS a generic block, the parent `pre` (defined above) provides the container.
            // The `code` inside will look like an inline pill. To prevent this, we can add a class to `pre` that resets child code styles?
            // See the `pre` implementation: it doesn't pass a clear signal.
            // Let's just return a styled span-like code for inline.

            return (
              <code
                className={`bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded px-1.5 py-0.5 text-sm font-mono border border-gray-200 dark:border-gray-700 ${className || ''}`}
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
