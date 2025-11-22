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
          
          // Paragraphs
          p: ({ node, ...props }) => (
            <p className="my-3 leading-relaxed text-gray-800 dark:text-gray-200" {...props} />
          ),
          
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
          
          // Code blocks
          code({node, className, children, ...props}: any) {
            const language = className ? className.replace('language-', '') : '';
            const isInline = props.inline;
            return !isInline ? (
              <div className="my-4">
                {language && (
                  <div className="bg-gray-800 dark:bg-gray-900 text-gray-400 text-xs px-4 py-2 rounded-t-lg border-b border-gray-700">
                    {language}
                  </div>
                )}
                <pre className={`overflow-x-auto max-w-full ${language ? 'rounded-b-lg' : 'rounded-lg'} bg-gray-900 dark:bg-black text-gray-100 p-4 text-sm leading-relaxed`}>
                  <code className={className} {...props}>
                    {children}
                  </code>
                </pre>
              </div>
            ) : (
              <code className="bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded px-1.5 py-0.5 text-sm font-mono" {...props}>
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
