// Project_Frontend_Copy/components/ui/markdown.tsx

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeRaw]}
      components={{
        // Customize table rendering to apply Tailwind CSS classes
        table: ({ node, ...props }) => (
          <div className="overflow-x-auto">
            <table className="table-auto w-full my-4" {...props} />
          </div>
        ),
        thead: ({ node, ...props }) => <thead className="bg-gray-100 dark:bg-gray-800" {...props} />,
        tbody: ({ node, ...props }) => <tbody className="divide-y divide-gray-200 dark:divide-gray-700" {...props} />,
        tr: ({ node, ...props }) => <tr className="hover:bg-gray-50 dark:hover:bg-gray-900" {...props} />,
        th: ({ node, ...props }) => <th className="px-4 py-2 text-left font-semibold" {...props} />,
        td: ({ node, ...props }) => <td className="px-4 py-2" {...props} />,
        // Fix for empty src attribute error
        img: ({ node, ...props }) => {
          // Check if src is empty and replace with null to prevent browser warnings
          if (props.src === "") {
            return <img {...props} src={null} alt={props.alt || ""} />;
          }
          return <img {...props} />;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

export default Markdown;
