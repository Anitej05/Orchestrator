# Document Preview System

A comprehensive document preview system for Next.js that supports **PDF, DOCX, Excel, CSV, JSON, and images**.

## 📋 Supported File Types

| Format | Extensions | Preview Method |
|--------|-----------|----------------|
| **PDF** | `.pdf` | @react-pdf-viewer/core with full UI (zoom, pagination, search) |
| **Word Documents** | `.docx` | mammoth.js (client-side HTML conversion for privacy) |
| **Excel Spreadsheets** | `.xlsx`, `.xls` | SheetJS (xlsx) with multi-sheet support and navigation |
| **CSV Files** | `.csv` | PapaParse with table rendering |
| **JSON Files** | `.json` | Native JSON parsing with syntax highlighting |
| **Images** | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.svg`, `.bmp` | Native browser image rendering |
| **Text Files** | `.txt`, `.md`, `.log` | Basic text preview (can be enhanced) |

## 🏗️ Architecture

### Component Structure

```
components/
├── document-viewer.tsx          # Main routing component
└── preview/
    ├── index.ts                 # Barrel export
    ├── pdf-preview.tsx          # PDF viewer (@react-pdf-viewer)
    ├── docx-preview.tsx         # DOCX converter (mammoth.js)
    ├── excel-preview.tsx        # Excel viewer (SheetJS)
    ├── csv-preview.tsx          # CSV viewer (PapaParse)
    └── json-preview.tsx         # JSON viewer (syntax highlighted)
```

### Key Features

1. **Smart File Type Detection**
   - Extension-based detection (more reliable)
   - MIME type fallback
   - Automatic routing to appropriate preview component

2. **Dynamic Imports**
   - All preview components use Next.js `dynamic()` with `ssr: false`
   - Prevents SSR issues with browser-only libraries
   - Reduces initial bundle size

3. **Privacy & Security**
   - DOCX conversion happens client-side (mammoth.js)
   - No external APIs required for file processing
   - Files never leave the user's browser for preview

4. **Viewport Locking**
   - Root-level `overflow-hidden` on `<html>` and `<body>`
   - Prevents horizontal scrollbar issues
   - Proper containment of canvas and document content

## 📦 Dependencies

```json
{
  "dependencies": {
    "@react-pdf-viewer/core": "^3.12.0",
    "@react-pdf-viewer/default-layout": "^3.12.0",
    "mammoth": "^1.6.0",
    "xlsx": "^0.18.5",
    "papaparse": "^5.4.1"
  }
}
```

## 🚀 Usage

### Basic Implementation

```tsx
import DocumentViewer from "@/components/document-viewer"

function MyComponent() {
  const [viewingFile, setViewingFile] = useState(null)

  return (
    <>
      {viewingFile ? (
        <DocumentViewer
          file={{
            name: "example.pdf",
            type: "application/pdf",
            content: "https://example.com/file.pdf", // or blob URL
            file_path: "optional/backend/path"
          }}
          onBack={() => setViewingFile(null)}
        />
      ) : (
        <button onClick={() => setViewingFile(myFile)}>
          View Document
        </button>
      )}
    </>
  )
}
```

### File Object Structure

```typescript
interface FileProps {
  name: string          // File nameX with extension
  type: string          // MIME type (e.g., "application/pdf")
  content?: string      // URL, blob URL, or base64 data
  file_path?: string    // Backend file path (will construct API URL)
}
```

### URL Resolution Priority

1. **HTTP/HTTPS URL**: Used directly if content starts with `http://` or `https://`
2. **Backend file_path**: Constructs `/api/files/{file_path}` URL
3. **API_BASE_URL**: Used if content already contains API base URL
4. **Fallback**: Uses content as-is

## 🎨 Customization

### Styling

All preview components use Tailwind CSS with your design system tokens:
- `bg-bg-card`, `bg-bg-subtle`
- `text-text-primary`, `text-text-secondary`
- `border-border-color`
- Custom scrollbar styles from globals.css

### Adding New File Types

1. Create new preview component in `components/preview/`:

```tsx
"use client"

export default function MyPreview({ fileUrl }: { fileUrl: string }) {
  // Your preview logic
  return <div>Preview content</div>
}
```

2. Add to `document-viewer.tsx`:

```tsx
// Import
const MyPreview = dynamic(() => import("@/components/preview/my-preview"), { ssr: false })

// Add to getFileType()
if (ext === 'myext') return 'mytype'

// Add to renderPreview()
case 'mytype':
  return <MyPreview fileUrl={documentUri} />
```

## 🔒 Security Considerations

### Client-Side Processing
- **DOCX**: Converted client-side (no server upload needed)
- **PDF**: Rendered client-side using PDF.js
- **Excel/CSV**: Parsed client-side

### Private/Local Files

For files on private servers or local storage:

```tsx
// Option 1: Blob URL
const blobUrl = URL.createObjectURL(fileBlob)
<DocumentViewer file={{ content: blobUrl, ... }} />

// Option 2: Backend signed URL
const signedUrl = await fetch('/api/get-signed-url?file=...')
<DocumentViewer file={{ content: signedUrl, ... }} />

// Option 3: Base64 (small files only)
const base64 = `data:${mimeType};base64,${base64String}`
<DocumentViewer file={{ content: base64, ... }} />
```

## 📊 Excel Features

- **Multi-sheet support**: Tab navigation between sheets
- **Responsive tables**: Scrollable with fixed layout
- **Cell formatting**: Preserves basic formatting
- **Large file handling**: Client-side parsing with SheetJS

## 📄 PDF Features

- **Full UI controls**: Zoom, page navigation, search
- **Responsive**: Works on mobile and desktop
- **Plugin system**: Default layout plugin included
- **Custom toolbar**: Can be customized via plugins

## 🐛 Troubleshooting

### "Cannot find module" errors
Ensure all packages are installed:
```bash
npm install @react-pdf-viewer/core @react-pdf-viewer/default-layout mammoth xlsx papaparse
```

### SSR Errors
All preview components must use `"use client" directive and be imported with `dynamic(..., { ssr: false })`.

### Horizontal Scrollbar Issues
Ensure root layout has overflow constraints:
```tsx
<html className="overflow-hidden">
  <body className="overflow-hidden">
```

### PDF Worker Error
PDF.js requires a worker. Using CDN version:
```tsx
workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js"
```

## 🚀 Performance

- **Code splitting**: Dynamic imports reduce initial bundle
- **Lazy loading**: Preview components load only when needed
- **Memoization**: `useMemo` prevents unnecessary re-renders
- **Worker threads**: PDF rendering uses web workers

## 🔮 Future Enhancements

Potential additions:
- [ ] PowerPoint preview (pptx)
- [ ] Video/audio previews
- [ ] 3D model viewers (STL, OBJ)
- [ ] Code syntax highlighting (enhanced)
- [ ] Markdown rendering with TOC
- [ ] Text file search and highlighting
- [ ] Print functionality per format
- [ ] Annotation/commenting system

## 📝 License

Part of the Orbimesh project.

---

**Built with ❤️ using Next.js, React, and modern browser APIs**
