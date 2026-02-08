---
id: document_agent
name: Document Agent
port: 8050
version: 2.0.0
---

# Document Agent

LLM-powered document processing for PDF, Word, and text files.

## Capabilities

- Read and extract text from PDF documents
- Process Word documents (.docx)
- Create new documents from scratch
- Edit existing documents based on instructions
- Summarize document contents
- Answer questions about document content
- Convert between document formats

## When to Use

Use this agent when the user:
- Uploads or mentions PDF files
- Works with Word documents (.docx)
- Wants to create or edit documents
- Asks to summarize a document
- Has questions about document content
- Needs document format conversion

## NOT For

- Spreadsheets (CSV, Excel) → use Spreadsheet Agent
- Emails → use Mail Agent
- Web pages → use Browser Agent
- Images without text → (future Vision Agent)

## Example Prompts

- "Summarize this PDF document"
- "Create a report about quarterly sales"
- "Extract all dates mentioned in this document"
- "Edit the introduction to be more formal"
- "What are the key points in this contract?"

## Notes

- Uses python-docx for Word document manipulation
- PDF text extraction via PyMuPDF
- Supports chunked processing for large documents
- Planning-based approach for complex edits
