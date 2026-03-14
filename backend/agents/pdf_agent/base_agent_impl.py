"""
PDF Agent v1.0 — Complete BaseAgent Implementation

Full PDF processing with:
- Text extraction (pypdf, pdfplumber, vision fallback)
- Table extraction
- Merge / Split / Rotate
- Create PDFs (reportlab)
- OCR for scanned documents
- Watermark, password protection, image extraction
- Metadata extraction
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentServices,
    capability,
    ExecutionContext,
    AgentRequest,
    CapabilityResult,
)
from .llm_helpers import PDFLLMHelpers
from . import utils

logger = logging.getLogger("agents.pdf_agent")

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DEFAULT_STORAGE_DIR = WORKSPACE_ROOT / "storage" / "pdf_agent"


class PDFAgentConfig:
    """Configuration for PDF Agent."""

    max_file_size_mb: int = 100
    enable_ocr: bool = True
    enable_vision_fallback: bool = True


class PDFAgent(BaseAgent, PDFLLMHelpers):
    """
    Comprehensive PDF processing agent.

    Features:
    - Text extraction with multiple backends
    - Table extraction to structured data
    - Merge / Split / Rotate operations
    - PDF creation from text / structured content
    - OCR for scanned documents
    - Watermark, password protection
    - Image extraction
    - Metadata extraction
    
    Inherits from PDFLLMHelpers for LLM-powered features:
    - analyze_pdf_structure()
    - summarize_pdf()
    - extract_key_information()
    - answer_questions_about_pdf()
    - suggest_pdf_improvements()
    - generate_pdf_metadata()
    """

    def __init__(
        self,
        agent_id: str = "pdf_agent",
        agent_name: str = "PDF Agent",
        services: Optional[AgentServices] = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services,
            config=config or AgentConfig(
                max_retries=2,
                request_timeout=240.0,
            ),
        )
        self.storage_dir = DEFAULT_STORAGE_DIR
        self.agent_config = PDFAgentConfig()

    async def _initialize_resources(self):
        """Initialize PDF agent resources."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("PDF Agent resources initialized")

    async def _cleanup_resources(self):
        """Cleanup resources."""
        pass

    def _get_custom_metrics(self) -> Dict[str, Any]:
        """Return PDF agent metrics."""
        return {
            "storage_dir": str(self.storage_dir),
        }

    async def _get_step_context(
        self,
        request: AgentRequest,
        context: ExecutionContext,
        previous_results: List[Any],
    ) -> str:
        """Provide context about files mentioned in the request."""
        ctx_parts = [
            f"Storage directory: {self.storage_dir}",
            f"Previous results: {len(previous_results)} steps completed",
        ]
        # Check if payload has file paths
        payload = request.payload or {}
        if "file_path" in payload:
            fp = payload["file_path"]
            if os.path.exists(fp):
                size_kb = os.path.getsize(fp) / 1024
                ctx_parts.append(f"Input file: {fp} ({size_kb:.1f} KB)")
            else:
                ctx_parts.append(f"Input file: {fp} (NOT FOUND)")

        return "\n".join(ctx_parts)

    async def _update_state_post_step(
        self, step: Any, result: Any, context: ExecutionContext
    ):
        """Post-step logging."""
        pass

    async def _llm_synthesize_response(
        self,
        results: List[Any],
        understanding: Dict[str, Any],
        request: AgentRequest,
    ) -> Any:  # Returns AgentResponse
        """Synthesize final response with CanvasService-powered display."""
        successful = [r for r in results if getattr(r, "success", False)]

        from backend.agents.base import AgentResponse

        if not successful:
            return AgentResponse.error(
                message="No PDF operations completed successfully.",
            )

        # Aggregate results
        all_data = {}
        canvas_display = None
        messages = []
        file_path = None
        text_content = None

        for r in results:
            data = getattr(r, "data", {}) or {}
            all_data.update(data)
            if "canvas_display" in data:
                canvas_display = data["canvas_display"]
            if "message" in data:
                messages.append(data["message"])
            # Track outputs for canvas generation
            if "file_path" in data:
                file_path = data["file_path"]
            if "text" in data:
                text_content = data["text"]

        # --- Dynamic Canvas Generation via CanvasService ---
        if not canvas_display:
            try:
                from services.canvas_service import CanvasService

                if file_path and str(file_path).lower().endswith(".pdf"):
                    # File output → use PDF viewer
                    canvas_obj = CanvasService.build_pdf_view(
                        file_path=file_path,
                        title=Path(file_path).name,
                    )
                    canvas_display = canvas_obj.model_dump()
                elif text_content:
                    # Text extraction → use pdf_summary template
                    preview = text_content[:3000]
                    if len(text_content) > 3000:
                        preview += f"\n\n... ({len(text_content) - 3000} more characters)"
                    canvas_obj = CanvasService.build_from_template(
                        "pdf_summary",
                        data={
                            "content": preview,
                            "title": "PDF Text Extraction",
                            "page_count": all_data.get("metadata", {}).get("page_count"),
                            "file_path": all_data.get("file_path"),
                        },
                        title="PDF Content",
                    )
                    if canvas_obj:
                        canvas_display = canvas_obj.model_dump()
                else:
                    # Fallback: LLM-based canvas decision
                    summary_text = " | ".join(messages) if messages else str(all_data)
                    canvas_obj = await CanvasService.decide_canvas_llm(
                        output=summary_text,
                        agent_name="pdf_agent",
                        primary_canvas_type="pdf",
                    )
                    if canvas_obj:
                        canvas_display = canvas_obj.model_dump()
            except Exception as e:
                logger.warning(f"Canvas generation failed (non-fatal): {e}")

        summary_msg = " | ".join(messages) if messages else "PDF process completed."
        
        return AgentResponse.success(
            result=summary_msg,
            summary=summary_msg,
            data=all_data,
            canvas_display=canvas_display
        )

    # =========================================================================
    # CAPABILITIES
    # =========================================================================

    @capability(
        name="extract_text",
        description=(
            "Extract text from a PDF file. "
            "Params: file_path (str), method ('pypdf'|'pdfplumber'|'auto', default 'auto')"
        ),
    )
    async def extract_text(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Extract text from PDF with multiple backends."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        self.emit_progress(f"📄 Extracting text from {Path(file_path).name}...")
        method = params.get("method", "auto")

        try:
            if method == "pdfplumber":
                text = utils.extract_text_pdfplumber(file_path)
            elif method == "pypdf":
                text = utils.extract_text_pypdf(file_path)
            else:
                # Auto: try pypdf first, fall back to pdfplumber, then vision
                text = utils.extract_text_pypdf(file_path)
                if not text.strip():
                    text = utils.extract_text_pdfplumber(file_path)
                if not text.strip() and self.agent_config.enable_vision_fallback:
                    self.emit_progress("🔍 No text found — trying vision extraction...")
                    text = utils.extract_text_with_vision(file_path)

            return CapabilityResult(
                success=True,
                data={
                    "text": text,
                    "char_count": len(text),
                    "message": f"Extracted {len(text)} characters from {Path(file_path).name}",
                },
            )
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="extract_tables",
        description=(
            "Extract tables from a PDF file. "
            "Params: file_path (str)"
        ),
    )
    async def extract_tables(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Extract tables from PDF."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        self.emit_progress(f"📊 Extracting tables from {Path(file_path).name}...")

        try:
            tables = utils.extract_tables(file_path)
            return CapabilityResult(
                success=True,
                data={
                    "tables": tables,
                    "table_count": len(tables),
                    "message": f"Extracted {len(tables)} table(s) from {Path(file_path).name}",
                },
            )
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="merge_pdfs",
        description=(
            "Merge multiple PDF files into one. "
            "Params: file_paths (list[str]), output_path (str, optional)"
        ),
    )
    async def merge_pdfs(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Merge multiple PDFs."""
        file_paths = params.get("file_paths", [])
        if not file_paths or len(file_paths) < 2:
            return CapabilityResult(
                success=False,
                data={"error": "Need at least 2 PDF files to merge."},
            )

        output_path = params.get(
            "output_path",
            str(self.storage_dir / f"merged_{int(time.time())}.pdf"),
        )

        self.emit_progress(f"📎 Merging {len(file_paths)} PDFs...")

        try:
            result_path = utils.merge_pdfs(file_paths, output_path)
            canvas = utils.create_pdf_canvas_display(result_path, "Merged PDF")

            return CapabilityResult(
                success=True,
                data={
                    "file_path": result_path,
                    "canvas_display": canvas,
                    "message": f"Merged {len(file_paths)} PDFs → {Path(result_path).name}",
                },
            )
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="split_pdf",
        description=(
            "Split a PDF into separate files. "
            "Params: file_path (str), page_ranges (list of [start, end] pairs, optional), "
            "output_dir (str, optional)"
        ),
    )
    async def split_pdf(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Split PDF by page ranges or into individual pages."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        output_dir = params.get(
            "output_dir",
            str(self.storage_dir / f"split_{int(time.time())}"),
        )
        page_ranges = params.get("page_ranges")
        if page_ranges:
            page_ranges = [tuple(r) for r in page_ranges]

        self.emit_progress(f"✂️ Splitting {Path(file_path).name}...")

        try:
            output_files = utils.split_pdf(file_path, output_dir, page_ranges)
            return CapabilityResult(
                success=True,
                data={
                    "output_files": output_files,
                    "file_count": len(output_files),
                    "message": f"Split into {len(output_files)} file(s)",
                },
            )
        except Exception as e:
            logger.error(f"Split failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="create_pdf",
        description=(
            "Create a new PDF from text or structured content. "
            "Params: content (str) OR sections (list of {heading, body, level}), "
            "output_path (str, optional), file_name (str, optional)"
        ),
    )
    async def create_pdf(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Create a new PDF."""
        content = params.get("content")
        sections = params.get("sections")
        file_name = params.get("file_name", f"document_{int(time.time())}.pdf")
        output_path = params.get(
            "output_path",
            str(self.storage_dir / file_name),
        )

        self.emit_progress(f"📝 Creating PDF: {file_name}...")

        try:
            if sections:
                result_path = utils.create_pdf_with_structure(sections, output_path)
            elif content:
                result_path = utils.create_pdf_from_text(content, output_path)
            else:
                return CapabilityResult(
                    success=False,
                    data={"error": "Provide 'content' (text) or 'sections' (structured)."},
                )

            canvas = utils.create_pdf_canvas_display(result_path, file_name)
            return CapabilityResult(
                success=True,
                data={
                    "file_path": result_path,
                    "canvas_display": canvas,
                    "message": f"Created PDF: {Path(result_path).name}",
                },
            )
        except Exception as e:
            logger.error(f"Create PDF failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="ocr_scan",
        description=(
            "OCR a scanned/image-only PDF to extract text. "
            "Params: file_path (str)"
        ),
    )
    async def ocr_scan(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """OCR a scanned PDF."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        self.emit_progress(f"🔍 Running OCR on {Path(file_path).name}...")

        try:
            text = utils.ocr_pdf(file_path)
            return CapabilityResult(
                success=True,
                data={
                    "text": text,
                    "char_count": len(text),
                    "message": f"OCR extracted {len(text)} characters from {Path(file_path).name}",
                },
            )
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="add_watermark",
        description=(
            "Add a watermark to a PDF. "
            "Params: file_path (str), watermark_path (str), output_path (str, optional)"
        ),
    )
    async def add_watermark(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Add watermark overlay to PDF."""
        file_path = params.get("file_path")
        watermark_path = params.get("watermark_path")

        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"Source file not found: {file_path}"},
            )
        if not watermark_path or not os.path.exists(watermark_path):
            return CapabilityResult(
                success=False,
                data={"error": f"Watermark file not found: {watermark_path}"},
            )

        output_path = params.get(
            "output_path",
            str(self.storage_dir / f"watermarked_{int(time.time())}.pdf"),
        )

        self.emit_progress("💧 Adding watermark...")

        try:
            result_path = utils.add_watermark(file_path, watermark_path, output_path)
            canvas = utils.create_pdf_canvas_display(result_path, "Watermarked PDF")
            return CapabilityResult(
                success=True,
                data={
                    "file_path": result_path,
                    "canvas_display": canvas,
                    "message": f"Watermarked PDF → {Path(result_path).name}",
                },
            )
        except Exception as e:
            logger.error(f"Watermark failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="rotate_pages",
        description=(
            "Rotate pages in a PDF. "
            "Params: file_path (str), angle (int, default 90), "
            "pages (list[int], optional — 1-indexed, rotates all if omitted), "
            "output_path (str, optional)"
        ),
    )
    async def rotate_pages(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Rotate PDF pages."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        angle = params.get("angle", 90)
        pages = params.get("pages")
        output_path = params.get(
            "output_path",
            str(self.storage_dir / f"rotated_{int(time.time())}.pdf"),
        )

        self.emit_progress(f"🔄 Rotating pages by {angle}°...")

        try:
            result_path = utils.rotate_pages(file_path, output_path, pages, angle)
            canvas = utils.create_pdf_canvas_display(result_path, "Rotated PDF")
            return CapabilityResult(
                success=True,
                data={
                    "file_path": result_path,
                    "canvas_display": canvas,
                    "message": f"Rotated PDF → {Path(result_path).name}",
                },
            )
        except Exception as e:
            logger.error(f"Rotate failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="extract_metadata",
        description=(
            "Extract metadata from a PDF file. "
            "Params: file_path (str)"
        ),
    )
    async def extract_metadata(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Extract PDF metadata."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        self.emit_progress(f"ℹ️ Extracting metadata from {Path(file_path).name}...")

        try:
            metadata = utils.extract_metadata(file_path)
            return CapabilityResult(
                success=True,
                data={
                    "metadata": metadata,
                    "message": f"Metadata extracted: {metadata.get('page_count', '?')} pages",
                },
            )
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="password_protect",
        description=(
            "Encrypt a PDF with a password. "
            "Params: file_path (str), user_password (str), "
            "owner_password (str, optional), output_path (str, optional)"
        ),
    )
    async def password_protect(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Password-protect a PDF."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        user_password = params.get("user_password")
        if not user_password:
            return CapabilityResult(
                success=False,
                data={"error": "user_password is required."},
            )

        owner_password = params.get("owner_password")
        output_path = params.get(
            "output_path",
            str(self.storage_dir / f"encrypted_{int(time.time())}.pdf"),
        )

        self.emit_progress("🔒 Encrypting PDF...")

        try:
            result_path = utils.password_protect(
                file_path, output_path, user_password, owner_password
            )
            return CapabilityResult(
                success=True,
                data={
                    "file_path": result_path,
                    "message": f"Encrypted PDF → {Path(result_path).name}",
                },
            )
        except Exception as e:
            logger.error(f"Password protection failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="extract_images",
        description=(
            "Extract embedded images from a PDF. "
            "Params: file_path (str), output_dir (str, optional)"
        ),
    )
    async def extract_images(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Extract images from PDF."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        output_dir = params.get(
            "output_dir",
            str(self.storage_dir / f"images_{int(time.time())}"),
        )

        self.emit_progress(f"🖼️ Extracting images from {Path(file_path).name}...")

        try:
            image_paths = utils.extract_images_from_pdf(file_path, output_dir)
            return CapabilityResult(
                success=True,
                data={
                    "image_paths": image_paths,
                    "image_count": len(image_paths),
                    "message": f"Extracted {len(image_paths)} image(s)",
                },
            )
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )
