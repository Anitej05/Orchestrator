"""
PPT Agent v1.0 — Complete BaseAgent Implementation

Presentation processing with:
- Read and analyze PPTX structure
- Create presentations with design-aware layouts
- Edit existing presentations
- Extract text from all slides
- Convert slides to images
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
from .llm_helpers import PPTLLMHelpers
from . import utils

logger = logging.getLogger("agents.ppt_agent")

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DEFAULT_STORAGE_DIR = WORKSPACE_ROOT / "storage" / "ppt_agent"


class PPTAgentConfig:
    """Configuration for PPT Agent."""

    max_file_size_mb: int = 100
    default_palette: str = "midnight_executive"
    default_font_index: int = 0


class PPTAgent(BaseAgent, PPTLLMHelpers):
    """
    Presentation processing agent.

    Features:
    - Read and analyze PPTX structure and content
    - Create new presentations with design-aware layouts, color palettes, typography
    - Edit existing presentations (add/remove/reorder slides, change content)
    - Extract text from all slides
    - Convert slides to images for preview/inspection
    
    Inherits from PPTLLMHelpers for LLM-powered features:
    - plan_presentation_structure()
    - suggest_slide_layout()
    - generate_slide_content()
    - enhance_text_for_presentation()
    - suggest_color_palette()
    - suggest_visual_elements()
    - check_presentation_consistency()
    - generate_speaker_notes()
    """

    def __init__(
        self,
        agent_id: str = "ppt_agent",
        agent_name: str = "PPT Agent",
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
        self.agent_config = PPTAgentConfig()

    async def _initialize_resources(self):
        """Initialize PPT agent resources."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("PPT Agent resources initialized")

    async def _cleanup_resources(self):
        """Cleanup resources."""
        pass

    def _get_custom_metrics(self) -> Dict[str, Any]:
        """Return PPT agent metrics."""
        return {
            "storage_dir": str(self.storage_dir),
        }

    async def _get_step_context(
        self,
        request: AgentRequest,
        context: ExecutionContext,
        previous_results: List[Any],
    ) -> str:
        """Provide context about current state."""
        ctx_parts = [
            f"Storage directory: {self.storage_dir}",
            f"Previous results: {len(previous_results)} steps completed",
            f"Available color palettes: {', '.join(utils.COLOR_PALETTES.keys())}",
        ]
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
                message="No presentation operations completed successfully.",
            )

        all_data = {}
        canvas_display = None
        messages = []
        file_path = None
        text_content = None
        slide_count = None

        for r in results:
            data = getattr(r, "data", {}) or {}
            all_data.update(data)
            if "canvas_display" in data:
                canvas_display = data["canvas_display"]
            if "message" in data:
                messages.append(data["message"])
            if "file_path" in data:
                file_path = data["file_path"]
            if "text" in data:
                text_content = data["text"]
            if "slide_count" in data:
                slide_count = data["slide_count"]

        # --- Dynamic Canvas Generation via CanvasService ---
        if not canvas_display:
            try:
                from services.canvas_service import CanvasService

                if file_path and str(file_path).lower().endswith(".pptx"):
                    # File output → use PPTX viewer
                    canvas_obj = CanvasService.build_pptx_view(
                        file_path=file_path,
                        title=Path(file_path).name,
                        slide_count=slide_count,
                    )
                    canvas_display = canvas_obj.model_dump()
                elif text_content:
                    # Text extraction → use pptx_summary template
                    preview = text_content[:3000]
                    if len(text_content) > 3000:
                        preview += f"\n\n... ({len(text_content) - 3000} more characters)"
                    canvas_obj = CanvasService.build_from_template(
                        "pptx_summary",
                        data={
                            "content": preview,
                            "title": "Presentation Summary",
                            "slide_count": slide_count,
                            "file_path": file_path,
                        },
                        title="Presentation Content",
                    )
                    if canvas_obj:
                        canvas_display = canvas_obj.model_dump()
                else:
                    # Fallback: LLM-based canvas decision
                    summary_text = " | ".join(messages) if messages else str(all_data)
                    canvas_obj = await CanvasService.decide_canvas_llm(
                        output=summary_text,
                        agent_name="ppt_agent",
                        primary_canvas_type="pptx",
                    )
                    if canvas_obj:
                        canvas_display = canvas_obj.model_dump()
            except Exception as e:
                logger.warning(f"Canvas generation failed (non-fatal): {e}")

        summary_msg = " | ".join(messages) if messages else "Presentation operation completed."
        
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
        name="read_presentation",
        description=(
            "Read and analyze a PPTX presentation structure, slides, layouts, and content. "
            "Params: file_path (str)"
        ),
    )
    async def read_presentation(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Analyze presentation structure and content."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        self.emit_progress(f"📊 Analyzing presentation {Path(file_path).name}...")

        try:
            structure = utils.analyze_presentation_structure(file_path)
            text = utils.extract_text_from_presentation(file_path)

            return CapabilityResult(
                success=True,
                data={
                    "structure": structure,
                    "text": text,
                    "slide_count": structure["slide_count"],
                    "message": (
                        f"Analyzed {Path(file_path).name}: "
                        f"{structure['slide_count']} slides"
                    ),
                },
            )
        except Exception as e:
            logger.error(f"Read presentation failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="create_presentation",
        description=(
            "Create a new PPTX presentation with design-aware layouts. "
            "Params: slides (list of {title, content, layout, notes, subtitle}), "
            "file_name (str, optional), palette (str, optional — see available palettes), "
            "font_index (int, optional — 0-5)"
        ),
    )
    async def create_presentation(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Create a new presentation."""
        slides_data = params.get("slides", [])
        if not slides_data:
            return CapabilityResult(
                success=False,
                data={"error": "Provide 'slides' list with at least one slide."},
            )

        file_name = params.get(
            "file_name", f"presentation_{int(time.time())}.pptx"
        )
        output_path = params.get(
            "output_path",
            str(self.storage_dir / file_name),
        )
        palette = params.get("palette", self.agent_config.default_palette)
        font_index = params.get("font_index", self.agent_config.default_font_index)

        self.emit_progress(
            f"🎨 Creating presentation: {file_name} "
            f"({len(slides_data)} slides, palette: {palette})..."
        )

        try:
            result_path = utils.create_presentation(
                slides_data, output_path, palette, font_index
            )
            canvas = utils.create_pptx_canvas_display(result_path, file_name)

            return CapabilityResult(
                success=True,
                data={
                    "file_path": result_path,
                    "canvas_display": canvas,
                    "slide_count": len(slides_data),
                    "palette_used": palette,
                    "message": (
                        f"Created {Path(result_path).name} "
                        f"with {len(slides_data)} slides"
                    ),
                },
            )
        except Exception as e:
            logger.error(f"Create presentation failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="edit_presentation",
        description=(
            "Edit an existing PPTX presentation. "
            "Params: file_path (str), action ('edit_slide'|'add_slide'|'remove_slide'), "
            "slide_index (int, 0-based), updates (dict, for edit_slide), "
            "slide_data (dict, for add_slide), position (int, optional, for add_slide), "
            "output_path (str, optional)"
        ),
    )
    async def edit_presentation(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Edit an existing presentation."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        action = params.get("action", "edit_slide")
        output_path = params.get("output_path")

        self.emit_progress(f"✏️ Editing presentation: {action}...")

        try:
            if action == "edit_slide":
                slide_index = params.get("slide_index", 0)
                updates = params.get("updates", {})
                result_path = utils.edit_slide_content(
                    file_path, slide_index, updates, output_path
                )
                msg = f"Updated slide {slide_index + 1}"

            elif action == "add_slide":
                slide_data = params.get("slide_data", {})
                position = params.get("position")
                result_path = utils.add_slide(
                    file_path, slide_data, position, output_path
                )
                msg = "Added new slide"

            elif action == "remove_slide":
                slide_index = params.get("slide_index", 0)
                result_path = utils.remove_slide(
                    file_path, slide_index, output_path
                )
                msg = f"Removed slide {slide_index + 1}"

            else:
                return CapabilityResult(
                    success=False,
                    data={"error": f"Unknown action: {action}"},
                )

            canvas = utils.create_pptx_canvas_display(
                result_path, Path(result_path).name
            )

            return CapabilityResult(
                success=True,
                data={
                    "file_path": result_path,
                    "canvas_display": canvas,
                    "message": msg,
                },
            )
        except Exception as e:
            logger.error(f"Edit presentation failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )

    @capability(
        name="extract_text",
        description=(
            "Extract all text content from a PPTX presentation. "
            "Params: file_path (str)"
        ),
    )
    async def extract_text(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Extract text from all slides."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        self.emit_progress(f"📄 Extracting text from {Path(file_path).name}...")

        try:
            text = utils.extract_text_from_presentation(file_path)
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
        name="convert_to_images",
        description=(
            "Convert PPTX slides to image files. "
            "Params: file_path (str), output_dir (str, optional), dpi (int, optional, default 150)"
        ),
    )
    async def convert_to_images(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Convert slides to images."""
        file_path = params.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return CapabilityResult(
                success=False,
                data={"error": f"File not found: {file_path}"},
            )

        output_dir = params.get(
            "output_dir",
            str(self.storage_dir / f"slides_{int(time.time())}"),
        )
        dpi = params.get("dpi", 150)

        self.emit_progress(f"🖼️ Converting slides to images...")

        try:
            image_files = utils.convert_slides_to_images(file_path, output_dir, dpi)
            return CapabilityResult(
                success=True,
                data={
                    "image_files": image_files,
                    "image_count": len(image_files),
                    "message": f"Converted {len(image_files)} slides to images",
                },
            )
        except Exception as e:
            logger.error(f"Convert to images failed: {e}")
            return CapabilityResult(
                success=False,
                data={"error": str(e)},
            )
