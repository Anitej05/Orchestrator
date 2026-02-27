"""
Browser Agent - Vision Utilities

Image processing utilities for the browser agent:
- Set-of-Mark (SoM) overlays for element annotation
- Screenshot compression and resizing
- Standalone image analysis via InferenceService

NOTE: Action planning is now handled entirely by the unified multimodal
LLMClient in llm.py. This module is only for image processing utilities.
"""

import base64
import logging
import io
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv

# Use centralized inference service
from backend.services.inference_service import inference_service, InferencePriority
from langchain_core.messages import HumanMessage

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)


class VisionUtils:
    """Vision utility class for image processing and analysis.
    
    Provides:
    - SoM overlay annotation for screenshots  
    - Screenshot compression/resizing for token-efficient multimodal calls
    - Standalone image analysis (not action planning)
    """
    
    def __init__(self):
        self.mark_elements: Dict[int, Dict] = {}
    
    @property
    def available(self) -> bool:
        """Check if vision utilities are available (PIL required)."""
        return PIL_AVAILABLE
    
    @staticmethod
    def compress_screenshot(screenshot_b64: str, max_width: int = 800, quality: int = 50) -> str:
        """Compress and resize a screenshot for token-efficient multimodal calls.
        
        Reduces a full-res screenshot to ~10-20KB (~250-500 image tokens).
        """
        if not PIL_AVAILABLE:
            return screenshot_b64
        
        try:
            img_bytes = base64.b64decode(screenshot_b64)
            img = Image.open(io.BytesIO(img_bytes))
            
            # Resize if wider than max_width
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.LANCZOS)
            
            # Convert to RGB (drop alpha) and compress as JPEG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            compressed_b64 = base64.b64encode(output.getvalue()).decode()
            
            original_size = len(screenshot_b64)
            compressed_size = len(compressed_b64)
            ratio_pct = (compressed_size / original_size * 100) if original_size > 0 else 0
            logger.debug(f"📸 Screenshot compressed: {original_size//1024}KB → {compressed_size//1024}KB ({ratio_pct:.0f}%)")
            
            return compressed_b64
        except Exception as e:
            logger.warning(f"Screenshot compression failed: {e}")
            return screenshot_b64
    
    def add_som_overlay(self, screenshot_b64: str, elements: List[Dict]) -> Tuple[str, Dict[int, Dict]]:
        """Add Set-of-Mark overlays with element boundaries and numbered labels.
        
        Returns:
            Tuple of (annotated_screenshot_b64, mark_mapping)
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available for SoM overlay")
            return screenshot_b64, {}
        
        try:
            img_bytes = base64.b64decode(screenshot_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            
            # Create overlay layer for semi-transparent elements
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except Exception:
                font = ImageFont.load_default()
            
            mark_mapping = {}
            
            type_colors = {
                'button': (0, 150, 0, 180),      # Green
                'link': (0, 100, 255, 180),      # Blue
                'input': (255, 165, 0, 180),     # Orange
                'textbox': (255, 165, 0, 180),   # Orange
                'checkbox': (128, 0, 128, 180),  # Purple
                'default': (255, 0, 0, 180)      # Red
            }
            
            for idx, el in enumerate(elements[:300]):
                mark_num = idx + 1
                x, y = el.get('x', 0), el.get('y', 0)
                w, h = el.get('width', 80), el.get('height', 30)
                role = el.get('role', 'element').lower()
                
                color = type_colors.get(role, type_colors['default'])
                border_color = (color[0], color[1], color[2], 255)
                
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(img.width - 1, int(x + w)), min(img.height - 1, int(y + h))
                
                if w > 10 and h > 10 and x2 > x1 and y2 > y1:
                    overlay_draw.rectangle([x1, y1, x2, y2], fill=(color[0], color[1], color[2], 40), outline=border_color, width=2)
                
                # Label
                label = f"{mark_num}"
                bbox = draw.textbbox((0, 0), label, font=font)
                label_w = bbox[2] - bbox[0] + 8
                label_h = bbox[3] - bbox[1] + 6

                label_x = max(0, min(x1, img.width - label_w))
                label_y = max(0, y1 - label_h - 2)
                if label_y < 5: label_y = y1 + 2
                
                draw.rectangle([label_x, label_y, label_x + label_w, label_y + label_h], fill=(color[0], color[1], color[2]), outline=(255, 255, 255))
                draw.text((label_x + 4, label_y + 1), label, fill=(255, 255, 255), font=font)
                
                mark_mapping[mark_num] = {
                    'role': role,
                    'name': el.get('name', '')[:100],
                    'xpath': el.get('xpath', ''),
                    'section': el.get('section', ''),
                    'x': x, 'y': y, 'width': w, 'height': h
                }
            
            img = Image.alpha_composite(img, overlay).convert('RGB')
            
            # Compress for the multimodal model
            max_width = 800
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=50, optimize=True)
            modified_b64 = base64.b64encode(output.getvalue()).decode()
            
            logger.info(f"🎨 SoM overlay: marked {len(mark_mapping)} elements")
            return modified_b64, mark_mapping
            
        except Exception as e:
            logger.error(f"Failed to add SoM overlay: {e}")
            return screenshot_b64, {}
    
    async def analyze_image(
        self,
        screenshot_base64: str,
        task: str,
        page_url: str
    ) -> Optional[str]:
        """Analyze image using InferenceService (standalone, not for action planning)."""
        try:
            prompt = f"""Analyze this screenshot. 
            TASK: {task}
            URL: {page_url}
            
            Describe visual elements relevant to the task (buttons, banners, modals)."""
            
            return await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                images=[screenshot_base64],
                priority=InferencePriority.QUALITY
            )
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return None


# Backward compatibility alias
VisionClient = VisionUtils
