"""
Browser Agent - Python-based Screenshot Highlights

Draws bounding boxes and coordinate rulers directly on screenshot images
using PIL, replacing the previous JS DOM injection approach.

Advantages over JS injection:
- Zero DOM pollution (page is never touched)
- No render wait needed (draws on captured image buffer)
- Consistent rendering regardless of page CSS
- Smart element filtering in Python (easy to debug)
- Live browser stream stays clean
"""

import io
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ─── Font Setup ──────────────────────────────────────────────────────────────

_FONT_CACHE: Dict[int, Optional[ImageFont.FreeTypeFont]] = {}

_FONT_PATHS = [
    'C:\\Windows\\Fonts\\consola.ttf',     # Windows - Consolas (monospace)
    'C:\\Windows\\Fonts\\arial.ttf',        # Windows - Arial
    '/System/Library/Fonts/Arial.ttf',      # macOS
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
    'arial.ttf',
]

def _get_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Load a cross-platform font with caching."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            break
        except OSError:
            continue
    _FONT_CACHE[size] = font
    return font


# ─── Element Filtering ──────────────────────────────────────────────────────
# NOTE: filter_elements() has been REMOVED. 
# Filtering now happens upstream in dom.py's selector_map pipeline.
# This ensures text [N] indices match visual [N] boxes exactly.

# ─── HSL Color Generation ───────────────────────────────────────────────────

def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """Convert HSL (0-360, 0-1, 0-1) to RGB (0-255, 0-255, 0-255)."""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    
    if h < 60:    r1, g1, b1 = c, x, 0
    elif h < 120: r1, g1, b1 = x, c, 0
    elif h < 180: r1, g1, b1 = 0, c, x
    elif h < 240: r1, g1, b1 = 0, x, c
    elif h < 300: r1, g1, b1 = x, 0, c
    else:         r1, g1, b1 = c, 0, x
    
    return (int((r1 + m) * 255), int((g1 + m) * 255), int((b1 + m) * 255))


def _get_element_color(index: int) -> Tuple[int, int, int]:
    """Generate a unique color using Golden Angle HSL rotation."""
    hue = (index * 137.508) % 360
    return _hsl_to_rgb(hue, 1.0, 0.45)


# ─── Drawing Functions ───────────────────────────────────────────────────────

def _draw_dashed_rect(draw: ImageDraw.Draw, rect: Tuple[int, int, int, int], 
                       color: Tuple[int, int, int], width: int = 1, 
                       dash: int = 6, gap: int = 4):
    """Draw a dashed rectangle."""
    x1, y1, x2, y2 = rect
    
    def dashed_line(sx, sy, ex, ey):
        if sx == ex:  # vertical
            y = sy
            step = 1 if ey > sy else -1
            while (step > 0 and y < ey) or (step < 0 and y > ey):
                end = min(y + dash * step, ey) if step > 0 else max(y + dash * step, ey)
                draw.line([(sx, y), (sx, end)], fill=color, width=width)
                y = end + gap * step
        else:  # horizontal
            x = sx
            step = 1 if ex > sx else -1
            while (step > 0 and x < ex) or (step < 0 and x > ex):
                end = min(x + dash * step, ex) if step > 0 else max(x + dash * step, ex)
                draw.line([(x, sy), (end, sy)], fill=color, width=width)
                x = end + gap * step
    
    dashed_line(x1, y1, x2, y1)  # Top
    dashed_line(x2, y1, x2, y2)  # Right
    dashed_line(x2, y2, x1, y2)  # Bottom  
    dashed_line(x1, y2, x1, y1)  # Left


def _draw_label(draw: ImageDraw.Draw, text: str, x: int, y: int, 
                el_width: int, el_height: int,
                color: Tuple[int, int, int], font: Optional[ImageFont.FreeTypeFont],
                img_size: Tuple[int, int]):
    """Draw a compact label for an element."""
    # Measure text
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = len(text) * 6, 10
    
    padding = 2
    lw = tw + padding * 2
    lh = th + padding * 2
    
    # Position: inside top-left corner for normal elements,
    # above the box for small elements
    if el_width < 50 or el_height < 25:
        lx = x
        ly = max(0, y - lh - 1)
    else:
        lx = x + 2
        ly = y + 2
    
    # Clamp to image bounds
    lx = max(0, min(lx, img_size[0] - lw))
    ly = max(0, min(ly, img_size[1] - lh))
    
    # Draw background + text
    draw.rectangle([lx, ly, lx + lw, ly + lh], fill=color)
    text_x = lx + padding
    text_y = ly + padding - (bbox[1] if font else 0)
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)


def _draw_coordinate_grid(draw: ImageDraw.Draw, width: int, height: int,
                           font: Optional[ImageFont.FreeTypeFont]):
    """Draw orange coordinate rulers along edges."""
    step = 200
    tick_len = 12
    orange = (255, 165, 0, 200)
    
    # X-axis (top)
    for x in range(0, width, step):
        draw.line([(x, 0), (x, tick_len)], fill=orange, width=2)
        label = str(x)
        draw.text((x + 3, tick_len + 1), label, fill=orange, font=font)
    
    # Y-axis (left)
    for y in range(0, height, step):
        draw.line([(0, y), (tick_len, y)], fill=orange, width=2)
        label = str(y)
        draw.text((tick_len + 2, y + 1), label, fill=orange, font=font)


# ─── Main Entry Point ────────────────────────────────────────────────────────

def annotate_screenshot(
    screenshot_bytes: bytes,
    elements: List[Dict[str, Any]],
    viewport_width: int = 1280,
    viewport_height: int = 900,
    draw_grid: bool = True,
) -> bytes:
    """
    Take raw screenshot bytes, draw bounding boxes and rulers,
    return annotated screenshot as JPEG bytes.
    
    Args:
        screenshot_bytes: Raw JPEG/PNG screenshot from Playwright
        elements: Pre-filtered selector_map elements from dom.py
                  (interactive-only, already viewport/size filtered upstream)
        viewport_width: Browser viewport width for coordinate scaling
        viewport_height: Browser viewport height for coordinate scaling
        draw_grid: Whether to draw coordinate grid rulers
    
    Returns:
        Annotated screenshot as JPEG bytes
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        
        # Load fonts
        label_font = _get_font(11)
        grid_font = _get_font(10)
        
        # NO filtering here — elements are pre-filtered by dom.py's selector_map
        # This ensures text [N] indices match visual [N] boxes exactly
        
        # Calculate scale factor (screenshot pixels vs CSS viewport)
        scale_x = img_w / viewport_width if viewport_width > 0 else 1.0
        scale_y = img_h / viewport_height if viewport_height > 0 else 1.0
        
        # Draw bounding boxes and labels for ALL elements
        drawn_count = 0
        for i, el in enumerate(elements):
            # Skip elements without valid coordinates
            x_raw = el.get('top_left_x')
            y_raw = el.get('top_left_y')
            w_raw = el.get('width', 0)
            h_raw = el.get('height', 0)
            if x_raw is None or y_raw is None or w_raw < 2 or h_raw < 2:
                continue
            
            color = _get_element_color(i)
            
            # Scale coordinates from CSS to screenshot pixels
            x1 = int(x_raw * scale_x)
            y1 = int(y_raw * scale_y)
            w = int(w_raw * scale_x)
            h = int(h_raw * scale_y)
            x2 = x1 + w
            y2 = y1 + h
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))
            
            # Draw dashed bounding box
            _draw_dashed_rect(draw, (x1, y1, x2, y2), color, width=1)
            
            # Draw index label — 1-based, matching text list
            _draw_label(draw, str(i + 1), x1, y1, w, h, color, label_font, (img_w, img_h))
            drawn_count += 1
        
        logger.info(f"🎨 Drew {drawn_count} bounding boxes on screenshot")
        
        # Draw coordinate grid
        if draw_grid:
            _draw_coordinate_grid(draw, img_w, img_h, grid_font)
        
        # Encode back to JPEG
        out = io.BytesIO()
        img.save(out, format='JPEG', quality=80)
        return out.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to annotate screenshot: {e}")
        return screenshot_bytes  # Return original on failure
