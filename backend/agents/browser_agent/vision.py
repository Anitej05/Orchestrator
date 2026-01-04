"""
Browser Agent - Vision Module (SOTA with Set-of-Mark)

Vision model integration with Set-of-Mark (SoM) overlays for precise element selection.
"""

import os
import re
import json
import base64
import asyncio
import logging
import io
import httpx  # For direct Ollama API calls
from typing import Dict, Any, Optional, List, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
from .schemas import ActionPlan, AtomicAction

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


class VisionClient:
    """Vision model client with Set-of-Mark overlays for precise action planning - uses true async calls"""
    
    def __init__(self):
        # Primary: Ollama with Qwen VL (using direct httpx for native API format)
        self.ollama_api_key = OLLAMA_API_KEY
        self.ollama_base_url = "https://ollama.com/v1/chat/completions"
        
        # Fallback: NVIDIA with Mistral Large (for vision)
        self.nvidia = AsyncOpenAI(
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1"
        ) if NVIDIA_API_KEY else None
        
        self.model = "qwen3-vl:235b-cloud"
        self.model_nvidia = "mistralai/mistral-large-3-675b-instruct-2512"  # NVIDIA vision fallback
        self.max_tokens = None  # No limit - let model decide
        self.valid_actions = ["click", "type", "scroll", "search", "navigate", "done", "wait", "fail", "hover", "press", "go_back", "save_info", "skip_subtask", "run_js"]
        self.mark_elements: Dict[int, Dict] = {}  # Store mark→element mapping
    
    @property
    def available(self) -> bool:
        """Check if vision is available (either Ollama or NVIDIA fallback)"""
        return self.ollama_api_key is not None or self.nvidia is not None
    
    def _add_som_overlay(self, screenshot_b64: str, elements: List[Dict]) -> Tuple[str, Dict[int, Dict]]:
        """Add Set-of-Mark overlays with element boundaries and numbered labels
        
        Enhanced visualization:
        - Draws bounding boxes around elements
        - Semi-transparent highlighting for visibility
        - Numbered labels with element type indicators
        
        Returns:
            - Modified screenshot as base64
            - Mapping of mark number → element data
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available for SoM overlay")
            return screenshot_b64, {}
        
        try:
            # Decode image
            img_bytes = base64.b64decode(screenshot_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            
            # Create overlay layer for semi-transparent elements
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Main draw context
            draw = ImageDraw.Draw(img)
            
            # Try to get a good font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
                small_font = ImageFont.truetype("arial.ttf", 10)
            except:
                font = ImageFont.load_default()
                small_font = font
            
            mark_mapping = {}
            
            # Color palette with distinct colors for different element types
            type_colors = {
                'button': (0, 150, 0, 180),      # Green for buttons
                'link': (0, 100, 255, 180),      # Blue for links
                'input': (255, 165, 0, 180),     # Orange for inputs
                'textbox': (255, 165, 0, 180),   # Orange for textboxes
                'checkbox': (128, 0, 128, 180),  # Purple for checkboxes
                'default': (255, 0, 0, 180)      # Red for others
            }
            
            # Mark ALL viewport-visible elements (increased limit for better coverage)
            for idx, el in enumerate(elements[:300]):  # increased from 150 to 300
                mark_num = idx + 1
                x = el.get('x', 0)
                y = el.get('y', 0)
                w = el.get('width', 80)  # Default width if not provided
                h = el.get('height', 30)  # Default height if not provided
                role = el.get('role', 'element').lower()
                name = el.get('name', '')
                xpath = el.get('xpath', '')
                
                # Get color based on element type
                color = type_colors.get(role, type_colors['default'])
                border_color = (color[0], color[1], color[2], 255)  # Solid border
                
                # Ensure valid coordinates
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img.width - 1, int(x + w))
                y2 = min(img.height - 1, int(y + h))
                
                # Draw semi-transparent bounding box on overlay
                # Validate coordinates: x2 must be > x1 and y2 must be > y1
                if w > 10 and h > 10 and x2 > x1 and y2 > y1:
                    overlay_draw.rectangle(
                        [x1, y1, x2, y2],
                        fill=(color[0], color[1], color[2], 40),  # Very light fill
                        outline=border_color,
                        width=2
                    )
                
                # Draw numbered label with type indicator
                type_icon = '🔘' if 'button' in role else '🔗' if 'link' in role else '📝' if 'input' in role or 'text' in role else '◆'
                label = f"{mark_num}"
                bbox = draw.textbbox((0, 0), label, font=font)
                label_w = bbox[2] - bbox[0] + 8
                label_h = bbox[3] - bbox[1] + 6
                
                # Position label at top-left of element
                label_x = max(0, min(x1, img.width - label_w))
                label_y = max(0, y1 - label_h - 2)
                if label_y < 5:  # If too close to top, put inside element
                    label_y = y1 + 2
                
                # Draw label background (solid color for visibility)
                draw.rectangle(
                    [label_x, label_y, label_x + label_w, label_y + label_h],
                    fill=(color[0], color[1], color[2]),
                    outline=(255, 255, 255)
                )
                
                # Draw label text
                draw.text((label_x + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
                
                # Store mapping with section context
                mark_mapping[mark_num] = {
                    'role': role,
                    'name': name[:100],  # Increased from 50
                    'xpath': xpath,
                    'section': el.get('section', ''),
                    'x': x, 'y': y, 'width': w, 'height': h
                }
            
            # Composite overlay onto image
            img = Image.alpha_composite(img, overlay)
            
            # Convert back to RGB for encoding
            img = img.convert('RGB')
            
            # COMPRESS for vision API: resize and use JPEG
            # Max width 1024px to reduce size while keeping enough detail
            max_width = 1024
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                logger.info(f"📐 Resized image from {img.width}x{img.height} to {new_size[0]}x{new_size[1]}")
            
            # Encode as JPEG for smaller size (85% quality per user request)
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            modified_b64 = base64.b64encode(output.getvalue()).decode()
            
            logger.info(f"🎨 SoM overlay: marked {len(mark_mapping)} elements, image size: {len(modified_b64)} chars")
            return modified_b64, mark_mapping
            
        except Exception as e:
            logger.error(f"Failed to add SoM overlay: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return screenshot_b64, {}
    
    async def plan_action_with_vision(
        self,
        task: str,
        screenshot_base64: str,
        page_content: Dict[str, Any],
        history: list,
        step: int
    ) -> Optional[ActionPlan]:
        """Plan action based on screenshot analysis with Set-of-Mark overlays"""
        
        if not self.available:
            logger.warning("Vision not available (no OLLAMA_API_KEY or NVIDIA_API_KEY)")
            return None
        
        try:
            # Add Set-of-Mark overlays
            elements = page_content.get('elements', [])
            marked_screenshot, self.mark_elements = self._add_som_overlay(screenshot_base64, elements)
            
            # Build prompt with mark legend
            prompt = self._build_vision_prompt(task, page_content, history, step, self.mark_elements)
            
            logger.info(f"📸 Calling vision model: {self.model}")
            logger.debug(f"Vision prompt length: {len(prompt)} chars, image size: {len(marked_screenshot)} chars")
            
            # Log exact image size for debugging empty response issues
            print(f"DEBUG: marked_screenshot length: {len(marked_screenshot)}", flush=True)
            
            # STRICT SYSTEM MESSAGE - enforces JSON output with ENHANCED VISUAL UNDERSTANDING
            system_msg = """You are an EXPERT browser automation agent with ADVANCED VISUAL UNDERSTANDING capabilities.

## YOUR VISUAL ANALYSIS STRENGTHS
You can see and understand:
- Complex UI layouts (Apple, Samsung, premium e-commerce sites)
- 3D animations, hero sections, and visual effects
- Overlays, modals, sticky headers, and floating elements
- Visual hierarchy and design patterns
- Elements that may be hidden from text extraction

## CRITICAL ANALYSIS GUIDELINES

### 1. VISUAL HIERARCHY FIRST
Before acting, analyze:
- What is the MAIN content area vs navigation/sidebars?
- Are there any OVERLAYS or MODALS blocking interaction?
- What elements are VISUALLY PROMINENT (large, centered, high contrast)?
- Is there a STICKY HEADER or FLOATING NAVIGATION?

### 2. ELEMENT IDENTIFICATION
The image has colored bounding boxes and numbered labels:
- GREEN boxes = Buttons (click actions)
- BLUE boxes = Links (navigation)
- ORANGE boxes = Input fields (typing)
- RED boxes = Other interactive elements
Use the MARK NUMBER to reference elements: {"mark": N}

### 3. COMPLEX UI PATTERNS
Watch for:
- MEGA MENUS: Hover-triggered dropdowns that may not be captured
- CAROUSELS: Multiple items that need arrow clicks to see more
- LAZY LOADING: Content that appears on scroll
- VISUAL CTA BUTTONS: Large "Buy Now", "Shop", "Learn More" buttons
- COOKIE BANNERS: Usually at top/bottom, click to dismiss

### 4. WHEN TEXT FAILS, VISION SUCCEEDS
Use vision when:
- Premium sites with mostly visual content (Apple, Tesla)
- Product images are more informative than text
- Navigation is icon-based without text labels
- Modal/popup elements overlay the main content

### 5. USE HIERARCHY FOR CONTEXT
You also receive a "PAGE HIERARCHY" tree. Use it to:
- Understand grouping (what belongs to what)
- Identify headings and sections not obvious visually
- Confirm text that might be hard to read in the image

## OUTPUT RULES
1. Your response MUST be valid JSON starting with { and ending with }
2. NO text before or after JSON
3. NO markdown code blocks
4. Switch to "next_mode": "text" after you've understood the visual layout

REQUIRED JSON FORMAT:
{
  "reasoning": "Visual analysis: [what you see] → Decision: [what to do]",
  "actions": [{"name": "action_name", "params": {...}}],
  "confidence": 0.9,
  "next_mode": "text"
}

Valid actions: click, type, scroll, navigate, done, wait, hover, press, go_back, save_info, skip_subtask, run_js
For clicks: use {"mark": N} where N is the bounding box number"""

            # MERGE SYSTEM PROMPT INTO USER MESSAGE for better Qwen compatibility
            full_prompt_text = system_msg + "\n\n" + prompt

            # Ollama Cloud uses NATIVE format: images as array of raw base64 strings
            # NOT OpenAI format with image_url data URIs
            ollama_messages = [
                {
                    "role": "user",
                    "content": full_prompt_text,
                    "images": [marked_screenshot]  # Raw base64, no data URI prefix!
                }
            ]
            
            # NVIDIA uses OpenAI format with image_url (kept as fallback)
            nvidia_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{marked_screenshot}"
                            }
                        }
                    ]
                }
            ]
            
            # Define tool for structured output - UPDATED FOR MULTI-ACTION
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "browser_action",
                        "description": "Output a SEQUENCE of browser actions based on visual analysis.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reasoning": {
                                    "type": "string",
                                    "description": "Detailed reasoning for the action sequence, analyzing the UI elements and task state."
                                },
                                "actions": {
                                    "type": "array",
                                    "description": "List of actions to execute in order.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "enum": ["click", "type", "scroll", "search", "navigate", "done", "wait", "fail", "hover", "press", "go_back", "save_info", "skip_subtask", "run_js"],
                                                "description": "The name of the action."
                                            },
                                            "params": {
                                                "type": "object",
                                                "description": "Parameters for the action (e.g., {'mark': 1})."
                                            }
                                        },
                                        "required": ["name", "params"]
                                    }
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence score between 0.0 and 1.0."
                                },
                                "next_mode": {
                                    "type": "string",
                                    "enum": ["vision", "text"],
                                    "description": "The next mode to switch to."
                                }
                            },
                            "required": ["reasoning", "actions", "confidence", "next_mode"]
                        }
                    }
                }
            ]

            response = None
            nvidia_content = None  # Store content for NVIDIA (primary)
            ollama_content = None  # Store content for Ollama (fallback)
            
            # NVIDIA is primary (works reliably), Ollama is fallback (often times out)
            
            # Try NVIDIA first (primary vision model) with 30s timeout
            if self.nvidia:
                try:
                    logger.info(f"🎨 Trying NVIDIA vision: {self.model_nvidia} (30s timeout)")
                    # NVIDIA with OpenAI format (with image support)
                    response = await asyncio.wait_for(
                        self.nvidia.chat.completions.create(
                            model=self.model_nvidia,
                            messages=nvidia_messages,  # OpenAI format with image_url
                        ),
                        timeout=90.0
                    )
                    if response and response.choices:
                        nvidia_content = response.choices[0].message.content
                        if nvidia_content:
                            logger.info(f"🔮 NVIDIA Vision Response: {nvidia_content[:200]}...")
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ NVIDIA vision timed out after 30s - trying Ollama fallback")
                    nvidia_content = None
                except Exception as nvidia_err:
                    logger.warning(f"⚠️ NVIDIA vision failed: {nvidia_err}")
                    nvidia_content = None
            
            # Fallback to Ollama if NVIDIA failed (30s timeout)
            # Using direct httpx for native Ollama API format
            if not nvidia_content and self.ollama_api_key:
                try:
                    logger.info(f"🎨 Falling back to Ollama vision: {self.model} (30s timeout)")
                    
                    # Direct httpx call with native Ollama format
                    async with httpx.AsyncClient(timeout=90.0) as client:
                        ollama_response = await client.post(
                            self.ollama_base_url,
                            headers={
                                "Authorization": f"Bearer {self.ollama_api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": self.model,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": full_prompt_text,
                                        "images": [marked_screenshot]  # Raw base64, native format
                                    }
                                ],
                                "temperature": 0.1
                            }
                        )
                        
                        if ollama_response.status_code == 200:
                            data = ollama_response.json()
                            # Extract content from response
                            choices = data.get("choices", [])
                            if choices and len(choices) > 0:
                                ollama_content = choices[0].get("message", {}).get("content", "")
                                if ollama_content:
                                    logger.info(f"🔮 Ollama Vision Response: {ollama_content[:200]}...")
                                else:
                                    logger.warning("🔮 Ollama Vision: Empty content in response")
                            else:
                                logger.warning("🔮 Ollama Vision: No choices in response")
                        else:
                            logger.warning(f"⚠️ Ollama vision failed with status {ollama_response.status_code}: {ollama_response.text[:200]}")

                except httpx.TimeoutException:
                    logger.warning(f"⏱️ Ollama vision timed out after 30s - trying NVIDIA fallback")
                    ollama_content = None

                except Exception as ollama_err:
                    print(f"\n\n❌ OLLAMA VISION ERROR: {ollama_err}\n\n", flush=True)
                    logger.warning(f"⚠️ Ollama vision failed: {ollama_err}")
                    ollama_content = None
            
            # Process response - prioritize NVIDIA content (primary), then Ollama (fallback)
            content_to_parse = None
            
            if nvidia_content:
                content_to_parse = nvidia_content
            elif ollama_content:
                content_to_parse = ollama_content
            elif response and response.choices:
                msg = response.choices[0].message
                # Check for tool calls first
                if msg.tool_calls:
                    logger.info(f"🎨 Vision returned Tool Call: {msg.tool_calls[0].function.name}")
                    try:
                        args = json.loads(msg.tool_calls[0].function.arguments)
                        actions_list = []
                        if "actions" in args:
                            for act in args["actions"]:
                                actions_list.append(AtomicAction(
                                    name=act.get("name"),
                                    params=act.get("params", {})
                                ))
                        
                        if actions_list:
                            return ActionPlan(
                                reasoning=args.get("reasoning", ""),
                                actions=actions_list,
                                confidence=args.get("confidence", 1.0),
                                next_mode=args.get("next_mode", "text"),
                                completed_subtasks=args.get("completed_subtasks", [])
                            )
                    except Exception as e:
                        logger.error(f"Error processing tool call: {e}")
                
                # Fall back to content parsing
                content_to_parse = msg.content or getattr(msg, 'reasoning', None)
            
            if not content_to_parse:
                logger.warning("Vision returned no parseable content")
                return None
            
            # Parse the content into ActionPlan
            logger.debug(f"Parsing vision response content (len={len(content_to_parse)})...")
            return self._parse_action(content_to_parse, self.mark_elements)
            
        except Exception as e:
            logger.error(f"Vision failed: {e}", exc_info=True)
            return None
    
    async def analyze_image(
        self,
        screenshot_base64: str,
        task: str,
        page_url: str
    ) -> Optional[str]:
        """Analyze/describe an image on the page (Compressed)"""
        
        if not self.available:
            return None
            
        # COMPRESS raw screenshot if needed
        # analyze_image often receives raw PNGs which are huge (2-4MB)
        try:
            if len(screenshot_base64) > 500000: # If > ~375KB
                img_bytes = base64.b64decode(screenshot_base64)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                if img.width > 1024:
                    ratio = 1024 / img.width
                    img = img.resize((1024, int(img.height * ratio)), Image.LANCZOS)
                
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=85, optimize=True)
                screenshot_base64 = base64.b64encode(output.getvalue()).decode()
                logger.info(f"📉 Compressed analyze_image input to {len(screenshot_base64)} chars")
        except Exception as e:
            logger.warning(f"Failed to compress analyze_image input: {e}")

        try:
            prompt = f"""Analyze this screenshot and describe what you see.

TASK: {task}
URL: {page_url}

Describe any images, logos, doodles, or visual content you see on the page.
Focus on visual elements that are relevant to the task.

Provide a detailed description of what you observe."""

            response = None
            
            # Try Ollama first if available
            if self.ollama:
                try:
                    response = await self.ollama.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_base64}"}}
                                ]
                            }
                        ],
                        max_tokens=1000,
                        temperature=0.3
                    )
                except Exception as ollama_err:
                    logger.warning(f"Ollama image analysis failed: {ollama_err}")
                    response = None
            
            # Fallback to NVIDIA if Ollama failed or unavailable
            if (not response or not response.choices) and self.nvidia:
                try:
                    logger.info("Falling back to NVIDIA for image analysis")
                    # NVIDIA doesn't support images directly, use text-only prompt
                    response = await self.nvidia.chat.completions.create(
                        model=self.model_nvidia,
                        messages=[{"role": "user", "content": prompt + "\n\n(Note: Analyzing based on context only, no image available)"}],
                        max_tokens=1000,
                        temperature=0.3
                    )
                except Exception as nvidia_err:
                    logger.warning(f"NVIDIA image analysis fallback failed: {nvidia_err}")
                    response = None
            
            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    # Strip any thinking tags
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    return content.strip()
            return None
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return None
    
    def _build_vision_prompt(
        self,
        task: str,
        page_content: Dict[str, Any],
        history: list,
        step: int,
        mark_elements: Dict[int, Dict] = None
    ) -> str:
        """Build prompt for vision-based action planning with Set-of-Mark legend"""
        
        # Recent actions with success/failure status
        history_str = ""
        if history:
            recent = history[-20:]
            lines = []
            for h in recent:
                try:
                    step_num = h.get('step', '?')
                    action_plan = h.get('action', {})
                    reasoning = action_plan.get('reasoning', 'No reasoning')[:100]
                    actions = action_plan.get('actions', [])
                    action_names = [a['name'] for a in actions] if isinstance(actions, list) else ['?']
                    result = h.get('result', {})
                    success = "✅" if result.get('success', False) else "🛑 FAILED"
                    msg = result.get('message', '')[:80]
                    line = f"  {success} Step {step_num}: {action_names}\n    Reasoning: {reasoning}\n    Result: {msg}"
                    lines.append(line)
                except:
                    pass
            history_str = "\n".join(lines)
        
        # Build Set-of-Mark legend with section context
        mark_legend = ""
        if mark_elements:
            legend_lines = []
            for mark_num, el in mark_elements.items():
                section = el.get('section', '')
                line = f'  [{mark_num}] {el.get("role", "element")}: "{el.get("name", "")}"'
                if section:
                    line += f'  [under: {section}]'
                legend_lines.append(line)
            mark_legend = "\n".join(legend_lines)  # Full legend with section context
        
        return f"""You are a browser automation agent with VISION capabilities. Analyze this screenshot and decide the next action.

═══════════════════════════════════════════════════════════════════════════════
HOW TO USE THIS SCREENSHOT
═══════════════════════════════════════════════════════════════════════════════

The screenshot has NUMBERED LABELS [1], [2], [3]... overlaid on interactive elements.
- To click an element, use: {{"name": "click", "params": {{"mark": 3}}}}
- The mark number corresponds to the element labeled [3] in the screenshot
- I will automatically resolve the mark to coordinates/xpath for you

═══════════════════════════════════════════════════════════════════════════════
CURRENT TASK
═══════════════════════════════════════════════════════════════════════════════
{task}

CURRENT STATE:
• URL: {page_content.get('url', 'about:blank')}
• Title: {page_content.get('title', '')}
• Step: {step}
• Scroll: {page_content.get('scroll_position', 0)}px / {page_content.get('max_scroll', 0)}px ({page_content.get('scroll_percent', 100)}% down)

PAGE HIERARCHY (Semantic Structure):
{page_content.get('unified_page_tree', '(hierarchy unavailable)')}

SET-OF-MARK LEGEND (elements visible in screenshot):
{mark_legend if mark_legend else "(no marks available)"}

═══════════════════════════════════════════════════════════════════════════════
PREVIOUS ACTIONS (CRITICAL: Study these to avoid repeating failed attempts!)
═══════════════════════════════════════════════════════════════════════════════
{history_str if history_str else "(none yet)"}

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE ACTIONS
═══════════════════════════════════════════════════════════════════════════════
• navigate    → {{"url": "https://example.com"}}
• click       → {{"mark": 3}} (PREFERRED - use numbered labels from screenshot)
• click       → {{"index": 52}} (use element # from accessibility tree for multiple same-name elements)
• click       → {{"x": 640, "y": 400}} (fallback if mark not visible)
• type        → {{"text": "search query", "submit": true}}
• scroll      → {{"direction": "down", "amount": 500}}
• wait        → {{"seconds": 2}}
• hover       → {{"mark": 5}}
• press       → {{"key": "Escape"}}
• go_back     → {{}}
• save_info   → {{"key": "price", "value": "₹1,07,970"}}
• skip_subtask→ {{"reason": "Site is down"}}
• done        → {{}}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════════════════

1. USE MARKS: Click by mark number [{{"mark": 3}}] is the MOST RELIABLE method!
2. AVOID LOOPS: If history shows 2+ failed attempts on same action, try DIFFERENT approach
3. CHECK URL: If URL already shows task is done (e.g., "price-asc-rank" means sorted), MOVE ON to next task!
4. BE SPECIFIC: When using save_info, extract ACTUAL values from the page (prices, names)

═══════════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT (STRICT JSON - MUST FOLLOW EXACTLY!)
═══════════════════════════════════════════════════════════════════════════════

⚠️ CRITICAL: next_mode MUST be either "text" or "vision" - NO OTHER VALUES ALLOWED!
- "text" = Use text-based DOM analysis for next step (faster, use for most actions)
- "vision" = Use screenshot analysis for next step (use when visual verification needed)

VALID RESPONSE EXAMPLE:
{{
  "reasoning": "I see [3] is the search box at the top, clicking it to enter search query",
  "actions": [{{"name": "click", "params": {{"mark": 3}}}}],
  "confidence": 0.95,
  "next_mode": "text",
  "completed_subtasks": []
}}

ANOTHER VALID EXAMPLE (saving extracted data):
{{
  "reasoning": "I found the product price ₹1,07,970 shown on the page. Saving it now.",
  "actions": [{{"name": "save_info", "params": {{"key": "price", "value": "₹1,07,970"}}}}],
  "confidence": 0.9,
  "next_mode": "text",
  "completed_subtasks": []
}}

COMPLETING A SUBTASK (when visual goal is achieved - DESCRIBE something):
{{
  "reasoning": "I can see the Picture of the Day shows a snow-covered mountain (Nuptse) in the Himalayas with dramatic lighting and clouds.",
  "actions": [
    {{"name": "save_info", "params": {{"key": "picture_description", "value": "Snow-covered mountain Nuptse in the Himalayas with dramatic lighting and clouds"}}}},
    {{"name": "done", "params": {{}}}}
  ],
  "confidence": 0.95,
  "next_mode": "text",
  "completed_subtasks": [1]
}}

⚠️ CRITICAL WORKFLOW for "describe" tasks:
1. FIRST use "save_info" to store what you see/describe
2. THEN use "done" to mark the subtask complete
3. Set "completed_subtasks" to include the current subtask ID
This ensures the description is saved before moving to the next task.

NOW RESPOND WITH YOUR ACTION (JSON ONLY - remember next_mode must be "text" or "vision"):"""
    
    def _parse_action(self, response: str, mark_elements: Dict[int, Dict] = None) -> Optional[ActionPlan]:
        """Parse vision response into ActionPlan, resolving mark numbers to coordinates/xpath/role+name"""
        try:
            # Strip thinking tags
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
            
            # Extract JSON from code blocks if present
            code_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response, re.IGNORECASE)
            if code_match:
                json_str = code_match.group(1)
            else:
                # Find JSON object
                json_match = re.search(r'\{[\s\S]*\}', response)
                json_str = json_match.group() if json_match else None
            
            if json_str:
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                data = json.loads(json_str)
                
                # Handle actions list or single action fallback
                actions = []
                if 'actions' in data:
                    for act in data['actions']:
                        params = act.get('params', {})
                        
                        # Resolve mark number to coordinates/xpath/role+name!
                        if 'mark' in params and mark_elements:
                            mark_num = params['mark']
                            if mark_num in mark_elements:
                                el = mark_elements[mark_num]
                                x_coord = el.get('x', 0)
                                y_coord = el.get('y', 0)
                                
                                # Always add coordinates if valid
                                if x_coord > 0 and y_coord > 0:
                                    params['x'] = x_coord
                                    params['y'] = y_coord
                                    logger.info(f"🎯 Resolved mark [{mark_num}] to coords ({x_coord}, {y_coord})")
                                
                                # Also include xpath as fallback if available
                                xpath = el.get('xpath', '')
                                if xpath:
                                    params['xpath'] = xpath
                                    logger.info(f"🎯 Mark [{mark_num}] also has xpath: {xpath[:40]}")
                                
                                # ALWAYS include role+name as fallback (most reliable)
                                role = el.get('role', '')
                                name = el.get('name', '')
                                if role and name:
                                    params['role'] = role
                                    params['name'] = name
                                    logger.info(f"🎯 Mark [{mark_num}] also has role+name: [{role}] {name[:30]}")
                            else:
                                logger.warning(f"⚠️ Mark [{mark_num}] not found in mapping (available: {list(mark_elements.keys())[:10]})")
                        
                        actions.append(AtomicAction(name=act['name'], params=params))
                else:
                    actions.append(AtomicAction(
                        name=data.get('action', 'wait'),  # Use wait instead of done
                        params=data.get('params', {'seconds': 1})
                    ))

                # CRITICAL FIX: Normalize next_mode to valid values
                # Vision model sometimes returns 'done', 'click', etc. which are invalid
                raw_next_mode = data.get('next_mode', 'text')
                if raw_next_mode not in ['text', 'vision']:
                    logger.warning(f"⚠️ Vision returned invalid next_mode '{raw_next_mode}', normalizing to 'text'")
                    raw_next_mode = 'text'

                return ActionPlan(
                    reasoning=data.get('reasoning', ''),
                    actions=actions,
                    confidence=data.get('confidence', 0.8),
                    next_mode=raw_next_mode,
                    completed_subtasks=data.get('completed_subtasks', []),
                    updated_plan=data.get('updated_plan', None)
                )
        except Exception as e:
            logger.error(f"Failed to parse vision response: {e}")
            logger.error(f"Response was: {response[:300]}...")
        
        return None

