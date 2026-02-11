"""
Browser Agent - Vision Module (SOTA with Set-of-Mark)

Vision model integration with Set-of-Mark (SoM) overlays for precise element selection.
"""

import os
import re
import json
import base64
import logging
import io
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

# Use centralized inference service
from backend.services.inference_service import inference_service, ProviderType, InferencePriority
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
from .agent_schemas import ActionPlan, AtomicAction

load_dotenv()
logger = logging.getLogger(__name__)

class VisionClient:
    """Vision model client with Set-of-Mark overlays for precise action planning - uses Unified InferenceService"""
    
    def __init__(self):
        # Models configuration
        self.model_ollama = "qwen3-vl:235b-cloud" # Legacy/Specific model name for Ollama fallback
        self.mark_elements: Dict[int, Dict] = {} 
    
    @property
    def available(self) -> bool:
        """Check if vision is available - assume yes via InferenceService"""
        return True
    
    def _add_som_overlay(self, screenshot_b64: str, elements: List[Dict]) -> Tuple[str, Dict[int, Dict]]:
        """Add Set-of-Mark overlays with element boundaries and numbered labels"""
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
            draw = ImageDraw.Draw(img)
            
            # Try to get a good font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
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
            
            # Mark ALL viewport-visible elements (limit to 300)
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
            
            # COMPRESS for vision API
            max_width = 1024
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            modified_b64 = base64.b64encode(output.getvalue()).decode()
            
            logger.info(f"🎨 SoM overlay: marked {len(mark_mapping)} elements")
            return modified_b64, mark_mapping
            
        except Exception as e:
            logger.error(f"Failed to add SoM overlay: {e}")
            return screenshot_b64, {}
    
    async def plan_action_with_vision(
        self,
        task: str,
        screenshot_base64: str,
        page_content: Dict[str, Any],
        history: list,
        step: int
    ) -> Optional[ActionPlan]:
        """Plan action based on screenshot analysis using InferenceService"""
        
        try:
            # Add SoM Overlays
            marked_screenshot, self.mark_elements = self._add_som_overlay(screenshot_base64, page_content.get('elements', []))
            
            # Build Prompt
            user_prompt = self._build_vision_prompt(task, page_content, history, step, self.mark_elements)
            system_msg = """You are an EXPERT browser automation agent with ADVANCED VISUAL UNDERSTANDING.
            
            Look at the screenshot with NUMBERED LABELS [1], [2]...
            Your job is to identify the best action to take.
            
            RESPONSE FORMAT (JSON ONLY):
            {
              "reasoning": "Visual analysis... decision...",
              "actions": [{"name": "click", "params": {"mark": 3}}],
              "confidence": 0.9,
              "next_mode": "text"
            }
            
            Valid actions: click, type, scroll, navigate, done, wait, hover, press, go_back, save_info, skip_subtask, run_js
            For 'click', always use {"mark": N} if possible.
            """
            
            logger.info(f"📸 Calling Vision Model (Unified InferenceService)")
            
            content = None
            
            # 1. Try NVIDIA (Primary - Llama 3.2 90b Vision or similar)
            try:
                content = await inference_service.generate(
                    messages=[
                        SystemMessage(content=system_msg),
                        HumanMessage(content=user_prompt)
                    ],
                    images=[marked_screenshot],
                    provider=ProviderType.NVIDIA,
                    model_name="meta/llama-3.2-90b-vision-instruct", # Specific Vision Model
                    priority=InferencePriority.QUALITY,
                    json_mode=True,
                    fallback_enabled=False # Handle fallback manually here
                )
            except Exception as e:
                logger.warning(f"⚠️ NVIDIA Vision failed: {e}. Trying fallback...")
            
            # 2. Key Fallback: OLLAMA (if NVIDIA failed/timed out)
            if not content:
                try:
                    content = await inference_service.generate(
                        messages=[
                            SystemMessage(content=system_msg),
                            HumanMessage(content=user_prompt)
                        ],
                        images=[marked_screenshot],
                        provider=ProviderType.OPENAI, # Maps to OLLAMA via InferenceService logic
                        model_name=self.model_ollama,
                        priority=InferencePriority.SPEED,
                        json_mode=True
                    )
                except Exception as e:
                     logger.error(f"❌ Ollama Vision also failed: {e}")

            if not content:
                logger.warning("Vision returned empty content from all providers")
                return None
                
            return self._parse_action(content, self.mark_elements)
            
        except Exception as e:
            logger.error(f"plan_action_with_vision generic error: {e}", exc_info=True)
            return None
    
    async def analyze_image(
        self,
        screenshot_base64: str,
        task: str,
        page_url: str
    ) -> Optional[str]:
        """Analyze image using InferenceService"""
        try:
            # Compress if needed (simple heuristic)
            if len(screenshot_base64) > 500000 and PIL_AVAILABLE:
                 # Re-use SoM logic to just resize without drawing? 
                 # Or just pass raw if InferenceService handles it.
                 # InferenceService assumes valid base64.
                 pass

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

    def _build_vision_prompt(self, task: str, page_content: Dict, history: list, step: int, mark_elements: Dict = None) -> str:
        # Simplified prompt builder
        legend = ""
        if mark_elements:
             # Limit legend size
             items = list(mark_elements.items())[:50] 
             legend = "\n".join([f"[{k}] {v.get('role')} '{v.get('name')}'" for k, v in items])
             if len(mark_elements) > 50: legend += "\n...(more entries truncated)..."

        # Build history string
        history_str = ""
        if history:
            for h in history[-5:]:
                res = h.get('result', {})
                status = "✅" if res.get('success') else "❌"
                history_str += f"- Step {h.get('step')}: {h.get('action', {}).get('actions')} ({status})\n"

        return f"""TASK: {task}
URL: {page_content.get('url')}
Step: {step}

LEGEND (SoM - [mark] Role 'Name'):
{legend}

HISTORY:
{history_str}

Analyze the UI and provide the JSON action plan."""

    def _parse_action(self, response: str, mark_elements: Dict[int, Dict] = None) -> Optional[ActionPlan]:
        """Parse JSON response and resolve marks"""
        try:
             json_str = response.strip()
             # Basic cleanup
             if "```" in json_str:
                 match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
                 if match: json_str = match.group(1)
             
             data = json.loads(json_str)
             
             actions = []
             raw_actions = data.get('actions', [])
             if isinstance(raw_actions, dict): raw_actions = [raw_actions] # Handle single object
             
             for act in raw_actions:
                 params = act.get('params', {})
                 # Resolve Mark
                 if 'mark' in params and mark_elements:
                     m = int(params['mark']) # Ensure int
                     if m in mark_elements:
                         el = mark_elements[m]
                         # Inject precise coordinates/xpath
                         params.update({
                             'x': el['x'], 
                             'y': el['y'], 
                             'xpath': el['xpath'], 
                             'role': el['role'], 
                             'name': el['name']
                         })
                 actions.append(AtomicAction(name=act['name'], params=params))
             
             return ActionPlan(
                 reasoning=data.get('reasoning', ''),
                 actions=actions,
                 confidence=data.get('confidence', 1.0),
                 next_mode=data.get('next_mode', 'text'),
                 completed_subtasks=data.get('completed_subtasks', [])
             )
        except Exception as e:
            logger.error(f"Parse error: {e}")
            logger.debug(f"Failed JSON: {response}")
            return None
