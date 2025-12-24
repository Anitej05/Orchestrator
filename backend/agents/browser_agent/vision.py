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
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
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


class VisionClient:
    """Vision model client with Set-of-Mark overlays for precise action planning"""
    
    def __init__(self):
        self.ollama = OpenAI(
            api_key=OLLAMA_API_KEY,
            base_url="https://ollama.com/v1"
        ) if OLLAMA_API_KEY else None
        
        self.model = "qwen3-vl:235b-cloud"
        self.max_tokens = 1200
        self.valid_actions = ["click", "type", "scroll", "search", "navigate", "done", "wait", "fail", "hover", "press", "go_back", "save_info", "skip_subtask"]
        self.mark_elements: Dict[int, Dict] = {}  # Store mark→element mapping
    
    @property
    def available(self) -> bool:
        """Check if vision is available"""
        return self.ollama is not None
    
    def _add_som_overlay(self, screenshot_b64: str, elements: List[Dict]) -> Tuple[str, Dict[int, Dict]]:
        """Add Set-of-Mark numbered labels to screenshot
        
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
            img = Image.open(io.BytesIO(img_bytes))
            draw = ImageDraw.Draw(img)
            
            # Try to get a good font
            try:
                font = ImageFont.truetype("arial.ttf", 14)
                small_font = ImageFont.truetype("arial.ttf", 10)
            except:
                font = ImageFont.load_default()
                small_font = font
            
            mark_mapping = {}
            
            # Color palette for marks
            colors = [
                (255, 0, 0), (0, 150, 0), (0, 0, 255), (255, 165, 0),
                (128, 0, 128), (0, 128, 128), (255, 0, 255), (128, 128, 0)
            ]
            
            # Draw marks on top 30 elements (limit to avoid clutter)
            for idx, el in enumerate(elements[:50]):  # More elements for better coverage
                mark_num = idx + 1
                x = el.get('x', 0)
                y = el.get('y', 0)
                role = el.get('role', 'element')
                name = el.get('name', '')
                xpath = el.get('xpath', '')
                
                color = colors[idx % len(colors)]
                
                # Draw label box
                label = f"[{mark_num}]"
                bbox = draw.textbbox((0, 0), label, font=font)
                label_w = bbox[2] - bbox[0] + 4
                label_h = bbox[3] - bbox[1] + 4
                
                # Position label above element center
                label_x = max(0, min(x - label_w // 2, img.width - label_w))
                label_y = max(0, y - 20)
                
                # Draw background rectangle
                draw.rectangle(
                    [label_x, label_y, label_x + label_w, label_y + label_h],
                    fill=color,
                    outline=(255, 255, 255)
                )
                
                # Draw label text
                draw.text((label_x + 2, label_y + 2), label, fill=(255, 255, 255), font=font)
                
                # Store mapping
                mark_mapping[mark_num] = {
                    'role': role,
                    'name': name,
                    'xpath': xpath,
                    'x': x,
                    'y': y
                }
            
            # Encode modified image
            output = io.BytesIO()
            img.save(output, format='PNG')
            modified_b64 = base64.b64encode(output.getvalue()).decode()
            
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
        """Plan action based on screenshot analysis with Set-of-Mark overlays"""
        
        if not self.available:
            logger.warning("Vision not available (no OLLAMA_API_KEY)")
            return None
        
        try:
            # Add Set-of-Mark overlays
            elements = page_content.get('elements', [])
            marked_screenshot, self.mark_elements = self._add_som_overlay(screenshot_base64, elements)
            
            # Build prompt with mark legend
            prompt = self._build_vision_prompt(task, page_content, history, step, self.mark_elements)
            
            logger.info(f"📸 Calling vision model: {self.model}")
            logger.debug(f"Vision prompt length: {len(prompt)} chars, image size: {len(marked_screenshot)} chars")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{marked_screenshot}"
                            }
                        }
                    ]
                }
            ]
            
            # Define tool for structured output
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "browser_action",
                        "description": "Output the next browser action based on visual analysis.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reasoning": {
                                    "type": "string",
                                    "description": "Detailed reasoning for the action, analyzing the UI elements and task state."
                                },
                                "action_name": {
                                    "type": "string",
                                    "enum": ["click", "type", "scroll", "search", "navigate", "done", "wait", "fail", "hover", "press", "go_back", "save_info", "skip_subtask"],
                                    "description": "The name of the action to execute."
                                },
                                "params": {
                                    "type": "object",
                                    "description": "Parameters for the action (e.g., {'mark': 1} or {'text': 'shoes'})."
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
                            "required": ["reasoning", "action_name", "confidence", "next_mode"]
                        }
                    }
                }
            ]

            response = self.ollama.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.1,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "browser_action"}} 
            )
            
            # Check for valid response structure
            if not response or not response.choices:
                logger.warning("Vision returned no choices in response")
                return None
            
            # 1. Try to get Tool Call arguments (Preferred)
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                logger.info(f"🎨 Vision returned Tool Call: {tool_calls[0].function.name}")
                # DEBUG VERIFICATION LOG
                with open("vision_tool_success.log", "a") as f:
                     f.write(f"Tool Call: {tool_calls[0].function.name} Args: {tool_calls[0].function.arguments}\n")
                
                try:
                    args = json.loads(tool_calls[0].function.arguments)
                    # Map to Action object format expected by agent
                    # Note: Action Name in tool is 'action_name', agent expects 'name' inside valid_actions checking, 
                    # but _parse_action logic usually handles conversion. 
                    # Let's construct the dict to match what _parse_action produces.
                    
                    action_data = {
                        "reasoning": args.get("reasoning", ""),
                        "actions": [{
                            "name": args.get("action_name"),
                            "params": args.get("params", {})
                        }],
                        "confidence": args.get("confidence", 1.0),
                        "next_mode": args.get("next_mode", "text")
                    }
                    
                    # Validate action
                    if action_data["actions"][0]["name"] not in self.valid_actions:
                        logger.warning(f"Vision tool returned invalid action: {action_data['actions'][0]['name']}")
                        # Fallback to text parsing if invalid? Or just fail.
                    else:
                        # Return properly formatted Action object (via _parse_action helper usually, but manually here)
                        return ActionPlan(
                            reasoning=action_data["reasoning"],
                            actions=[AtomicAction(name=action_data["actions"][0]["name"], params=action_data["actions"][0]["params"])],
                            confidence=action_data["confidence"],
                            next_mode=action_data.get("next_mode")
                        )

                except json.JSONDecodeError:
                    logger.error("Failed to decode tool arguments json")
                except Exception as e:
                    logger.error(f"Error processing tool call: {e}")

            # 2. Fallback: Check reasoning field (legacy/Qwen-specific fallback)
            # If tool call failed or wasn't present, but reasoning has content, maybe we can scrape it?
            # Actually, if we use tools, the model shouldn't be outputting free text actions unless it refuses the tool.
            
            content = response.choices[0].message.content
            if not content and hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                 content = response.choices[0].message.reasoning
                 
            if content and content.strip():
                 logger.info("Falling back to text parsing from content/reasoning...")
                 return self._parse_action(content, self.mark_elements)
            
            logger.warning("Vision failed to return tool call or parseable content.")
            return None
            
        except Exception as e:
            logger.error(f"Vision failed: {e}", exc_info=True)
            return None
    
    async def analyze_image(
        self,
        screenshot_base64: str,
        task: str,
        page_url: str
    ) -> Optional[str]:
        """Analyze/describe an image on the page"""
        
        if not self.available:
            return None
        
        try:
            prompt = f"""Analyze this screenshot and describe what you see.

TASK: {task}
URL: {page_url}

Describe any images, logos, doodles, or visual content you see on the page.
Focus on visual elements that are relevant to the task.

Provide a detailed description of what you observe."""

            response = self.ollama.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
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
        
        # Build Set-of-Mark legend
        mark_legend = ""
        if mark_elements:
            legend_lines = []
            for mark_num, el in mark_elements.items():
                legend_lines.append(f"  [{mark_num}] {el.get('role', 'element')}: \"{el.get('name', '')}\"")
            mark_legend = "\n".join(legend_lines)  # Full legend
        
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

