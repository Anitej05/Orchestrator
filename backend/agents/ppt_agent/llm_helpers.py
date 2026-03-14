# agents/ppt_agent/llm_helpers.py
"""
PPT Agent - LLM Helper Functions

Domain-specific LLM methods for PowerPoint operations.
All methods use inference_service directly.

Preserves ALL original functionality and adds comprehensive LLM-powered features.
"""
import json
import logging
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage
from backend.services.inference_service import inference_service, InferencePriority

logger = logging.getLogger("ppt_agent.llm")


class PPTLLMHelpers:
    """
    PPT-specific LLM helpers.
    
    Mix this into PPTAgent to get presentation-specific LLM methods.
    All methods use inference_service directly.
    """
    
    # ========================================================================
    # PRESENTATION PLANNING & STRUCTURE
    # ========================================================================
    
    async def plan_presentation_structure(
        self,
        topic: str,
        audience: str = "general",
        slide_count: int = 5,
        style: str = "professional"
    ) -> Dict[str, Any]:
        """
        Plan the structure of a presentation.
        
        Args:
            topic: Presentation topic
            audience: Target audience (executive, technical, general, etc.)
            slide_count: Desired number of slides
            style: Presentation style (professional, creative, minimal, etc.)
            
        Returns:
            Dict with slide-by-slide outline including:
            - title, subtitle, content_type, key_points, notes
        """
        prompt = f"""You are a professional presentation designer. Create a structured outline for a presentation.

TOPIC: {topic}
AUDIENCE: {audience}
SLIDE COUNT: {slide_count} slides
STYLE: {style}

Create a slide-by-slide outline. For each slide include:
1. slide_number: Integer (1, 2, 3, ...)
2. slide_type: One of [title, agenda, content, data, comparison, quote, conclusion]
3. title: Clear, concise slide title
4. subtitle: Optional subtitle (or null)
5. key_points: 3-5 bullet points of content
6. speaker_notes: Brief notes for presenter
7. visual_suggestion: Suggested visual (chart, image, icon, etc.)

Return JSON with "slides" array and "presentation_title" and "presentation_subtitle".

Example structure:
{{
  "presentation_title": "...",
  "presentation_subtitle": "...",
  "slides": [
    {{
      "slide_number": 1,
      "slide_type": "title",
      "title": "...",
      "subtitle": "...",
      "key_points": [],
      "speaker_notes": "...",
      "visual_suggestion": "..."
    }}
  ]
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                temperature=0.3,
                json_mode=True,
                max_tokens=3000,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Presentation planning failed: {e}")
            # Return minimal fallback structure
            return {
                "presentation_title": topic,
                "slides": [
                    {
                        "slide_number": 1,
                        "slide_type": "title",
                        "title": topic,
                        "key_points": [],
                        "speaker_notes": "",
                        "visual_suggestion": "Professional title slide"
                    }
                ]
            }
    
    async def suggest_slide_layout(
        self,
        content_type: str,
        content_points: List[str]
    ) -> Dict[str, Any]:
        """
        Suggest optimal slide layout for given content.
        
        Args:
            content_type: Type of content (text, data, comparison, etc.)
            content_points: List of content points to include
            
        Returns:
            Layout suggestion with layout_name, placeholders, design_tips
        """
        content_str = "\n".join(content_points[:5])  # Limit content
        
        prompt = f"""You are a slide design expert. Suggest the best layout for this slide content.

CONTENT TYPE: {content_type}
CONTENT POINTS:
{content_str}

Suggest a layout from these options:
- title_only: Just a title
- title_and_content: Title with bullet points
- two_column: Two columns of content
- comparison: Side-by-side comparison
- image_with_caption: Large image with caption
- data_chart: Chart or graph
- quote: Large quote display
- timeline: Timeline or process flow

Return JSON with:
{{
  "layout_name": "...",
  "reasoning": "Why this layout works",
  "placeholders": ["title", "content", "image", etc.],
  "design_tips": ["tip 1", "tip 2"],
  "color_suggestion": "Suggested color from palette"
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                temperature=0.2,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Layout suggestion failed: {e}")
            return {
                "layout_name": "title_and_content",
                "reasoning": "Default layout",
                "placeholders": ["title", "content"],
                "design_tips": ["Keep it simple", "Use consistent formatting"],
                "color_suggestion": "primary"
            }
    
    # ========================================================================
    # CONTENT GENERATION & ENHANCEMENT
    # ========================================================================
    
    async def generate_slide_content(
        self,
        slide_title: str,
        key_message: str,
        bullet_count: int = 4
    ) -> Dict[str, Any]:
        """
        Generate compelling slide content from a title and key message.
        
        Args:
            slide_title: Slide title
            key_message: Main message to convey
            bullet_count: Number of bullet points desired
            
        Returns:
            Generated content with title, bullets, speaker_notes
        """
        prompt = f"""You are a professional presentation content writer.

SLIDE TITLE: {slide_title}
KEY MESSAGE: {key_message}
BULLET COUNT: {bullet_count} points

Generate compelling, concise slide content:
1. Refine the title if needed (keep under 10 words)
2. Create {bullet_count} bullet points (each under 15 words)
3. Add speaker notes (2-3 sentences)
4. Suggest a visual element

Return JSON:
{{
  "title": "...",
  "bullets": ["point 1", "point 2", ...],
  "speaker_notes": "...",
  "visual_suggestion": "..."
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                temperature=0.4,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return {
                "title": slide_title,
                "bullets": [key_message[:100]],
                "speaker_notes": key_message,
                "visual_suggestion": "Relevant professional image"
            }
    
    async def enhance_text_for_presentation(
        self,
        text: str,
        max_words: int = 50
    ) -> str:
        """
        Enhance and condense text for presentation slides.
        
        Args:
            text: Original text to enhance
            max_words: Maximum word count
            
        Returns:
            Enhanced, presentation-ready text
        """
        prompt = f"""You are a presentation editor. Condense and enhance this text for a slide.

ORIGINAL TEXT:
{text}

MAXIMUM WORDS: {max_words}

Guidelines:
1. Keep only essential information
2. Use active voice
3. Make it scannable (short phrases, not sentences)
4. Maintain professional tone
5. Start bullet points with action verbs if applicable

Return ONLY the enhanced text, no explanations."""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                temperature=0.2,
                max_tokens=200,
            )
            
            return content.strip()
        except Exception as e:
            logger.error(f"Text enhancement failed: {e}")
            return text[:max_words*6]  # Rough character limit
    
    # ========================================================================
    # DESIGN & VISUAL SUGGESTIONS
    # ========================================================================
    
    async def suggest_color_palette(
        self,
        topic: str,
        mood: str = "professional"
    ) -> Dict[str, str]:
        """
        Suggest appropriate color palette for presentation.
        
        Args:
            topic: Presentation topic
            mood: Desired mood (professional, energetic, calm, etc.)
            
        Returns:
            Color palette with primary, secondary, accent colors
        """
        prompt = f"""You are a presentation design expert. Suggest a color palette.

TOPIC: {topic}
MOOD: {mood}

Available palettes:
- midnight_executive: Deep blue, professional, corporate
- forest_moss: Green tones, natural, growth
- coral_energy: Vibrant, energetic, creative
- warm_terracotta: Earthy, warm, inviting
- ocean_gradient: Blue tones, trustworthy, stable
- charcoal_minimal: Minimal, modern, sophisticated
- teal_trust: Teal, trustworthy, balanced
- berry_cream: Rich, elegant, premium
- sage_calm: Calming, balanced, harmonious
- cherry_bold: Bold, attention-grabbing, dynamic

Return JSON:
{{
  "palette_name": "...",
  "reasoning": "Why this palette fits",
  "primary_hex": "...",
  "secondary_hex": "...",
  "accent_hex": "..."
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                temperature=0.2,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Color suggestion failed: {e}")
            return {
                "palette_name": "midnight_executive",
                "reasoning": "Default professional palette",
                "primary_hex": "1E2761",
                "secondary_hex": "CADCFC",
                "accent_hex": "FFFFFF"
            }
    
    async def suggest_visual_elements(
        self,
        slide_content: str,
        slide_type: str
    ) -> List[Dict[str, str]]:
        """
        Suggest visual elements for a slide.
        
        Args:
            slide_content: Slide content/text
            slide_type: Type of slide (content, data, comparison, etc.)
            
        Returns:
            List of visual suggestions with type, description, placement
        """
        prompt = f"""You are a visual design expert. Suggest visuals for this slide.

SLIDE TYPE: {slide_type}
SLIDE CONTENT:
{slide_content[:500]}

Suggest 2-3 visual elements that would enhance this slide.
Consider: charts, icons, images, diagrams, infographics.

Return JSON array:
[
  {{
    "type": "icon|chart|image|diagram",
    "description": "What visual to use",
    "placement": "top|bottom|left|right|center",
    "purpose": "Why this visual helps"
  }}
]"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.SPEED,
                temperature=0.3,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Visual suggestion failed: {e}")
            return []
    
    # ========================================================================
    # QUALITY & CONSISTENCY CHECKS
    # ========================================================================
    
    async def check_presentation_consistency(
        self,
        slide_titles: List[str],
        content_style: str
    ) -> Dict[str, Any]:
        """
        Check presentation for consistency and quality.
        
        Args:
            slide_titles: List of all slide titles
            content_style: Content style (formal, casual, technical, etc.)
            
        Returns:
            Consistency check with issues, suggestions, overall_score
        """
        titles_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(slide_titles))
        
        prompt = f"""You are a presentation quality reviewer. Check for consistency.

SLIDE TITLES:
{titles_str}

CONTENT STYLE: {content_style}

Check for:
1. Title consistency (similar length, style, capitalization)
2. Logical flow between slides
3. Consistent tone and style
4. Clear narrative arc

Return JSON:
{{
  "overall_score": 0-10,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "flow_assessment": "Good|Needs improvement",
  "consistency_notes": "..."
}}"""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                temperature=0.2,
                json_mode=True,
            )
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return {
                "overall_score": 7,
                "issues": [],
                "suggestions": ["Review slide titles for consistency"],
                "flow_assessment": "Good",
                "consistency_notes": "Automated check unavailable"
            }
    
    async def generate_speaker_notes(
        self,
        slide_title: str,
        slide_content: List[str],
        duration_minutes: int = 2
    ) -> str:
        """
        Generate speaker notes for a slide.
        
        Args:
            slide_title: Slide title
            slide_content: Bullet points on slide
            duration_minutes: How long to spend on this slide
            
        Returns:
            Speaker notes (natural, conversational)
        """
        content_str = "\n".join(slide_content)
        
        prompt = f"""You are a professional speechwriter. Write speaker notes for this slide.

SLIDE TITLE: {slide_title}
SLIDE CONTENT:
{content_str}
DURATION: {duration_minutes} minutes

Write natural, conversational speaker notes that:
1. Expand on the bullet points (don't just read them)
2. Add context and examples
3. Include transitions to next slide
4. Sound natural when spoken
5. Fit the time allocation (~130 words per minute)

Return ONLY the speaker notes, no explanations."""

        try:
            content = await inference_service.generate(
                messages=[HumanMessage(content=prompt)],
                priority=InferencePriority.QUALITY,
                temperature=0.4,
                max_tokens=duration_minutes * 200,  # ~130 wpm
            )
            
            return content.strip()
        except Exception as e:
            logger.error(f"Speaker notes generation failed: {e}")
            return f"Speaker notes for: {slide_title}"
