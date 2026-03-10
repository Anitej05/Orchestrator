# agents/integrations_agent/app_detector.py
"""
App Detector

Identifies which Composio app (and routing method) a given task requires.

Detection strategy (two layers):
  1. Keyword / pattern matching – fast, zero LLM cost, high confidence for obvious apps.
  2. LLM-based detection – for ambiguous tasks where keyword matching is insufficient.

Returns:
  {
    "app_slug":  "gmail",               # Composio app slug (lowercase)
    "app_name":  "Gmail",               # Human-readable name
    "confidence": 0.95,                 # 0.0 – 1.0
    "method":    "DEDICATED_AGENT",     # or "INTEGRATIONS_AGENT"
    "agent_id":  "gmail_agent",         # set when method == DEDICATED_AGENT
  }

Dedicated-agent apps bypass the Integrations Agent and are routed directly by
the orchestrator (Brain).  Everything else runs through the Integrations Agent.
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger("integrations_agent.app_detector")

# ---------------------------------------------------------------------------
# Apps that have a dedicated agent in this backend.
# Key = Composio app slug; Value = canonical agent route key (used by Brain).
# ---------------------------------------------------------------------------
DEDICATED_AGENT_APPS: Dict[str, str] = {
    "gmail": "gmail_agent",
    "zohobooks": "zoho_books",
    "zoho_books": "zoho_books",
    "googlesheets": "spreadsheet_agent",
    "excel": "spreadsheet_agent",
}

# ---------------------------------------------------------------------------
# Keyword → app_slug mapping for fast detection.
# Each entry: app_slug → list of keyword patterns (plain strings or regex).
# ---------------------------------------------------------------------------
APP_KEYWORD_MAP: Dict[str, list] = {
    # Communication
    "gmail": [
        r"\bemail\b", r"\bgmail\b", r"\binbox\b", r"\bsend mail\b",
        r"\bmail\b", r"\bdrafter?\b", r"\bemails?\b",
    ],
    "slack": [
        r"\bslack\b", r"\b#[\w-]+\b.*message",
        r"\bslack message\b", r"\bslack channel\b",
    ],
    "discord": [r"\bdiscord\b"],
    "outlook": [r"\boutlook\b", r"\bmicrosoft mail\b"],
    # Productivity
    "notion": [r"\bnotion\b", r"\bnotion page\b", r"\bnotion database\b"],
    "asana": [r"\basana\b", r"\basana task\b"],
    "trello": [r"\btrello\b", r"\btrello board\b", r"\btrello card\b"],
    "linear": [r"\blinear\b", r"\blinear issue\b", r"\blinear ticket\b"],
    "jira": [r"\bjira\b", r"\bjira ticket\b", r"\bjira issue\b"],
    "monday": [r"\bmonday\.?com\b", r"\bmonday board\b"],
    "clickup": [r"\bclickup\b", r"\bclick up\b"],
    # Code / Dev
    "github": [
        r"\bgithub\b", r"\bgit hub\b", r"\bpull request\b",
        r"\bgithub pr\b", r"\bgithub repo\b", r"\bgithub issue\b",
    ],
    "gitlab": [r"\bgitlab\b"],
    # Spreadsheets / Docs
    "googlesheets": [
        r"\bgoogle sheets?\b", r"\bgsheets?\b", r"\bspreadsheet\b",
        r"\b\.xlsx?\b", r"\bcsv.*upload\b",
    ],
    "googledrive": [r"\bgoogle drive\b", r"\bgdrive\b", r"\bdrive folder\b"],
    "googledocs": [r"\bgoogle docs?\b", r"\bgdoc\b"],
    # CRM / Sales
    "hubspot": [r"\bhubspot\b", r"\bhub spot\b"],
    "salesforce": [r"\bsalesforce\b", r"\bsfdc\b"],
    "pipedrive": [r"\bpipedrive\b"],
    # Accounting
    "zohobooks": [
        r"\bzoho books?\b", r"\bzoho invoice\b",
        r"\baccounting\b", r"\binvoice\b", r"\bquotation\b",
    ],
    # Design
    "figma": [r"\bfigma\b", r"\bfigma file\b", r"\bfigma design\b"],
    # Video / Media
    "youtube": [r"\byoutube\b", r"\byoutube channel\b", r"\byoutube video\b"],
    "zoom": [r"\bzoom call\b", r"\bzoom meeting\b", r"\bschedule zoom\b"],
    # Finance
    "stripe": [r"\bstripe\b", r"\bstripe payment\b"],
    "quickbooks": [r"\bquickbooks\b", r"\bquick books\b"],
    # Other
    "shopify": [r"\bshopify\b", r"\bshopify store\b", r"\bshopify order\b"],
    "airtable": [r"\bairtable\b"],
    "dropbox": [r"\bdropbox\b"],
    "box": [r"\bbox\.com\b", r"\bbox folder\b"],
    "intercom": [r"\bintercom\b"],
    "zendesk": [r"\bzendesk\b"],
    "twilio": [r"\btwilio\b", r"\bsms\b"],
    "sendgrid": [r"\bsendgrid\b"],
    "mailchimp": [r"\bmailchimp\b"],
    "typeform": [r"\btypeform\b"],
    "calendly": [r"\bcalendly\b"],
}

# Human-readable display name for each app slug
APP_DISPLAY_NAMES: Dict[str, str] = {
    "gmail": "Gmail",
    "slack": "Slack",
    "discord": "Discord",
    "outlook": "Outlook",
    "notion": "Notion",
    "asana": "Asana",
    "trello": "Trello",
    "linear": "Linear",
    "jira": "Jira",
    "monday": "Monday.com",
    "clickup": "ClickUp",
    "github": "GitHub",
    "gitlab": "GitLab",
    "googlesheets": "Google Sheets",
    "googledrive": "Google Drive",
    "googledocs": "Google Docs",
    "hubspot": "HubSpot",
    "salesforce": "Salesforce",
    "pipedrive": "Pipedrive",
    "zohobooks": "Zoho Books",
    "figma": "Figma",
    "youtube": "YouTube",
    "zoom": "Zoom",
    "stripe": "Stripe",
    "quickbooks": "QuickBooks",
    "shopify": "Shopify",
    "airtable": "Airtable",
    "dropbox": "Dropbox",
    "box": "Box",
    "intercom": "Intercom",
    "zendesk": "Zendesk",
    "twilio": "Twilio",
    "sendgrid": "SendGrid",
    "mailchimp": "Mailchimp",
    "typeform": "Typeform",
    "calendly": "Calendly",
    "excel": "Microsoft Excel",
}


class AppDetector:
    """
    Detect which Composio app a task requires.

    Two-layer detection:
    1. Fast keyword scan (no LLM cost).
    2. LLM fallback for ambiguous tasks.
    """

    def detect_app_from_task(self, task: str) -> Dict[str, Any]:
        """
        Detect the required Composio app from a natural-language task description.

        Args:
            task: Natural-language task string

        Returns:
            Detection result dict (see module docstring).
        """
        task_lower = task.lower()

        # ------------------------------------------------------------------
        # Layer 1: Keyword / regex matching
        # ------------------------------------------------------------------
        best_match: Optional[Dict[str, Any]] = None
        best_score = 0

        for app_slug, patterns in APP_KEYWORD_MAP.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    score += 1

            if score > best_score:
                best_score = score
                best_match = self._build_result(app_slug, confidence=min(0.7 + score * 0.1, 0.99))

        if best_match:
            logger.debug(
                f"[AppDetector] Keyword match: {best_match['app_slug']} "
                f"(confidence={best_match['confidence']:.2f})"
            )
            return best_match

        # ------------------------------------------------------------------
        # Layer 2: LLM-based detection (fallback)
        # ------------------------------------------------------------------
        llm_result = self._detect_via_llm(task)
        if llm_result:
            logger.debug(
                f"[AppDetector] LLM detection: {llm_result['app_slug']} "
                f"(confidence={llm_result['confidence']:.2f})"
            )
            return llm_result

        # ------------------------------------------------------------------
        # Could not determine – return unknown
        # ------------------------------------------------------------------
        logger.warning(f"[AppDetector] Could not detect app for task: {task[:100]}")
        return {
            "app_slug": None,
            "app_name": None,
            "confidence": 0.0,
            "method": "INTEGRATIONS_AGENT",
            "agent_id": None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_result(self, app_slug: str, confidence: float) -> Dict[str, Any]:
        """Build a normalised detection result."""
        app_name = APP_DISPLAY_NAMES.get(app_slug, app_slug.title())
        if app_slug in DEDICATED_AGENT_APPS:
            return {
                "app_slug": app_slug,
                "app_name": app_name,
                "confidence": confidence,
                "method": "DEDICATED_AGENT",
                "agent_id": DEDICATED_AGENT_APPS[app_slug],
            }
        return {
            "app_slug": app_slug,
            "app_name": app_name,
            "confidence": confidence,
            "method": "INTEGRATIONS_AGENT",
            "agent_id": "integrations_agent",
        }

    def _detect_via_llm(self, task: str) -> Optional[Dict[str, Any]]:
        """
        Use an LLM to identify the Composio app from the task description.
        Returns None if detection fails or LLM is unavailable.
        """
        try:
            import os, json
            from backend.services.inference_service import inference_service, InferencePriority

            known_apps = list(APP_KEYWORD_MAP.keys())
            prompt = (
                "You are a routing assistant. Given a task description, identify which "
                "single external app/service needs to be used. Return ONLY valid JSON "
                "like: {\"app_slug\": \"slack\", \"confidence\": 0.9}\n"
                f"Known apps: {', '.join(known_apps)}\n\n"
                f"Task: {task}"
            )

            import asyncio

            async def _call():
                return await inference_service.complete(
                    messages=[{"role": "user", "content": prompt}],
                    priority=InferencePriority.LOW,
                )

            # Run in current event loop or create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We are inside an async context – use a thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, _call())
                        raw = future.result(timeout=8)
                else:
                    raw = loop.run_until_complete(_call())
            except RuntimeError:
                raw = asyncio.run(_call())

            # Parse JSON from response
            if isinstance(raw, dict):
                text = raw.get("content") or raw.get("text") or ""
            else:
                text = str(raw)

            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                app_slug = data.get("app_slug", "").lower()
                confidence = float(data.get("confidence", 0.5))
                if app_slug:
                    return self._build_result(app_slug, confidence)

        except Exception as e:
            logger.debug(f"[AppDetector] LLM detection failed: {e}")

        return None


# Module-level singleton
_app_detector: Optional[AppDetector] = None


def get_app_detector() -> AppDetector:
    """Return the module-level AppDetector singleton."""
    global _app_detector
    if _app_detector is None:
        _app_detector = AppDetector()
    return _app_detector
