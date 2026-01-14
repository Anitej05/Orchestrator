"""
Intent Classification System - Fast pattern-based routing for common queries.

This module provides deterministic intent classification using:
1. Pattern matching (no LLM) - for 90% of common queries
2. File-based routing - for document/spreadsheet tasks
3. Fast LLM classification - fallback for complex queries

Industry standard: Route simple queries through deterministic paths.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from pathlib import Path
import os
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from orchestrator.nodes.utils import invoke_llm_with_fallback

logger = logging.getLogger("AgentOrchestrator")


class Intent(BaseModel):
    """Classified user intent with extracted entities"""
    category: str = Field(
        ...,
        description="data_query | document_task | spreadsheet_task | web_navigation | complex_workflow"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    entities: Dict[str, str] = Field(
        default_factory=dict,
        description="Extracted entities: ticker, query, url, file_path, etc."
    )
    tool_hint: Optional[str] = Field(
        None,
        description="Suggested tool name for direct execution"
    )
    requires_context: bool = Field(
        default=False,
        description="Whether this is part of a multi-step workflow"
    )
    reasoning: str = Field(
        default="",
        description="Why this classification was chosen"
    )


class IntentClassifier:
    """Fast intent classification with pattern matching"""
    
    def __init__(self):
        # TIER 1: Pattern-based rules (deterministic, instant)
        self.patterns = self._build_patterns()
        
    def _build_patterns(self) -> List[Tuple[re.Pattern, Intent]]:
        """Build pattern matching rules for common queries"""
        return [
            # WEB NAVIGATION PATTERNS (should NOT use tools - needs browser agent)
            (
                re.compile(r"(navigate to|go to|open|browse|visit|click).*(website|url|page|\\.com|\\.org)", re.IGNORECASE),
                Intent(
                    category="web_navigation",
                    tool_hint=None,
                    confidence=0.90,
                    reasoning="Pattern matched: web navigation (requires browser agent)"
                )
            ),
            # NEWS PATTERNS - Must come before generic web search to avoid conflicts
            # Pattern 1: "news about X", "headlines on Y", "articles regarding Z"
            (
                re.compile(r"(news|headlines|articles|latest news).*(about|on|regarding|for)", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="search_news",
                    confidence=0.90,
                    reasoning="Pattern matched: news search query"
                )
            ),
            # Pattern 2: "X news", "Y headlines" (reverse order)
            (
                re.compile(r"\\b\\w+\\s+(news|headlines|articles)", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="search_news",
                    confidence=0.88,
                    reasoning="Pattern matched: news search query (reverse order)"
                )
            ),
            # Pattern 3: "find/search/get news about X"
            (
                re.compile(r"(find|search|get|fetch|show).*(news|headlines|articles).*(about|on|for|regarding)", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="search_news",
                    confidence=0.92,
                    reasoning="Pattern matched: explicit news search request"
                )
            ),
            # Pattern 4: Top headlines
            (
                re.compile(r"(top headlines|breaking news|latest headlines|news headlines)", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="get_top_headlines",
                    confidence=0.90,
                    reasoning="Pattern matched: top headlines query"
                )
            ),
            # WEB SEARCH PATTERN REMOVED - DELEGATED TO LLM FOR BETTER CONTEXT AWARENESS
            # (Old regex was too aggressive and captured agent "search" tasks)
            # FINANCE PATTERNS - Order matters! More specific patterns first
            # Stock price with company name (most flexible - catches "Tesla stock", "stock of Apple", etc.)
            (
                re.compile(r"(stock|price|quote|share|trading).*(tesla|tsla|apple|aapl|google|googl|microsoft|msft|amazon|amzn|meta|nvidia|nvda|netflix|nflx)", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="get_stock_quote",
                    confidence=0.95,
                    reasoning="Pattern matched: stock price by company name"
                )
            ),
            # Stock price with ticker symbol
            (
                re.compile(r"(stock|price|quote|share|trading).*?([A-Z]{1,6})\\b", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="get_stock_quote",
                    confidence=0.95,
                    reasoning="Pattern matched: stock price query with ticker"
                )
            ),
            (
                re.compile(r"(stock history|historical.*price|price history|ohlc).*?([A-Z]{1,6})", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="get_stock_history",
                    confidence=0.90,
                    reasoning="Pattern matched: stock history query"
                )
            ),
            (
                re.compile(r"(company info|company details|business summary|about.*company).*?([A-Z]{1,6})", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="get_company_info",
                    confidence=0.90,
                    reasoning="Pattern matched: company information query"
                )
            ),
            # WIKIPEDIA PATTERNS
            (
                re.compile(r"(wikipedia|wiki|look up on wiki).*(summary|article|page|about)", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="search_wikipedia",
                    confidence=0.90,
                    reasoning="Pattern matched: Wikipedia query"
                )
            ),
            (
                re.compile(r"(what is|who is|tell me about|explain|define)\\s+(.+?)(\\?|$)", re.IGNORECASE),
                Intent(
                    category="data_query",
                    tool_hint="get_wikipedia_summary",
                    confidence=0.80,
                    reasoning="Pattern matched: definition/explanation query (Wikipedia candidate)"
                )
            ),
        ]
    
    def classify(self, prompt: str, uploaded_files: List[Dict] = None) -> Intent:
        """
        Classify user intent using multi-tier approach.
        
        Args:
            prompt: User's natural language query
            uploaded_files: List of uploaded file objects
            
        Returns:
            Intent object with classification and extracted entities
        """
        uploaded_files = uploaded_files or []
        
        # TIER 1: Pattern matching (instant, deterministic)
        intent = self._pattern_match(prompt)
        if intent:
            logger.info(f"✅ Pattern-matched intent: {intent.tool_hint} (confidence={intent.confidence:.2f})")
            return intent
        
        # TIER 2: File-based routing (no LLM needed)
        if uploaded_files:
            intent = self._file_based_routing(prompt, uploaded_files)
            if intent:
                logger.info(f"✅ File-based intent: {intent.category} (confidence={intent.confidence:.2f})")
                return intent
        
        # TIER 3: LLM Classification (Smart Routing)
        logger.info("⚠️ No pattern match - delegating to LLM classifier")
        return self._classify_with_llm(prompt)
    
    def _pattern_match(self, prompt: str) -> Optional[Intent]:
        """Try to match prompt against known patterns"""
        for pattern, intent_template in self.patterns:
            match = pattern.search(prompt)
            if match:
                # Clone intent and extract entities
                intent = intent_template.model_copy()
                intent.entities = self._extract_entities(prompt, intent.tool_hint)
                return intent
        return None
    
    def _extract_entities(self, prompt: str, tool_hint: Optional[str]) -> Dict[str, str]:
        """Extract entities based on tool requirements"""
        entities = {}
        
        # Extract ticker/symbol
        ticker_match = re.search(r'\b([A-Z]{1,6})\b', prompt)
        if ticker_match:
            entities["ticker"] = ticker_match.group(1)
            entities["symbol"] = ticker_match.group(1)
        
        # Company name to ticker mapping
        company_map = {
            "tesla": "TSLA",
            "apple": "AAPL",
            "google": "GOOGL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "meta": "META",
            "netflix": "NFLX",
            "nvidia": "NVDA"
        }
        for company, ticker in company_map.items():
            if company in prompt.lower():
                entities["ticker"] = ticker
                entities["symbol"] = ticker
                break
        
        # Extract URL
        url_match = re.search(r'https?://[^\s]+', prompt)
        if url_match:
            entities["url"] = url_match.group(0)
        
        # Extract Wikipedia title for "what is X" queries
        if tool_hint == "get_wikipedia_summary":
            # Try to extract subject from "what is X", "who is X", "tell me about X"
            title_patterns = [
                r"what is\s+(.+?)(\?|$)",
                r"who is\s+(.+?)(\?|$)",
                r"tell me about\s+(.+?)(\?|$)",
                r"explain\s+(.+?)(\?|$)",
                r"define\s+(.+?)(\?|$)"
            ]
            for pattern in title_patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    # Clean up common artifacts
                    title = re.sub(r'\s+(please|plz|pls)$', '', title, flags=re.IGNORECASE)
                    entities["title"] = title
                    entities["query"] = title
                    break
        
        # Extract query (everything except command words)
        if tool_hint in ["search_news", "search_wikipedia", "web_search_and_summarize"]:
            query_text = prompt
            # Remove command words at start
            query_text = re.sub(r'^(get|fetch|find|search|show|tell|give)\s+(me\s+)?(the\s+)?(a\s+)?(an\s+)?', '', query_text, flags=re.IGNORECASE)
            # Remove "news about/on" prefix for news queries
            if tool_hint == "search_news":
                query_text = re.sub(r'^news\s+(about|on|for|regarding)\s+', '', query_text, flags=re.IGNORECASE)
            # Remove "search for" prefix
            query_text = re.sub(r'^(search|find|lookup)\s+(for\s+)?', '', query_text, flags=re.IGNORECASE)
            query_text = query_text.strip()
            
            if query_text and "query" not in entities:
                entities["query"] = query_text
        
        return entities
    
    def _file_based_routing(self, prompt: str, uploaded_files: List[Dict]) -> Optional[Intent]:
        """Route based on uploaded file types"""
        file_types = {f.get("file_type") for f in uploaded_files}
        
        if "document" in file_types:
            # Document analysis task
            file_path = uploaded_files[0].get("file_path")
            vector_store = uploaded_files[0].get("vector_store_path")
            
            return Intent(
                category="document_task",
                confidence=0.95,
                entities={
                    "file_path": file_path,
                    "vector_store_path": vector_store or "",
                    "query": prompt
                },
                tool_hint=None,  # Document tasks need agents
                reasoning="File-based routing: document uploaded",
                requires_context=True
            )
        
        if "spreadsheet" in file_types:
            # Spreadsheet task
            file_path = uploaded_files[0].get("file_path")
            file_id = uploaded_files[0].get("file_id")
            
            return Intent(
                category="spreadsheet_task",
                confidence=0.95,
                entities={
                    "file_path": file_path,
                    "file_id": file_id or "",
                    "query": prompt
                },
                tool_hint=None,  # Spreadsheet tasks need agents
                reasoning="File-based routing: spreadsheet uploaded",
                requires_context=True
            )
        
        if "image" in file_types:
            # Image analysis task
            file_path = uploaded_files[0].get("file_path")
            
            return Intent(
                category="image_analysis",
                confidence=0.95,
                entities={
                    "file_path": file_path,
                    "query": prompt
                },
                tool_hint="analyze_image",  # Image tool available
                reasoning="File-based routing: image uploaded"
            )
        
        return None
    
    def _classify_with_llm(self, prompt: str) -> Intent:
        """
        Use LLM to classify intent when patterns fail.
        Distinguishes between Generic Web Search, Specific Agent Tasks, and Complex Workflows.
        """
        try:
            # Initialize LLMs
            primary_llm = ChatCerebras(model="gpt-oss-120b") if os.getenv("CEREBRAS_API_KEY") else None
            fallback_llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct") if os.getenv("NVIDIA_API_KEY") else None
            
            if not primary_llm and not fallback_llm:
                logger.warning("No LLM keys available for intent classification - falling back to generic")
                return self._complex_workflow_fallback(prompt)
                
            system_prompt = """You are the Intent Classifier for an AI Orchestrator.
            Your job is to route user queries to the best category.
            
            Categories:
            1. "agent_task": User wants a specialized agent (Mail/Calendar/Browser Agent) OR tasks involving emails, files, calendar, web automation. Agents accept NATURAL LANGUAGE - you don't need to convert queries.
            2. "data_query": Simple public information queries (weather, stock prices, news) - handled by direct tools.
            3. "web_navigation": Requests to browse/open a specific URL or website.
            4. "complex_workflow": Ambiguous or multi-step requests needing planning.
            
            RULES:
            - If user mentions "agent" OR the task is about emails/files/calendar/browsing: output "agent_task"
            - Agents like Mail Agent accept natural language (e.g., "find emails about idioms") - pass the query as-is
            - Public facts/entities (e.g., "weather in NY"): output "data_query"
            
            Output JSON only matching the schema."""
            
            classification = invoke_llm_with_fallback(
                primary_llm=primary_llm,
                fallback_llm=fallback_llm,
                prompt=f"{system_prompt}\n\nUser Query: {prompt}",
                pydantic_schema=Intent,
            )
            
            if classification:
                # Post-process: specific override for agent tasks
                if classification.category == "agent_task":
                    classification.tool_hint = None # Ensure it goes to Agent selection, not specific tool
                    classification.requires_context = True
                elif classification.category == "data_query" and not classification.tool_hint:
                    classification.tool_hint = "web_search_and_summarize" # Default for data query
                    
                logger.info(f"🧠 LLM Classified Intent: {classification.category} (Hint: {classification.tool_hint})")
                return classification
                
        except Exception as e:
            logger.error(f"LLM Classification failed: {e}")
            
        return self._complex_workflow_fallback(prompt)

    def _complex_workflow_fallback(self, prompt: str) -> Intent:
        """Determine fallback mechanism"""
        prompt_lower = prompt.lower()
        
        # Check for web navigation keywords
        web_keywords = ["navigate", "browse", "click", "open website", "go to"]
        if any(keyword in prompt_lower for keyword in web_keywords):
            return Intent(
                category="web_navigation",
                confidence=0.60,
                entities={"query": prompt},
                tool_hint=None,
                reasoning="Generic classification: web navigation keywords detected"
            )
        
        # Default to complex workflow
        return Intent(
            category="complex_workflow",
            confidence=0.50,
            entities={"query": prompt},
            tool_hint=None,
            reasoning="Generic classification: no clear pattern, treating as complex",
            requires_context=True
        )


# Global classifier instance
_classifier = None

def get_classifier() -> IntentClassifier:
    """Get or create global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def classify_intent(prompt: str, uploaded_files: List[Dict] = None) -> Intent:
    """Convenience function for intent classification"""
    classifier = get_classifier()
    return classifier.classify(prompt, uploaded_files or [])
