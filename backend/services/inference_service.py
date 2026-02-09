
import os
import logging
import asyncio
import time
import re
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dataclasses import dataclass, asdict

from utils.key_manager import key_manager, get_cerebras_key, report_rate_limit

logger = logging.getLogger("InferenceService")

class ProviderType(str, Enum):
    CEREBRAS = "cerebras"
    GROQ = "groq"
    NVIDIA = "nvidia"
    OPENAI = "openai" # Conceptual, if added later

class InferencePriority(str, Enum):
    SPEED = "speed"       # Prefer fast providers (Cerebras, Groq)
    QUALITY = "quality"   # Prefer massive models (NVIDIA Llama 405b)
    COST = "cost"         # Prefer free/cheap tiers

@dataclass
class KeyUsageMetrics:
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    request_count: int = 0
    errors: int = 0


class InferenceService:
    """
    Centralized service for ALL LLM inference (Text & Vision).
    Features:
    - Multi-provider fallback (Cerebras -> Groq -> NVIDIA)
    - Key rotation (via KeyManager)
    - Response stripping (Think tags, Markdown)
    - Unified error handling
    - Metrics & Caching
    """
    
    def __init__(self):
        self._initialized = True
        # Default fallback order for standard queries
        self._default_providers = [
            ProviderType.CEREBRAS,
            ProviderType.GROQ, 
            ProviderType.NVIDIA
        ]
        
        # Metrics storage
        self._metrics: Dict[str, KeyUsageMetrics] = {
            p.value: KeyUsageMetrics() for p in ProviderType
        }
        self._metrics["total"] = KeyUsageMetrics()
        
        # Simple in-memory LRU cache (Dict[str, str])
        self._cache: Dict[str, str] = {}
        self._cache_size = 1000
        
    def get_metrics(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {k: asdict(v) for k, v in self._metrics.items()}

    def _update_metrics(self, provider: str, input_len: int, output_len: int, is_error: bool = False):
        """Update usage metrics for a provider."""
        # Estimated token counts (chars / 4)
        in_tokens = input_len // 4
        out_tokens = output_len // 4
        
        # Approximate costs (USD per 1M tokens) - generic mix
        COST_MAP = {
            ProviderType.CEREBRAS: 0.0, # Currently free-ish or low
            ProviderType.GROQ: 0.0, # Free tier
            ProviderType.NVIDIA: 0.0, # Free tier
            ProviderType.OPENAI: 5.0 # Placeholder
        }
        cost = (in_tokens + out_tokens) / 1_000_000 * COST_MAP.get(provider, 0.0)
        
        # Update Provider Metrics
        if provider in self._metrics:
            m = self._metrics[provider]
            if is_error:
                m.errors += 1
            else:
                m.request_count += 1
                m.input_tokens += in_tokens
                m.output_tokens += out_tokens
                m.total_tokens += (in_tokens + out_tokens)
                m.estimated_cost_usd += cost
                
        # Update Total Metrics
        t = self._metrics["total"]
        if is_error:
            t.errors += 1
        else:
            t.request_count += 1
            t.input_tokens += in_tokens
            t.output_tokens += out_tokens
            t.total_tokens += (in_tokens + out_tokens)
            t.estimated_cost_usd += cost

    def _get_cache_key(self, messages: List[BaseMessage], model: str, temperature: float) -> str:
        """Generate a deterministic cache key."""
        # Simple string concatenation of message content + params
        msg_str = "|".join([f"{m.type}:{m.content}" for m in messages])
        return f"{model}|{temperature}|{msg_str}"

    async def generate(
        self, 
        messages: List[BaseMessage], 
        model_name: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        priority: InferencePriority = InferencePriority.SPEED,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        images: Optional[List[str]] = None,  # New: Base64 strings or URLs
        strip_think_tags: bool = True,
        strip_markdown: bool = False,
        json_mode: bool = False,
        fallback_enabled: bool = True,
        use_cache: bool = True
    ) -> str:
        """
        Main generation method. Supports Text and Vision.
        """
        # Inject images into the last HumanMessage if provided
        if images:
            # Reconstruct messages with image blocks
            # Find the last human message
            last_human_idx = -1
            for i, m in enumerate(reversed(messages)):
                if isinstance(m, HumanMessage):
                    last_human_idx = len(messages) - 1 - i
                    break
            
            if last_human_idx != -1:
                original_content = messages[last_human_idx].content
                new_content = [{"type": "text", "text": original_content}]
                for img in images:
                    # Detect if URL or Base64
                    img_data = img if img.startswith("http") or img.startswith("data:") else f"data:image/jpeg;base64,{img}"
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": img_data}
                    })
                messages[last_human_idx] = HumanMessage(content=new_content)
        
        # Check Cache
        if use_cache:
            cache_key = self._get_cache_key(messages, model_name or "default", temperature)
            if cache_key in self._cache:
                logger.info("⚡ Inference Cache Hit")
                return self._cache[cache_key]
        
        # Determine provider order
        provider_order = self._get_provider_order(provider, priority)
        if images:
            # Filter for providers that support vision (currently NVIDIA, OpenAI, or specific others)
            # For now, we trust the caller knows what they are doing or we fallback to NVIDIA/OLLAMA
            # Explicitly favor NVIDIA for vision if not specified
            if not provider:
                provider_order = [ProviderType.NVIDIA, ProviderType.OPENAI] 
                
        last_error = None
        input_len_estimate = 0 # Difficult to estimate with images
        
        for current_provider in provider_order:
            try:
                logger.info(f"🤖 Inference: Using {current_provider} (Priority: {priority}, Vision: {bool(images)})")
                
                llm = self._get_llm_client(current_provider, model_name, temperature, max_tokens, json_mode)
                if not llm:
                    logger.warning(f"⏩ Skipping {current_provider} (Not configured/No Key)")
                    continue
                
                start_time = time.time()
                response = await llm.ainvoke(messages)
                duration = time.time() - start_time
                
                content = response.content
                
                # Update Metrics
                self._update_metrics(current_provider, input_len_estimate, len(content))
                logger.info(f"✅ Inference Success: {current_provider} ({duration:.2f}s)")
                
                # Post-processing
                if strip_think_tags:
                    content = self._strip_think_tags(content)
                
                if strip_markdown or json_mode:
                    content = self._strip_markdown(content)
                
                # Cache Result
                if use_cache:
                    if len(self._cache) >= self._cache_size:
                        first_key = next(iter(self._cache))
                        self._cache.pop(first_key)
                    self._cache[cache_key] = content
                
                return content
                
            except Exception as e:
                logger.error(f"❌ Inference Failed with {current_provider}: {e}")
                self._update_metrics(current_provider, 0, 0, is_error=True)
                last_error = e
                
                if not fallback_enabled:
                    raise e
                    
        raise Exception(f"All inference providers failed. Last error: {last_error}")

    async def generate_structured(
        self,
        messages: List[BaseMessage],
        schema: Any, # Pydantic model class
        model_name: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        priority: InferencePriority = InferencePriority.SPEED,
        temperature: float = 0.1,
        fallback_enabled: bool = True
    ) -> Any:
        """
        Generate structured output conforming to a Pydantic schema.
        """
        provider_order = self._get_provider_order(provider, priority)
        last_error = None
        input_char_len = sum(len(m.content) for m in messages)
        
        for current_provider in provider_order:
            try:
                logger.info(f"🤖 Inference (Structured): Using {current_provider}")
                llm = self._get_llm_client(current_provider, model_name, temperature, 4000, json_mode=True)
                if not llm: continue
                
                # Use standard LangChain structured output interface
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(schema)
                    result = await structured_llm.ainvoke(messages)
                    
                    if result:
                        # Update Metrics (Rough estimate for structured obj)
                        self._update_metrics(current_provider, input_char_len, 500) 
                        return result
                    else:
                        logger.warning(f"⚠️ {current_provider} returned None for structured output, trying fallback...")
                
                # Fallback implementation
                formatted_messages = list(messages)
                schema_json = schema.model_json_schema()
                instruction = f"\n\nRespond with valid JSON that matches this schema:\n{schema_json}"
                
                if isinstance(formatted_messages[-1], HumanMessage):
                    formatted_messages[-1].content += instruction
                else:
                    formatted_messages.append(HumanMessage(content=instruction))
                
                response = await llm.ainvoke(formatted_messages)
                content = self._strip_think_tags(response.content)
                content = self._strip_markdown(content)
                
                self._update_metrics(current_provider, input_char_len, len(content))
                
                # Parse JSON
                import json
                try:
                    # Clean the content
                    cleaned_content = self._strip_think_tags(content)
                    cleaned_content = self._strip_markdown(cleaned_content)
                    
                    # More robust JSON extraction using raw_decode
                    def extract_json(text):
                        # Find the first possible start of a JSON object or array
                        start_idx = text.find('{')
                        array_start = text.find('[')
                        if start_idx == -1 or (array_start != -1 and array_start < start_idx):
                            start_idx = array_start
                            
                        if start_idx == -1:
                            return None
                            
                        text = text[start_idx:]
                        decoder = json.JSONDecoder()
                        try:
                            # raw_decode returns the object and the index where it ended
                            obj, idx = decoder.raw_decode(text)
                            return obj
                        except Exception as e:
                            # If it failed, maybe there's garbage at the end that confused it
                            # We can try to manually find the last closing brace
                            for i in range(len(text), start_idx, -1):
                                try:
                                    return json.loads(text[:i])
                                except:
                                    continue
                        return None

                    data = extract_json(cleaned_content)
                    if data is not None:
                        return schema.model_validate(data)
                        
                    # Fallback to direct loads
                    data = json.loads(cleaned_content)
                    return schema.model_validate(data)
                except Exception as parse_error:
                    logger.warning(f"JSON Parse Error on {current_provider}: {parse_error}. Content snippet: {content[:200]}...")
                    last_error = parse_error
                    continue # Try next provider

            except Exception as e:
                logger.error(f"❌ Structured Inference Failed with {current_provider}: {e}")
                self._update_metrics(current_provider, input_char_len, 0, is_error=True)
                last_error = e
                if not fallback_enabled: raise e
        
        raise Exception(f"All structural inference providers failed. Last error: {last_error}")

    def _get_provider_order(self, requested_provider: Optional[ProviderType], priority: InferencePriority) -> List[ProviderType]:
        """Determine the sequence of providers to try."""
        if requested_provider:
            return [requested_provider] + [p for p in self._default_providers if p != requested_provider]
            
        if priority == InferencePriority.QUALITY:
             return [ProviderType.NVIDIA, ProviderType.CEREBRAS, ProviderType.GROQ]
        
        # Default SPEED
        return self._default_providers

    def _get_llm_client(self, provider: ProviderType, model: Optional[str], temp: float, max_tokens: int, json_mode: bool):
        """Instantiate the LangChain client for the provider."""
        try:
            if provider == ProviderType.CEREBRAS:
                api_key = get_cerebras_key()
                if not api_key: return None
                
                # Model validation: Cerebras only supports their specific models
                model_to_use = model if model and ("gpt-oss" in model or "llama" in model.lower()) else "gpt-oss-120b" 
                return ChatCerebras(model=model_to_use, api_key=api_key, temperature=temp, max_tokens=max_tokens)
                
            elif provider == ProviderType.GROQ:
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key: return None
                
                # Model validation: Groq models typically start with llama, mixtral, or gemma
                model_to_use = model if model and any(x in model.lower() for x in ["llama", "mixtral", "gemma", "gpt"]) else "openai/gpt-oss-120b"
                return ChatGroq(model=model_to_use, api_key=api_key, temperature=temp, max_tokens=max_tokens)
                
            elif provider == ProviderType.NVIDIA:
                api_key = os.getenv("NVIDIA_API_KEY")
                if not api_key: return None
                
                # Model validation: NVIDIA models usually have a prefix or are llama
                model_to_use = model if model and ("meta/" in model or "nvidia/" in model or "llama" in model.lower() or "minimax" in model.lower()) else "minimaxai/minimax-m2.1"
                
                # Handling for JSON mode if supported or using model_kwargs
                from langchain_nvidia_ai_endpoints import ChatNVIDIA
                return ChatNVIDIA(
                    model=model_to_use, 
                    api_key=api_key, 
                    temperature=temp, 
                    max_tokens=max_tokens
                )
            
            elif provider == ProviderType.OPENAI: # Also handles OLLAMA via OpenAI compatibility
                 # Check for local OLLAMA
                 base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
                 api_key = os.getenv("OLLAMA_API_KEY", "ollama") # Dummy key
                 model_to_use = model or "llama3.2-vision"
                 
                 from langchain_openai import ChatOpenAI
                 return ChatOpenAI(base_url=base_url, api_key=api_key, model=model_to_use, temperature=temp, max_tokens=max_tokens)

        except Exception as e:
            logger.error(f"Failed to initialize {provider} client: {e}")
            return None
        return None

    def _strip_think_tags(self, text: str) -> str:
        """
        Remove thinking/reasoning tags from LLM output.
        Handles ALL known formats from various models including:
        - Standard: <think>...</think>
        - Minimax: <|thinking|>...</|thinking|> and similar pipe formats
        - DeepSeek: <thought>...</thought>
        - Chinese: 【thinking】...【/thinking】
        - Reasoning: <reasoning>...</reasoning>
        """
        if not isinstance(text, str):
            return text
        
        # Pattern 1: <think>...</think> (closed tags)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Pattern 2: <think>... (unclosed - GREEDY to end)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Pattern 3: Minimax pipe format <|thinking|>...</|thinking|>
        text = re.sub(r'<\|thinking\|>.*?</\|thinking\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<\|thinking\|>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Pattern 4: Minimax alternate format <|thought|>...</|thought|>
        text = re.sub(r'<\|thought\|>.*?</\|thought\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<\|thought\|>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Pattern 5: DeepSeek format <thought>...</thought>
        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thought>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Pattern 6: Chinese thinking tags
        text = re.sub(r'【thinking】.*?【/thinking】', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'【thinking】.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Pattern 7: <reasoning>...</reasoning>
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<reasoning>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Pattern 8: Generic pipe format <|any_tag|>...</|any_tag|> for intermediate reasoning
        text = re.sub(r'<\|[a-z_]+\|>.*?</\|[a-z_]+\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text.strip()

    def _strip_markdown(self, text: str) -> str:
        """Strip markdown code blocks and return raw text."""
        if not isinstance(text, str):
            return text
        # Remove code block fences
        text = re.sub(r'```[\w]*\n', '', text)
        text = re.sub(r'\n```', '', text)
        # Also remove inline backticks if strict? (Usually not desired for variables)
        return text.strip()

# Global Singleton
inference_service = InferenceService()
