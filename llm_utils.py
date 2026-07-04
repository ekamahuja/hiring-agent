"""
Utility functions for LLM providers.
"""

import json
import logging
from typing import Any, Dict, Optional
from models import ModelProvider, OllamaProvider, GeminiProvider
from prompt import MODEL_PROVIDER_MAPPING, GEMINI_API_KEY

logger = logging.getLogger(__name__)


def structured_chat(
    provider: Any,
    model: str,
    system_message: str,
    prompt: str,
    schema: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """One structured-JSON chat round-trip (system + user turn -> parsed dict).

    Shared by the matcher, generator, and evaluator, which previously each
    re-implemented the same provider.chat + extract + json.loads dance. Callers
    validate the returned dict into their own Pydantic model.
    """
    response = provider.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        options={
            "stream": False,
            "temperature": params.get("temperature", 0.3),
            "top_p": params.get("top_p", 0.9),
        },
        format=schema,
    )
    content = extract_json_from_response(response["message"]["content"])
    return json.loads(_isolate_json_object(content))


def _isolate_json_object(text: str) -> str:
    """Trim anything outside the outermost ``{...}`` so json.loads doesn't choke
    on explanatory text/metadata the model appends after the closing brace.

    No brace pair -> return unchanged and let json.loads raise a clear error.
    Object-only: a caller wanting a top-level JSON array must not use this.
    """
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON content from markdown code blocks.

    Args:
        response_text: Text that may contain JSON wrapped in markdown code blocks

    Returns:
        Text with markdown code block syntax removed
    """

    response_text = response_text.strip()
    if "<think>" in response_text:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>")
        if think_start != -1 and think_end != -1:
            response_text = response_text[:think_start] + response_text[think_end + 8 :]

    # Remove leading ```json if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    # Remove trailing ``` if present
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text


def initialize_llm_provider(model_name: str) -> Any:
    """
    Initialize the appropriate LLM provider based on the model name.

    Args:
        model_name: The name of the model to use

    Returns:
        An initialized LLM provider (either OllamaProvider or GeminiProvider)
    """
    # Default to Ollama provider
    provider = OllamaProvider()
    # If using Gemini and API key is available, use Gemini provider
    model_provider = MODEL_PROVIDER_MAPPING.get(model_name, ModelProvider.OLLAMA)
    if model_provider == ModelProvider.GEMINI:
        if not GEMINI_API_KEY:
            logger.warning("⚠️ Gemini API key not found. Falling back to Ollama.")
        else:
            logger.info(f"🔄 Using Google Gemini API provider with model {model_name}")
            provider = GeminiProvider(api_key=GEMINI_API_KEY)
    else:
        logger.info(f"🔄 Using Ollama provider with model {model_name}")
    return provider
