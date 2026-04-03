"""DocWain V2 serving layer — dual-model vLLM serving with Ollama fallback."""

from src.serving.config import (
    VLLMInstanceConfig,
    FAST_PATH_CONFIG,
    SMART_PATH_CONFIG,
    OLLAMA_FALLBACK_CONFIG,
    get_openai_base_url,
)
from src.serving.vllm_manager import VLLMManager
from src.serving.model_router import IntentRouter, RouterResult
from src.serving.fast_path import FastPathHandler

__all__ = [
    "VLLMInstanceConfig",
    "FAST_PATH_CONFIG",
    "SMART_PATH_CONFIG",
    "OLLAMA_FALLBACK_CONFIG",
    "get_openai_base_url",
    "VLLMManager",
    "IntentRouter",
    "RouterResult",
    "FastPathHandler",
]
