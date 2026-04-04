"""DocWain serving layer — vLLM client, intent routing, fast-path handler."""

from src.serving.config import (
    VLLMInstanceConfig,
    FAST_PATH_CONFIG,
    SMART_PATH_CONFIG,
    OLLAMA_FALLBACK_CONFIG,
    OLLAMA_CLOUD_CONFIG,
    GPU_MODE_FILE,
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
    "OLLAMA_CLOUD_CONFIG",
    "GPU_MODE_FILE",
    "get_openai_base_url",
    "VLLMManager",
    "IntentRouter",
    "RouterResult",
    "FastPathHandler",
]
