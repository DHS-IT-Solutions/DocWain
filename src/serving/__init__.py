"""DocWain serving layer — unified vLLM client, intent routing, simple-intent handler."""

from src.serving.config import (
    VLLMInstanceConfig,
    DOCWAIN_CONFIG,
    OLLAMA_FALLBACK_CONFIG,
    OLLAMA_CLOUD_CONFIG,
    GPU_MODE_FILE,
    get_openai_base_url,
)
from src.serving.vllm_manager import VLLMManager
from src.serving.model_router import IntentRouter, RouterResult
from src.serving.intelligence_handler import IntelligenceHandler
from src.serving.metrics import record_request, get_metrics, reset_metrics

__all__ = [
    "VLLMInstanceConfig",
    "DOCWAIN_CONFIG",
    "OLLAMA_FALLBACK_CONFIG",
    "OLLAMA_CLOUD_CONFIG",
    "GPU_MODE_FILE",
    "get_openai_base_url",
    "VLLMManager",
    "IntentRouter",
    "RouterResult",
    "IntelligenceHandler",
    "record_request",
    "get_metrics",
    "reset_metrics",
]
