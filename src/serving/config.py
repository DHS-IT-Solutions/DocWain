"""Configuration for dual vLLM instances and Ollama fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class VLLMInstanceConfig:
    """Configuration for a single vLLM serving instance."""

    name: str
    model: str
    port: int
    dtype: str = "fp8"
    kv_cache_dtype: str = "fp8"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.25
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = False
    speculative_model: Optional[str] = None
    guided_decoding_backend: str = "outlines"
    tensor_parallel_size: int = 1
    host: str = "0.0.0.0"

    def to_vllm_args(self) -> list[str]:
        """Build the vLLM CLI argument list for this configuration."""
        args = [
            "--model", self.model,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--kv-cache-dtype", self.kv_cache_dtype,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--guided-decoding-backend", self.guided_decoding_backend,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--host", self.host,
        ]
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
        if self.speculative_model:
            args.extend(["--speculative-model", self.speculative_model])
        return args


# -- Pre-built configurations ------------------------------------------------

FAST_PATH_CONFIG = VLLMInstanceConfig(
    name="fast",
    model="Qwen/Qwen3-14B",
    port=8100,
    dtype="fp8",
    kv_cache_dtype="fp8",
    max_model_len=8192,
    gpu_memory_utilization=0.25,
    enable_prefix_caching=True,
    enable_chunked_prefill=False,
    speculative_model="yuhuili/EAGLE3-Qwen3-14B",
    guided_decoding_backend="outlines",
)

SMART_PATH_CONFIG = VLLMInstanceConfig(
    name="smart",
    model="Qwen/Qwen3.5-27B",
    port=8200,
    dtype="fp8",
    kv_cache_dtype="fp8",
    max_model_len=32768,
    gpu_memory_utilization=0.50,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    speculative_model=None,
    guided_decoding_backend="outlines",
)


@dataclass(frozen=True)
class OllamaFallbackConfig:
    """Configuration for falling back to the local Ollama instance."""

    host: str = "http://localhost:11434"
    model: str = "qwen3:14b"
    timeout_s: float = 300.0


OLLAMA_FALLBACK_CONFIG = OllamaFallbackConfig()


def get_openai_base_url(config: VLLMInstanceConfig) -> str:
    """Return the OpenAI-compatible base URL for a vLLM instance."""
    return f"http://localhost:{config.port}/v1"
