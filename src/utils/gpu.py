"""GPU detection and configuration utilities for DocWain."""

from dataclasses import dataclass


@dataclass
class GPUConfig:
    name: str
    vram_mb: int
    cuda_version: str
    is_high_memory: bool
    available: bool
    use_4bit_quantization: bool
    recommended_embedding_batch_size: int
    recommended_training_batch_size: int
    max_concurrent_models: int


def _build_config_for_gpu(name: str, vram_mb: int, cuda_version: str) -> GPUConfig:
    """Build a GPUConfig for the given GPU based on its VRAM tier."""
    if vram_mb >= 70_000:
        # A100 80GB tier
        return GPUConfig(
            name=name,
            vram_mb=vram_mb,
            cuda_version=cuda_version,
            is_high_memory=True,
            available=True,
            use_4bit_quantization=False,
            recommended_embedding_batch_size=256,
            recommended_training_batch_size=4,
            max_concurrent_models=3,
        )
    elif vram_mb >= 35_000:
        # A100 40GB tier
        return GPUConfig(
            name=name,
            vram_mb=vram_mb,
            cuda_version=cuda_version,
            is_high_memory=True,
            available=True,
            use_4bit_quantization=False,
            recommended_embedding_batch_size=128,
            recommended_training_batch_size=2,
            max_concurrent_models=2,
        )
    elif vram_mb >= 20_000:
        # V100 / A10 tier
        return GPUConfig(
            name=name,
            vram_mb=vram_mb,
            cuda_version=cuda_version,
            is_high_memory=False,
            available=True,
            use_4bit_quantization=True,
            recommended_embedding_batch_size=96,
            recommended_training_batch_size=1,
            max_concurrent_models=2,
        )
    else:
        # T4 / smaller tier
        return GPUConfig(
            name=name,
            vram_mb=vram_mb,
            cuda_version=cuda_version,
            is_high_memory=False,
            available=True,
            use_4bit_quantization=True,
            recommended_embedding_batch_size=64,
            recommended_training_batch_size=1,
            max_concurrent_models=1,
        )


def _build_cpu_config() -> GPUConfig:
    """Build a GPUConfig representing CPU-only execution."""
    return GPUConfig(
        name="CPU",
        vram_mb=0,
        cuda_version="",
        is_high_memory=False,
        available=False,
        use_4bit_quantization=True,
        recommended_embedding_batch_size=16,
        recommended_training_batch_size=1,
        max_concurrent_models=1,
    )


def detect_gpu() -> GPUConfig:
    """Detect the available GPU and return an appropriate GPUConfig.

    Falls back to a CPU config when CUDA is unavailable or an error occurs.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return _build_cpu_config()

        props = torch.cuda.get_device_properties(0)
        name: str = props.name
        vram_mb: int = props.total_memory // (1024 * 1024)

        # Derive the CUDA runtime version string (e.g. "12.4")
        cuda_version_int = torch.version.cuda or ""
        cuda_version = str(cuda_version_int)

        return _build_config_for_gpu(name=name, vram_mb=vram_mb, cuda_version=cuda_version)

    except Exception:
        return _build_cpu_config()
