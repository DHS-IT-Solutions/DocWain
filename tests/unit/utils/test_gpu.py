"""Unit tests for src/utils/gpu.py"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.utils.gpu import (
    GPUConfig,
    _build_config_for_gpu,
    _build_cpu_config,
    detect_gpu,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_torch_stub(*, cuda_available: bool, total_memory_bytes: int = 0, device_name: str = "", cuda_version: str = "12.4"):
    """Return a minimal torch stub suitable for patching into sys.modules."""
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = MagicMock()
    torch_stub.cuda.is_available = MagicMock(return_value=cuda_available)

    if cuda_available:
        props = MagicMock()
        props.name = device_name
        props.total_memory = total_memory_bytes
        torch_stub.cuda.get_device_properties = MagicMock(return_value=props)

    torch_stub.version = MagicMock()
    torch_stub.version.cuda = cuda_version
    return torch_stub


# ---------------------------------------------------------------------------
# _build_cpu_config
# ---------------------------------------------------------------------------

class TestBuildCpuConfig:
    def test_fields(self):
        cfg = _build_cpu_config()
        assert cfg.name == "CPU"
        assert cfg.vram_mb == 0
        assert cfg.cuda_version == ""
        assert cfg.is_high_memory is False
        assert cfg.available is False
        assert cfg.use_4bit_quantization is True
        assert cfg.recommended_embedding_batch_size == 16
        assert cfg.recommended_training_batch_size == 1
        assert cfg.max_concurrent_models == 1


# ---------------------------------------------------------------------------
# _build_config_for_gpu – tier tests
# ---------------------------------------------------------------------------

class TestBuildConfigForGpu:
    def test_a100_80gb_tier(self):
        cfg = _build_config_for_gpu("NVIDIA A100-SXM4-80GB", 81_920, "12.4")
        assert cfg.use_4bit_quantization is False
        assert cfg.is_high_memory is True
        assert cfg.recommended_embedding_batch_size == 256
        assert cfg.recommended_training_batch_size == 4
        assert cfg.max_concurrent_models == 3
        assert cfg.available is True

    def test_a100_40gb_tier(self):
        cfg = _build_config_for_gpu("NVIDIA A100-PCIe-40GB", 40_960, "12.1")
        assert cfg.use_4bit_quantization is False
        assert cfg.is_high_memory is True
        assert cfg.recommended_embedding_batch_size == 128
        assert cfg.recommended_training_batch_size == 2
        assert cfg.max_concurrent_models == 2
        assert cfg.available is True

    def test_v100_a10_tier(self):
        cfg = _build_config_for_gpu("Tesla V100-SXM2-32GB", 32_768, "11.8")
        assert cfg.use_4bit_quantization is True
        assert cfg.is_high_memory is False
        assert cfg.recommended_embedding_batch_size == 96
        assert cfg.recommended_training_batch_size == 1
        assert cfg.max_concurrent_models == 2
        assert cfg.available is True

    def test_t4_tier(self):
        cfg = _build_config_for_gpu("Tesla T4", 16_160, "11.7")
        assert cfg.use_4bit_quantization is True
        assert cfg.is_high_memory is False
        assert cfg.recommended_embedding_batch_size == 64
        assert cfg.recommended_training_batch_size == 1
        assert cfg.max_concurrent_models == 1
        assert cfg.available is True

    def test_boundary_exactly_70000_mb(self):
        """Exactly 70 000 MB should hit the A100-80GB tier."""
        cfg = _build_config_for_gpu("SomeGPU", 70_000, "12.0")
        assert cfg.recommended_embedding_batch_size == 256
        assert cfg.max_concurrent_models == 3

    def test_boundary_exactly_35000_mb(self):
        """Exactly 35 000 MB should hit the A100-40GB tier."""
        cfg = _build_config_for_gpu("SomeGPU", 35_000, "12.0")
        assert cfg.recommended_embedding_batch_size == 128
        assert cfg.max_concurrent_models == 2

    def test_boundary_exactly_20000_mb(self):
        """Exactly 20 000 MB should hit the V100/A10 tier."""
        cfg = _build_config_for_gpu("SomeGPU", 20_000, "11.8")
        assert cfg.recommended_embedding_batch_size == 96
        assert cfg.max_concurrent_models == 2

    def test_metadata_preserved(self):
        cfg = _build_config_for_gpu("MyGPU", 81_920, "12.4")
        assert cfg.name == "MyGPU"
        assert cfg.vram_mb == 81_920
        assert cfg.cuda_version == "12.4"


# ---------------------------------------------------------------------------
# detect_gpu – integration / mocked
# ---------------------------------------------------------------------------

class TestDetectGpu:
    def test_returns_gpuconfig_instance(self):
        cfg = detect_gpu()
        assert isinstance(cfg, GPUConfig)

    def test_a100_80gb_detected(self):
        torch_stub = _make_torch_stub(
            cuda_available=True,
            total_memory_bytes=81_920 * 1024 * 1024,
            device_name="NVIDIA A100-SXM4-80GB",
            cuda_version="12.4",
        )
        with patch.dict(sys.modules, {"torch": torch_stub}):
            cfg = detect_gpu()

        assert cfg.name == "NVIDIA A100-SXM4-80GB"
        assert cfg.vram_mb == 81_920
        assert cfg.cuda_version == "12.4"
        assert cfg.use_4bit_quantization is False
        assert cfg.recommended_embedding_batch_size == 256
        assert cfg.max_concurrent_models == 3
        assert cfg.available is True

    def test_t4_16gb_detected(self):
        torch_stub = _make_torch_stub(
            cuda_available=True,
            total_memory_bytes=16_160 * 1024 * 1024,
            device_name="Tesla T4",
            cuda_version="11.7",
        )
        with patch.dict(sys.modules, {"torch": torch_stub}):
            cfg = detect_gpu()

        assert cfg.name == "Tesla T4"
        assert cfg.vram_mb == 16_160
        assert cfg.use_4bit_quantization is True
        assert cfg.recommended_embedding_batch_size == 64
        assert cfg.max_concurrent_models == 1

    def test_a100_40gb_detected(self):
        torch_stub = _make_torch_stub(
            cuda_available=True,
            total_memory_bytes=40_960 * 1024 * 1024,
            device_name="NVIDIA A100-PCIe-40GB",
            cuda_version="12.1",
        )
        with patch.dict(sys.modules, {"torch": torch_stub}):
            cfg = detect_gpu()

        assert cfg.name == "NVIDIA A100-PCIe-40GB"
        assert cfg.vram_mb == 40_960
        assert cfg.use_4bit_quantization is False
        assert cfg.recommended_embedding_batch_size == 128
        assert cfg.max_concurrent_models == 2

    def test_cpu_fallback_when_no_cuda(self):
        torch_stub = _make_torch_stub(cuda_available=False)
        with patch.dict(sys.modules, {"torch": torch_stub}):
            cfg = detect_gpu()

        assert cfg.name == "CPU"
        assert cfg.vram_mb == 0
        assert cfg.available is False
        assert cfg.use_4bit_quantization is True
        assert cfg.recommended_embedding_batch_size == 16

    def test_cpu_fallback_when_torch_not_installed(self):
        """Simulate an environment where torch cannot be imported."""
        with patch.dict(sys.modules, {"torch": None}):
            cfg = detect_gpu()

        assert cfg.name == "CPU"
        assert cfg.available is False

    def test_cpu_fallback_on_exception(self):
        """If get_device_properties raises, fall back to CPU."""
        torch_stub = _make_torch_stub(
            cuda_available=True,
            total_memory_bytes=81_920 * 1024 * 1024,
            device_name="NVIDIA A100-SXM4-80GB",
        )
        torch_stub.cuda.get_device_properties = MagicMock(side_effect=RuntimeError("CUDA error"))
        with patch.dict(sys.modules, {"torch": torch_stub}):
            cfg = detect_gpu()

        assert cfg.name == "CPU"
        assert cfg.available is False
