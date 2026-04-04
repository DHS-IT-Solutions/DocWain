"""Integration test — verifies the full query pipeline routes through vLLM.

Requires vLLM fast instance running on port 8100.
Skip if not available.
"""

import json
import pytest
from urllib import request as urllib_request
from urllib.error import URLError


def _vllm_available(port: int) -> bool:
    try:
        req = urllib_request.Request(f"http://localhost:{port}/health")
        with urllib_request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


@pytest.mark.skipif(
    not _vllm_available(8100),
    reason="vLLM fast instance not running on port 8100",
)
class TestVLLMIntegration:
    def test_vllm_manager_query_fast(self):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager()
        result = mgr.query_fast("What is DocWain?")
        assert len(result) > 10
        assert isinstance(result, str)

    def test_vllm_manager_health_check(self):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager()
        assert mgr.health_check("fast") is True
        backends = mgr.get_active_backends()
        assert backends["fast"] is True

    def test_vllm_manager_fallback_on_bad_instance(self):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(smart_url="http://localhost:19999")
        # Smart is down, should fall back to Ollama Cloud or local
        result = mgr.query_smart("Hello")
        assert isinstance(result, str)

    def test_gpu_mode_serving(self, tmp_path):
        from src.serving.vllm_manager import VLLMManager

        mode_file = tmp_path / "mode.json"
        mode_file.write_text(json.dumps({"mode": "serving"}))
        mgr = VLLMManager(gpu_mode_file=str(mode_file))
        assert mgr.get_gpu_mode() == "serving"


@pytest.mark.skipif(
    not _vllm_available(8100),
    reason="vLLM fast instance not running on port 8100",
)
class TestVLLMIntentRouting:
    def test_intent_router_classifies(self):
        from src.serving.vllm_manager import VLLMManager
        from src.serving.model_router import IntentRouter

        mgr = VLLMManager()
        router = IntentRouter(mgr)
        result = router.route("What is the total amount on this invoice?")
        assert result.route in ("fast", "smart")
        assert result.intent is not None
