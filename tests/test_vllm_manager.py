"""Tests for VLLMManager client-only refactor."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch

import pytest


class FakeVLLMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if "/v1/chat/completions" in self.path:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            response = {
                "choices": [{"message": {"content": f"Response from {body.get('model', 'unknown')}"}}]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_vllm_server():
    server = HTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture()
def gpu_mode_file(tmp_path):
    path = tmp_path / "gpu-mode.json"
    path.write_text(json.dumps({"mode": "serving"}))
    return str(path)


class TestHealthCheck:
    def test_healthy_instance(self, fake_vllm_server):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
        )
        assert mgr.health_check("fast") is True
        assert mgr.health_check("smart") is True

    def test_unhealthy_instance(self):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(fast_url="http://127.0.0.1:19999", smart_url="http://127.0.0.1:19998")
        assert mgr.health_check("fast") is False

    def test_unknown_instance(self, fake_vllm_server):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(fast_url=f"http://127.0.0.1:{fake_vllm_server}")
        assert mgr.health_check("nonexistent") is False


class TestQuery:
    def test_query_fast(self, fake_vllm_server, gpu_mode_file):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
            gpu_mode_file=gpu_mode_file,
        )
        result = mgr.query_fast("Hello")
        assert "Response from" in result

    def test_query_smart(self, fake_vllm_server, gpu_mode_file):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
            gpu_mode_file=gpu_mode_file,
        )
        result = mgr.query_smart("Hello")
        assert "Response from" in result


class TestFallback:
    def test_falls_back_when_vllm_down(self, gpu_mode_file):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(
            fast_url="http://127.0.0.1:19999",
            smart_url="http://127.0.0.1:19998",
            gpu_mode_file=gpu_mode_file,
        )
        with patch.object(mgr, "_query_ollama_cloud", return_value="cloud response") as mock:
            result = mgr.query_fast("Hello")
            mock.assert_called_once()
            assert result == "cloud response"

    def test_training_mode_skips_vllm(self, fake_vllm_server, tmp_path):
        from src.serving.vllm_manager import VLLMManager
        mode_file = tmp_path / "gpu-mode.json"
        mode_file.write_text(json.dumps({"mode": "training"}))
        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
            gpu_mode_file=str(mode_file),
        )
        with patch.object(mgr, "_query_ollama_cloud", return_value="cloud fallback") as mock:
            result = mgr.query_fast("Hello")
            mock.assert_called_once()
            assert result == "cloud fallback"


class TestGPUMode:
    def test_serving_mode(self, gpu_mode_file):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(gpu_mode_file=gpu_mode_file)
        assert mgr.get_gpu_mode() == "serving"

    def test_training_mode(self, tmp_path):
        from src.serving.vllm_manager import VLLMManager
        f = tmp_path / "gpu-mode.json"
        f.write_text(json.dumps({"mode": "training"}))
        mgr = VLLMManager(gpu_mode_file=str(f))
        assert mgr.get_gpu_mode() == "training"

    def test_missing_file_defaults_serving(self, tmp_path):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(gpu_mode_file=str(tmp_path / "nonexistent.json"))
        assert mgr.get_gpu_mode() == "serving"

    def test_get_active_backends(self, fake_vllm_server):
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
        )
        backends = mgr.get_active_backends()
        assert backends["fast"] is True
        assert backends["smart"] is True
