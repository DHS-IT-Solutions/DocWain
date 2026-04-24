"""Gateway primary backend swap + identity shim behavior.

Verifies:
- Config.Model.PRIMARY_BACKEND == 'vllm' → gateway._primary is the vLLM client (OpenAICompatibleClient).
- Config.Model.PRIMARY_BACKEND == 'cloud' → gateway._primary is OllamaClient.
- Config.Model.IDENTITY_SHIM_ENABLED == True → outgoing calls have the shim prepended to system prompt.
- Config.Model.IDENTITY_SHIM_ENABLED == False → system prompt unmodified.
"""
from unittest.mock import MagicMock, patch


def test_gateway_uses_vllm_primary_when_flag_is_vllm(monkeypatch):
    monkeypatch.setenv("DOCWAIN_MODEL_PRIMARY_BACKEND", "vllm")
    # Reload Config + gateway modules to pick up env var
    import importlib
    from src.api import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.llm import gateway as gw_mod
    importlib.reload(gw_mod)

    gw = gw_mod.LLMGateway()
    # Primary client should be OpenAICompatibleClient (vLLM) — check class name
    primary_name = type(gw._primary).__name__ if gw._primary else None
    assert primary_name == "OpenAICompatibleClient", f"expected OpenAICompatibleClient, got {primary_name!r}"


def test_gateway_uses_cloud_primary_when_flag_is_cloud(monkeypatch):
    monkeypatch.setenv("DOCWAIN_MODEL_PRIMARY_BACKEND", "cloud")
    import importlib
    from src.api import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.llm import gateway as gw_mod
    importlib.reload(gw_mod)

    gw = gw_mod.LLMGateway()
    primary_name = type(gw._primary).__name__ if gw._primary else None
    assert primary_name == "OllamaClient", f"expected OllamaClient, got {primary_name!r}"


def test_identity_shim_prepended_when_enabled(monkeypatch):
    monkeypatch.setenv("DOCWAIN_MODEL_IDENTITY_SHIM_ENABLED", "true")
    monkeypatch.setenv("DOCWAIN_MODEL_IDENTITY_SHIM_TEXT", "You are DocWain (test).")
    import importlib
    from src.api import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.llm import gateway as gw_mod
    importlib.reload(gw_mod)

    # The gateway folds `system` into the prompt string passed to generate_with_metadata
    # via _call_client (f"{system}\n\n{prompt}"). Capture the first positional arg (the prompt).
    captured = {}

    def fake_generate_with_metadata(self, *args, **kwargs):
        # First positional arg after self is the prompt (with system folded in).
        captured["prompt"] = args[0] if args else kwargs.get("prompt", "")
        captured["system_kwarg"] = kwargs.get("system")
        return "ok", {"model": "test"}

    def fake_generate(self, *args, **kwargs):
        captured["prompt"] = args[0] if args else kwargs.get("prompt", "")
        return "ok"

    from src.llm import clients as clients_mod
    monkeypatch.setattr(clients_mod.OpenAICompatibleClient, "generate", fake_generate, raising=False)
    monkeypatch.setattr(clients_mod.OpenAICompatibleClient, "generate_with_metadata", fake_generate_with_metadata, raising=False)
    monkeypatch.setattr(clients_mod.OllamaClient, "generate", fake_generate, raising=False)
    monkeypatch.setattr(clients_mod.OllamaClient, "generate_with_metadata", fake_generate_with_metadata, raising=False)

    gw = gw_mod.LLMGateway()
    # Call with a system prompt; shim should be present in the combined prompt sent to backend.
    try:
        gw.generate(prompt="hi", system="custom system prompt here")
    except Exception:
        gw.generate_with_metadata(prompt="hi", system="custom system prompt here")

    # The shim + system are folded into the prompt by _call_client; verify both appear.
    seen = captured.get("prompt", "") or ""
    assert "You are DocWain (test)" in seen, f"shim not in prompt sent to backend; saw {seen[:300]!r}"
    assert "custom system prompt here" in seen, f"original system not in prompt; saw {seen[:300]!r}"


def test_identity_shim_absent_when_disabled(monkeypatch):
    monkeypatch.setenv("DOCWAIN_MODEL_IDENTITY_SHIM_ENABLED", "false")
    monkeypatch.setenv("DOCWAIN_MODEL_IDENTITY_SHIM_TEXT", "You are DocWain (test).")
    import importlib
    from src.api import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.llm import gateway as gw_mod
    importlib.reload(gw_mod)

    captured = {}

    def fake_generate_with_metadata(self, *args, **kwargs):
        captured["prompt"] = args[0] if args else kwargs.get("prompt", "")
        return "ok", {"model": "test"}

    def fake_generate(self, *args, **kwargs):
        captured["prompt"] = args[0] if args else kwargs.get("prompt", "")
        return "ok"

    from src.llm import clients as clients_mod
    monkeypatch.setattr(clients_mod.OpenAICompatibleClient, "generate", fake_generate, raising=False)
    monkeypatch.setattr(clients_mod.OpenAICompatibleClient, "generate_with_metadata", fake_generate_with_metadata, raising=False)
    monkeypatch.setattr(clients_mod.OllamaClient, "generate", fake_generate, raising=False)
    monkeypatch.setattr(clients_mod.OllamaClient, "generate_with_metadata", fake_generate_with_metadata, raising=False)

    gw = gw_mod.LLMGateway()
    try:
        gw.generate(prompt="hi", system="custom system prompt here")
    except Exception:
        gw.generate_with_metadata(prompt="hi", system="custom system prompt here")

    seen = captured.get("prompt", "") or ""
    assert "You are DocWain (test)" not in seen, f"shim present when disabled; saw {seen[:300]!r}"
    assert "custom system prompt here" in seen, f"original system missing when shim disabled; saw {seen[:300]!r}"
