import os
import pytest


def test_config_defaults():
    """Config loads sensible defaults when no env vars set."""
    env_keys = [
        "STANDALONE_PORT", "VLLM_BASE_URL", "VLLM_MODEL_NAME",
        "VLLM_TIMEOUT", "STANDALONE_MONGODB_URI", "STANDALONE_MONGODB_DB",
        "STANDALONE_ADMIN_SECRET", "STANDALONE_MAX_FILE_SIZE_MB", "STANDALONE_LOG_LEVEL",
    ]
    old_vals = {}
    for k in env_keys:
        old_vals[k] = os.environ.pop(k, None)

    try:
        import importlib
        import standalone.config as cfg_mod
        importlib.reload(cfg_mod)
        from standalone.config import Config

        assert Config.PORT == 8400
        assert Config.VLLM_BASE_URL == "http://localhost:8100/v1"
        assert Config.VLLM_MODEL_NAME == "docwain"
        assert Config.VLLM_TIMEOUT == 120
        assert Config.MONGODB_URI == "mongodb://localhost:27017"
        assert Config.MONGODB_DB == "docwain_standalone"
        assert Config.MAX_FILE_SIZE_MB == 50
        assert Config.LOG_LEVEL == "INFO"
    finally:
        for k, v in old_vals.items():
            if v is not None:
                os.environ[k] = v


def test_config_respects_env_vars():
    """Config reads values from environment variables."""
    os.environ["STANDALONE_PORT"] = "9999"
    os.environ["VLLM_BASE_URL"] = "http://gpu-server:8100/v1"
    os.environ["VLLM_MODEL_NAME"] = "docwain"

    try:
        import importlib
        import standalone.config as cfg_mod
        importlib.reload(cfg_mod)
        from standalone.config import Config

        assert Config.PORT == 9999
        assert Config.VLLM_BASE_URL == "http://gpu-server:8100/v1"
        assert Config.VLLM_MODEL_NAME == "docwain"
    finally:
        del os.environ["STANDALONE_PORT"]
        del os.environ["VLLM_BASE_URL"]
        del os.environ["VLLM_MODEL_NAME"]


def test_config_admin_secret_from_env():
    """ADMIN_SECRET must come from env."""
    os.environ["STANDALONE_ADMIN_SECRET"] = "my-secret-123"
    try:
        import importlib
        import standalone.config as cfg_mod
        importlib.reload(cfg_mod)
        from standalone.config import Config

        assert Config.ADMIN_SECRET == "my-secret-123"
    finally:
        del os.environ["STANDALONE_ADMIN_SECRET"]
