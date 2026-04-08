"""Tests for the standalone API router and config section."""


def test_standalone_router_exists():
    from src.api.standalone_api import standalone_router
    assert standalone_router is not None
    assert standalone_router.prefix == "/v1/docwain"


def test_templates_endpoint():
    from src.api.standalone_api import standalone_router
    routes = [r.path for r in standalone_router.routes]
    assert any("/templates" in r for r in routes)


def test_process_endpoint_registered():
    from src.api.standalone_api import standalone_router
    routes = [r.path for r in standalone_router.routes]
    assert any("/process" in r for r in routes)


def test_config_standalone_section():
    from src.api.config import Config
    assert hasattr(Config, "Standalone")
    assert hasattr(Config.Standalone, "ENABLED")
    assert hasattr(Config.Standalone, "MAX_BATCH_FILES")
    assert hasattr(Config.Standalone, "MAX_FILE_SIZE_MB")
