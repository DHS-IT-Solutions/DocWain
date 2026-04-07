import os
import pytest


def test_default_port():
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.port == 8300


def test_port_from_env(monkeypatch):
    monkeypatch.setenv("TEAMS_APP_PORT", "9000")
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.port == 9000


def test_default_proxy_url():
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.main_app_url == "http://localhost:8000"


def test_default_concurrency():
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.max_concurrent_documents == 3


def test_express_threshold():
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.express_min_chars == 50
