import pytest


@pytest.fixture
def all_flags_off(monkeypatch):
    """Ensure all 25 insights-portal flags are unset (default false)."""
    from src.api.feature_flags import FLAG_NAMES
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield
