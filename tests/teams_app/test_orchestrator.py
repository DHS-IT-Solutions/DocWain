import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from teams_app.pipeline.orchestrator import TeamsAutoOrchestrator
from teams_app.pipeline.fast_path import Pipeline


@pytest.fixture
def orchestrator():
    return TeamsAutoOrchestrator(
        storage=MagicMock(),
        state_store=MagicMock(),
        tenant_manager=MagicMock(),
        signal_capture=MagicMock(),
    )


def test_classify_express(orchestrator):
    assert orchestrator.classify("data.csv") == Pipeline.EXPRESS


def test_classify_full(orchestrator):
    assert orchestrator.classify("report.pdf") == Pipeline.FULL


def test_should_escalate_short_text(orchestrator):
    assert orchestrator.should_escalate("hi") is True


def test_should_not_escalate_long_text(orchestrator):
    assert orchestrator.should_escalate("x" * 200) is False
