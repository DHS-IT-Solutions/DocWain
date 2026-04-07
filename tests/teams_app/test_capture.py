import json
import os
import tempfile
import pytest
from teams_app.signals.capture import SignalCapture


@pytest.fixture
def signal_dir(tmp_path):
    return str(tmp_path)

@pytest.fixture
def capture(signal_dir):
    return SignalCapture(signals_dir=signal_dir)

def test_capture_positive_writes_to_high_quality(capture, signal_dir):
    capture.record(
        query="what is revenue?",
        response="Revenue is $1M",
        sources=[{"title": "report.pdf"}],
        grounded=True,
        context_found=True,
        signal="positive",
        tenant_id="t1",
    )
    path = os.path.join(signal_dir, "high_quality.jsonl")
    assert os.path.exists(path)
    with open(path) as f:
        entry = json.loads(f.readline())
    assert entry["query"] == "what is revenue?"
    assert entry["signal"] == "positive"
    assert entry["source"] == "teams"

def test_capture_negative_writes_to_finetune_buffer(capture, signal_dir):
    capture.record(
        query="bad question",
        response="bad answer",
        sources=[],
        grounded=False,
        context_found=False,
        signal="negative",
        tenant_id="t1",
    )
    path = os.path.join(signal_dir, "finetune_buffer.jsonl")
    assert os.path.exists(path)
    with open(path) as f:
        entry = json.loads(f.readline())
    assert entry["signal"] == "negative"
    assert entry["source"] == "teams"

def test_capture_implicit_writes_to_finetune_buffer(capture, signal_dir):
    capture.record(
        query="q", response="a", sources=[], grounded=True,
        context_found=True, signal="implicit", tenant_id="t1",
    )
    path = os.path.join(signal_dir, "finetune_buffer.jsonl")
    assert os.path.exists(path)

def test_no_document_content_in_signal(capture, signal_dir):
    capture.record(
        query="q", response="a", sources=[{"title": "doc.pdf", "content": "secret data"}],
        grounded=True, context_found=True, signal="positive", tenant_id="t1",
    )
    path = os.path.join(signal_dir, "high_quality.jsonl")
    with open(path) as f:
        entry = json.loads(f.readline())
    for src in entry["sources"]:
        assert "content" not in src
