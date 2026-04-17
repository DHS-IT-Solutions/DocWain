import json
import pytest
import time
from pathlib import Path


def _make_scheduler(tmp_path, mode="serving", training_pending=False):
    mode_file = tmp_path / "gpu-mode.json"
    mode_file.write_text(json.dumps({"mode": mode}))
    queue_file = tmp_path / "training-queue.json"
    queue_file.write_text(json.dumps({"pending": training_pending}))
    from scripts.gpu_scheduler import GPUScheduler
    return GPUScheduler(
        gpu_mode_file=str(mode_file),
        training_queue_file=str(queue_file),
        vllm_url="http://127.0.0.1:19999",
        idle_threshold_minutes=30,
    )


class TestDecision:
    def test_serving_no_training(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        assert sched.decide() == "keep_serving"

    def test_serving_training_not_idle(self, tmp_path):
        sched = _make_scheduler(tmp_path, training_pending=True)
        sched._last_request_time = time.time()
        assert sched.decide() == "keep_serving"

    def test_serving_training_idle(self, tmp_path):
        sched = _make_scheduler(tmp_path, training_pending=True)
        sched._last_request_time = time.time() - 3600
        assert sched.decide() == "enter_training"

    def test_training_complete(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="training", training_pending=False)
        assert sched.decide() == "resume_serving"

    def test_training_in_progress(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="training", training_pending=True)
        assert sched.decide() == "continue_training"


class TestModeFile:
    def test_write_mode(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        sched.set_gpu_mode("training")
        data = json.loads((tmp_path / "gpu-mode.json").read_text())
        assert data["mode"] == "training"
        assert "since" in data

    def test_read_mode(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="training")
        assert sched.get_gpu_mode() == "training"
