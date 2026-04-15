import json
import tempfile
from pathlib import Path


def test_generate_eval_bank_structure():
    from src.finetune.sprint.eval_bank import generate_eval_bank

    examples = generate_eval_bank()
    assert len(examples) == 700

    for ex in examples:
        assert "category" in ex
        assert "prompt" in ex
        assert "reference" in ex
        assert "difficulty" in ex
        assert ex["difficulty"] in ("easy", "medium", "hard")


def test_eval_bank_category_distribution():
    from src.finetune.sprint.eval_bank import generate_eval_bank, CATEGORY_COUNTS

    examples = generate_eval_bank()
    counts = {}
    for ex in examples:
        cat = ex["category"]
        counts[cat] = counts.get(cat, 0) + 1

    for cat, expected in CATEGORY_COUNTS.items():
        assert counts.get(cat, 0) == expected, f"{cat}: got {counts.get(cat, 0)}, expected {expected}"


def test_eval_bank_save_load():
    from src.finetune.sprint.eval_bank import generate_eval_bank, save_eval_bank, load_eval_bank

    examples = generate_eval_bank()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "eval_bank.jsonl"
        save_eval_bank(examples, path)
        loaded = load_eval_bank(path)
        assert len(loaded) == 700
        assert loaded[0]["prompt"] == examples[0]["prompt"]


def test_hallucination_probes_have_unanswerable():
    from src.finetune.sprint.eval_bank import generate_eval_bank

    examples = generate_eval_bank()
    halluc = [e for e in examples if e["category"] == "hallucination_probes"]
    unanswerable = [e for e in halluc if e["reference"].get("answerable") is False]
    assert len(unanswerable) >= 50, "Need at least 50 unanswerable probes"
