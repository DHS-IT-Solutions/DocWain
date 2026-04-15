"""Tests for src/finetune/sprint/domain_data.py"""


def test_domain_list():
    from src.finetune.sprint.domain_data import DOMAINS

    expected = ["financial", "legal", "medical", "hr", "insurance", "government", "technical", "education"]
    assert set(DOMAINS.keys()) == set(expected)


def test_generate_domain_detection_examples():
    from src.finetune.sprint.domain_data import generate_domain_examples

    examples = generate_domain_examples("financial", "detection", count=5, seed=42)
    assert len(examples) == 5
    for ex in examples:
        assert "text" in ex
        assert ex["category"] == "domain_detection"
        assert ex["domain"] == "financial"


def test_generate_domain_reasoning_examples():
    from src.finetune.sprint.domain_data import generate_domain_examples

    examples = generate_domain_examples("legal", "reasoning", count=5, seed=42)
    assert len(examples) == 5
    for ex in examples:
        assert ex["category"] == "domain_reasoning"


def test_generate_cross_domain_examples():
    from src.finetune.sprint.domain_data import generate_cross_domain_examples

    examples = generate_cross_domain_examples(count=5, seed=42)
    assert len(examples) == 5
    for ex in examples:
        assert ex["category"] == "cross_domain"


def test_generate_all_domain_data():
    from src.finetune.sprint.domain_data import generate_all_domain_data

    examples = generate_all_domain_data(seed=42)
    assert len(examples) == 12000  # 4000 detection + 6400 reasoning + 1600 cross-domain
    categories = {ex["category"] for ex in examples}
    assert "domain_detection" in categories
    assert "domain_reasoning" in categories
    assert "cross_domain" in categories
