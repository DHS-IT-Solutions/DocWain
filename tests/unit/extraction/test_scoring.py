from tests.extraction_bench.scoring import (
    compute_coverage,
    compute_fidelity,
    compute_hallucination,
    compute_structure,
    score_extraction,
)


def test_coverage_is_1_when_all_expected_blocks_present():
    expected = {"pages": [{"blocks": [{"text": "a"}, {"text": "b"}]}]}
    actual = {"pages": [{"blocks": [{"text": "a"}, {"text": "b"}]}]}
    assert compute_coverage(expected, actual) == 1.0


def test_coverage_is_0_when_any_block_missing():
    expected = {"pages": [{"blocks": [{"text": "a"}, {"text": "b"}]}]}
    actual = {"pages": [{"blocks": [{"text": "a"}]}]}
    assert compute_coverage(expected, actual) == 0.0


def test_fidelity_uses_levenshtein():
    expected = {"pages": [{"blocks": [{"text": "hello world"}]}]}
    actual = {"pages": [{"blocks": [{"text": "hello world"}]}]}
    assert compute_fidelity(expected, actual) == 1.0


def test_fidelity_low_for_mangled_text():
    expected = {"pages": [{"blocks": [{"text": "hello world"}]}]}
    actual = {"pages": [{"blocks": [{"text": "h3llo w0rld"}]}]}
    score = compute_fidelity(expected, actual)
    assert 0.5 < score < 1.0


def test_structure_preserved_for_matching_tables():
    expected = {"pages": [{"tables": [{"rows": [["a", "b"], ["1", "2"]]}]}]}
    actual = {"pages": [{"tables": [{"rows": [["a", "b"], ["1", "2"]]}]}]}
    assert compute_structure(expected, actual) == 1.0


def test_hallucination_penalizes_extra_blocks():
    expected = {"pages": [{"blocks": [{"text": "a"}]}]}
    actual = {"pages": [{"blocks": [{"text": "a"}, {"text": "not in source"}]}]}
    score = compute_hallucination(expected, actual)
    assert score > 0.0


def test_score_extraction_composite():
    expected = {"pages": [{"blocks": [{"text": "hello"}], "tables": []}]}
    actual = {"pages": [{"blocks": [{"text": "hello"}], "tables": []}]}
    total = score_extraction(expected, actual)
    assert total["coverage"] == 1.0
    assert total["fidelity"] == 1.0
    assert total["structure"] == 1.0
    assert total["hallucination"] == 0.0
    assert "weighted" in total
