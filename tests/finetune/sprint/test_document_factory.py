"""Tests for src/finetune/sprint/document_factory.py"""


def test_generate_document_all_types():
    from src.finetune.sprint.document_factory import generate_document, DOCUMENT_TYPES

    for doc_type in DOCUMENT_TYPES:
        doc = generate_document(doc_type, seed=42)
        assert isinstance(doc, dict), f"Expected dict for type {doc_type}"
        assert "content" in doc, f"Missing 'content' for type {doc_type}"
        assert "type" in doc, f"Missing 'type' for type {doc_type}"
        assert "metadata" in doc, f"Missing 'metadata' for type {doc_type}"
        assert len(doc["content"]) > 50, (
            f"Content too short for type {doc_type}: {len(doc['content'])} chars"
        )


def test_generate_document_has_metadata():
    from src.finetune.sprint.document_factory import generate_document

    doc = generate_document("invoice", seed=42)
    assert doc["type"] == "invoice"
    assert "ground_truth" in doc["metadata"]


def test_generate_batch():
    from src.finetune.sprint.document_factory import generate_batch

    docs = generate_batch(count=10, seed=42)
    assert len(docs) == 10
    types = {d["type"] for d in docs}
    assert len(types) > 1


def test_generate_spreadsheet():
    from src.finetune.sprint.document_factory import generate_document

    doc = generate_document("spreadsheet", seed=42)
    assert "Sheet:" in doc["content"] or "|" in doc["content"]


def test_generate_scanned_degraded():
    from src.finetune.sprint.document_factory import generate_document

    doc = generate_document("scanned_degraded", seed=42)
    assert (
        "OCR" in doc["content"]
        or "scan" in doc["content"].lower()
        or doc["metadata"].get("ground_truth", {}).get("quality")
    )


# ── Additional coverage tests ─────────────────────────────────────────────────

def test_document_types_count():
    from src.finetune.sprint.document_factory import DOCUMENT_TYPES

    assert len(DOCUMENT_TYPES) == 16


def test_deterministic_output():
    from src.finetune.sprint.document_factory import generate_document

    doc1 = generate_document("invoice", seed=99)
    doc2 = generate_document("invoice", seed=99)
    assert doc1["content"] == doc2["content"]
    assert doc1["metadata"] == doc2["metadata"]


def test_different_seeds_differ():
    from src.finetune.sprint.document_factory import generate_document

    doc1 = generate_document("invoice", seed=1)
    doc2 = generate_document("invoice", seed=2)
    assert doc1["content"] != doc2["content"]


def test_invoice_ground_truth_fields():
    from src.finetune.sprint.document_factory import generate_document

    doc = generate_document("invoice", seed=7)
    gt = doc["metadata"]["ground_truth"]
    for key in ("invoice_number", "issue_date", "vendor", "client", "total_due"):
        assert key in gt, f"Missing ground truth key: {key}"
    assert gt["total_due"] > 0


def test_contract_ground_truth_fields():
    from src.finetune.sprint.document_factory import generate_document

    doc = generate_document("contract", seed=7)
    gt = doc["metadata"]["ground_truth"]
    for key in ("contract_id", "party_a", "party_b", "effective_date", "contract_value"):
        assert key in gt, f"Missing ground truth key: {key}"


def test_medical_record_ground_truth_fields():
    from src.finetune.sprint.document_factory import generate_document

    doc = generate_document("medical_record", seed=7)
    gt = doc["metadata"]["ground_truth"]
    for key in ("patient_name", "mrn", "visit_date", "diagnosis", "icd_code"):
        assert key in gt, f"Missing ground truth key: {key}"


def test_financial_statement_ground_truth_fields():
    from src.finetune.sprint.document_factory import generate_document

    doc = generate_document("financial_statement", seed=7)
    gt = doc["metadata"]["ground_truth"]
    for key in ("revenue", "net_income", "total_assets", "equity"):
        assert key in gt, f"Missing ground truth key: {key}"


def test_batch_cycles_through_types():
    from src.finetune.sprint.document_factory import generate_batch, DOCUMENT_TYPES

    # A batch of exactly len(DOCUMENT_TYPES) should contain one of each type
    docs = generate_batch(count=len(DOCUMENT_TYPES), seed=0)
    types_seen = {d["type"] for d in docs}
    assert types_seen == set(DOCUMENT_TYPES)


def test_invalid_type_raises():
    from src.finetune.sprint.document_factory import generate_document
    import pytest

    with pytest.raises(ValueError):
        generate_document("not_a_real_type", seed=1)


def test_resume_ground_truth_fields():
    from src.finetune.sprint.document_factory import generate_document

    doc = generate_document("resume", seed=5)
    gt = doc["metadata"]["ground_truth"]
    for key in ("name", "role", "email", "skills", "degree"):
        assert key in gt, f"Missing ground truth key: {key}"
    assert isinstance(gt["skills"], list) and len(gt["skills"]) >= 5
