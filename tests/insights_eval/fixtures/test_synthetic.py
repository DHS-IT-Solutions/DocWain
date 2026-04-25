from tests.insights_eval.fixtures.synthetic import (
    synthetic_insurance_doc,
    SyntheticDoc,
)


def test_synthetic_doc_shape():
    doc = synthetic_insurance_doc()
    assert isinstance(doc, SyntheticDoc)
    assert doc.domain == "insurance"
    assert doc.text  # non-empty
    assert doc.expected_anomalies  # at least one planted
    assert doc.expected_gaps
