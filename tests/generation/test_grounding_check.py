"""Regression-lock the grounding gate fix.

Fixtures cover:
- grounded=True when answer numbers/words trace to evidence (normal case)
- grounded=True when answer is short (< 40 chars — bypass short answers)
- grounded=True for concise "not-found" answers (< 10 meaningful words)
- grounded=False when answer numbers are fabricated (number gate — unchanged)
- grounded=False when answer shares ~zero words with evidence (word gate hard fail)
- grounded=False when evidence is empty
- grounded=True when evidence items contain text reached via doc_context.key_facts
- grounded=True when evidence is paraphrased (e.g., "blood pressure" vs "BP") — the
  15% ratio threshold from before would fail this; the fix must not.
"""
from src.generation.reasoner import Reasoner


def _reasoner():
    return Reasoner(llm_gateway=None)


def test_grounded_when_answer_traces_to_evidence():
    answer = "The invoice total is **$9,000.00** per document INV-42."
    evidence = [{"text": "Invoice INV-42 shows a total of $9,000.00 due on 2026-03-15."}]
    assert _reasoner()._check_grounding(answer, evidence) is True


def test_short_answer_grounded_when_evidence_exists():
    answer = "**$9,000.00**"
    evidence = [{"text": "Invoice total: $9,000.00"}]
    assert _reasoner()._check_grounding(answer, evidence) is True


def test_concise_not_found_answer_grounded_when_evidence_present():
    """Short legitimate 'not-found' answers must not be marked ungrounded.

    Mirrors the Apr 11 FIELD_PHONE failure: the user asked for Gokul's phone
    number, retrieval returned on-topic chunks about Gokul, and the model
    correctly said 'not specified'. The answer shares a topical word
    ('Gokul') with the evidence — the `overlap >= 1` requirement on the
    negation short-circuit is satisfied, and the short-answer gate fires.
    Previously the `overlap < 5` absolute floor made this ungrounded.
    """
    answer = "**Gokul's phone number is not specified in the documents.**"
    evidence = [{"text": "Gokul's resume includes education, work experience, and email but no phone."}]
    assert _reasoner()._check_grounding(answer, evidence) is True


def test_ungrounded_when_numbers_fabricated():
    answer = (
        "The total is **$99,999.00** as of **2099-12-31** with reference **98765**, "
        "charged on invoice **54321** and approved by code **11223** before noon."
    )
    evidence = [{"text": "The invoice shows various line items but no totals."}]
    assert _reasoner()._check_grounding(answer, evidence) is False


def test_ungrounded_when_zero_word_overlap():
    """Answers with essentially no word overlap with evidence are hallucinations."""
    answer = "The quick brown fox jumps over the lazy dog in the meadow at dawn."
    evidence = [{"text": "Bacteria cultures proliferate inside Petri dishes under UV light."}]
    assert _reasoner()._check_grounding(answer, evidence) is False


def test_ungrounded_when_evidence_empty():
    assert _reasoner()._check_grounding("substantive answer text with many words to avoid trivial pass", []) is False


def test_grounded_via_doc_context_key_facts():
    answer = "Candidate **Jessica Jones** has 8 years of experience."
    evidence = [{"text": "unrelated chunk text about weather"}]
    doc_context = {"key_facts": ["Jessica Jones has 8 years of experience as engineering manager."]}
    assert _reasoner()._check_grounding(answer, evidence, doc_context=doc_context) is True


def test_grounded_when_evidence_paraphrases_answer():
    """Paraphrased answers (e.g., medical abbreviation expansion) must stay grounded.

    Matches Task 1's MEDICAL case: answer uses full terms, evidence uses abbreviations.
    Previous 15% ratio threshold was failing this; fix lowers threshold.
    """
    answer = (
        "The patient shows normal blood pressure at 128 over 82 and steady heart rate "
        "of 76 beats per minute, confirming stable vitals during observation."
    )
    evidence = [
        {"text": "BP: 128/82. HR: 76 bpm. Vitals stable throughout monitoring. "
                 "Patient comfortable. Temperature 36.8 C. Oxygen sat 98%."}
    ]
    assert _reasoner()._check_grounding(answer, evidence) is True


def test_negation_short_circuit_requires_topical_overlap():
    """Fabricated entities that embed a negation phrase must NOT short-circuit.

    Example: 'The emergency contact John Doe Smith Jones is not listed in
    employee records' shares zero topical words with the evidence (a resume
    for Jane Doe). The negation short-circuit alone would mark this grounded;
    the `len(answer_words & evidence_words) >= 1` additional requirement
    forces it to fall through to the 5% ratio gate, which rejects it.
    """
    answer = "The emergency contact Kareem Abdul Quincy is not listed in employee records."
    evidence = [{"text": "Jane Doe's resume includes work history at Acme Corp and a degree."}]
    assert _reasoner()._check_grounding(answer, evidence) is False
