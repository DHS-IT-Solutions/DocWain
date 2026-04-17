"""One-shot diagnostic: which gate in Reasoner._check_grounding over-fires?

Temporary file. Deleted after the grounding fix lands in Task 2.
"""
import json
import logging
import re
from pathlib import Path

from src.generation.reasoner import Reasoner, _NUMBER_RE


def test_diagnose_which_gate_fires(caplog):
    caplog.set_level(logging.WARNING, logger="src.generation.reasoner")

    audit_path = Path("tests/quality_audit_results.json")
    audit = json.loads(audit_path.read_text())

    reasoner = Reasoner(llm_gateway=None)

    # Build synthetic evidence per case: we don't have the original evidence
    # so we use the response itself as "evidence" to isolate which gate
    # fires for bare answers. This answers: given real evidence that CONTAINS
    # the answer numbers/words, does _check_grounding still return False?
    for case in audit["results"]:
        answer = case["response_preview"]
        # evidence = answer text (so numbers and words ARE present)
        evidence = [{"text": answer}]
        grounded = reasoner._check_grounding(answer, evidence, doc_context=None)
        nums_in_answer = set(_NUMBER_RE.findall(answer))
        nums_in_evidence = set(_NUMBER_RE.findall(answer))  # same text
        words_in_answer = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', answer))
        words_in_evidence = words_in_answer  # same text
        print(f"\n{case['name']}: grounded={grounded}")
        print(f"  answer_nums={len(nums_in_answer)} evidence_nums={len(nums_in_evidence)}")
        print(f"  answer_words={len(words_in_answer)} evidence_words={len(words_in_evidence)}")
        print(f"  overlap_ratio={len(words_in_answer & words_in_evidence) / max(len(words_in_answer), 1):.2f}")

    # Report which log warnings fired
    print("\n--- WARNINGS EMITTED ---")
    for record in caplog.records:
        print(f"  {record.message}")
