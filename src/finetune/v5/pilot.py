"""V5 pilot — 20-row dress rehearsal of the ensemble pipeline.

Hand-crafted prompts across the most safety-critical capabilities
(extraction, classification, schema adherence, grounded refusal).
Feeds each through the teacher ensemble and reports:

    * Per-capability acceptance rate
    * Mean agreement count
    * Nemotron tiebreak rate
    * Samples of quarantined rows (for manual review)

Gate for proceeding to full 100K run: ≥70% of pilot rows accepted,
≥1 accepted row per capability tested. If pilot fails, tighten the
prompts / system templates and re-pilot — do NOT run the 100K until
the pilot passes.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v5.teacher_ensemble import (
    EnsembleVote,
    _call_nemotron,
    _call_ollama,
    _call_vllm,
    vote,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


# Pilot prompts — hand-written by Claude to cover the critical capabilities.
# Every prompt has a clear "right answer" we can spot-check the consensus
# against. Mix of structured-output (expect_json=True) and free-text.
PILOT: List[Dict[str, Any]] = [
    # -- extraction / schema adherence --
    {
        "capability": "entity_extraction",
        "expect_json": True,
        "system": "You are a precise document extractor. Reply with ONLY a JSON object — no prose, no code fences.",
        "prompt": (
            "Extract the invoice number and total as JSON with keys "
            "'invoice_number' and 'total' (include currency symbol).\n\n"
            "INVOICE No. INV-2025-050\nDate: 22 July 2025\n"
            "Subtotal £5,000\nTax (20%) £1,000\nTotal £6,000"
        ),
    },
    {
        "capability": "entity_extraction",
        "expect_json": True,
        "system": "You are a precise document extractor. Reply with ONLY a JSON object.",
        "prompt": (
            "Extract vendor_name and po_number as JSON.\n\n"
            "AQUARIUS MARKETING\nPurchase Order No. PO508084\n"
            "Billed to: Assurity Ltd, Horsham, UK"
        ),
    },
    {
        "capability": "entity_extraction",
        "expect_json": True,
        "system": "Reply with ONLY a JSON object.",
        "prompt": (
            "Extract {\"effective_date\":..., \"parties\":[...], \"term_months\":...} as JSON.\n\n"
            "This MASTER SERVICES AGREEMENT is entered into as of April 6, 2023 "
            "by Horizon Biotech (Service Provider) and Pinnacle Dynamics (Client). "
            "Term: 36 months."
        ),
    },
    # -- domain recognition --
    {
        "capability": "domain_recognition",
        "expect_json": False,
        "system": (
            "Classify the following document into exactly one of: invoice, "
            "purchase_order, quote, contract, statement, clinical, legal, "
            "resume, technical. Reply with only the label, lowercase."
        ),
        "prompt": (
            "INVOICE No. INV-2025-050. Billed to Assurity Ltd. Subtotal £5,000. Tax £1,000. Total £6,000."
        ),
    },
    {
        "capability": "domain_recognition",
        "expect_json": False,
        "system": (
            "Classify the following document into exactly one of: invoice, "
            "purchase_order, quote, contract, statement, clinical, legal, "
            "resume, technical. Reply with only the label, lowercase."
        ),
        "prompt": (
            "MASTER SERVICES AGREEMENT between Horizon Biotech and Pinnacle Dynamics, "
            "effective April 6 2023, term 36 months."
        ),
    },
    {
        "capability": "domain_recognition",
        "expect_json": False,
        "system": (
            "Classify the following document into exactly one of: invoice, "
            "purchase_order, quote, contract, statement, clinical, legal, "
            "resume, technical. Reply with only the label, lowercase."
        ),
        "prompt": (
            "QUOTATION No. QUT-25-032 valid for 30 days. Prepared for Assurity Ltd. "
            "Marketing and brand development services £60,000."
        ),
    },
    # -- doctype classification (subtype within domain) --
    {
        "capability": "doctype_classification",
        "expect_json": False,
        "system": (
            "The following is an invoice. Subtype: commercial_invoice or proforma_invoice. "
            "Reply with only the label."
        ),
        "prompt": (
            "PROFORMA INVOICE — not for payment. Items to be shipped: 10 units @ $100."
        ),
    },
    # -- grounded refusal (hard negative) --
    {
        "capability": "grounded_refusal",
        "expect_json": False,
        "system": (
            "Answer the question using ONLY the document. If the document does "
            "not contain the answer, say 'NOT_IN_DOCUMENT'."
        ),
        "prompt": (
            "What is the vendor's tax identification number?\n\n"
            "INVOICE No. INV-2025-050. Billed to Assurity Ltd. Total £6,000. Thank you for your business."
        ),
    },
    {
        "capability": "grounded_refusal",
        "expect_json": False,
        "system": (
            "Answer using ONLY the document. If unanswerable from the document, reply 'NOT_IN_DOCUMENT'."
        ),
        "prompt": (
            "When was this invoice paid?\n\n"
            "INVOICE No. INV-25-062 issued 22 Dec 2025. Total £6,000 due."
        ),
    },
    # -- schema adherence --
    {
        "capability": "schema_adherence",
        "expect_json": True,
        "system": (
            "Output ONLY JSON matching this schema: "
            "{\"invoice\": {\"number\": str, \"date\": str, \"total\": str}}. "
            "No extra keys, no missing keys."
        ),
        "prompt": (
            "INVOICE No. INV-25-054. Date: 22 October 2025. Subtotal £5,000. Tax £1,000. Total £6,000."
        ),
    },
    # -- identity in weights (empty system) --
    {
        "capability": "identity_in_weights",
        "expect_json": False,
        "system": "",  # deliberately empty
        "prompt": "Who are you? Answer in one sentence.",
    },
    {
        "capability": "identity_in_weights",
        "expect_json": False,
        "system": "",
        "prompt": "What kind of documents can you work with?",
    },
    # -- intent narrative --
    {
        "capability": "intent_understanding",
        "expect_json": False,
        "system": (
            "In 1-2 sentences, describe the transaction this document represents. "
            "No greetings, no fluff."
        ),
        "prompt": (
            "INVOICE No. INV-2025-050\nDate: 22 July 2025\n"
            "Billed to Assurity Ltd. Subtotal £5,000. Tax £1,000. Total £6,000.\n"
            "Payment information: Briard Bank, Account 103-854-3586."
        ),
    },
    # -- context dependence --
    {
        "capability": "context_dependence",
        "expect_json": False,
        "system": "Answer the question using ONLY the provided document. One-line answer.",
        "prompt": (
            "What is the total?\n\nINVOICE INV-25-050. Subtotal £5,000. Total £6,000."
        ),
    },
    {
        "capability": "context_dependence",
        "expect_json": False,
        "system": "Answer the question using ONLY the provided document. One-line answer.",
        "prompt": (
            "What is the total?\n\nINVOICE INV-25-054. Subtotal £5,000. Total £8,400."
        ),
    },
    # -- cross-doc reasoning --
    {
        "capability": "cross_doc_reasoning",
        "expect_json": False,
        "system": "Compare the invoice vs the PO. Does the invoice match the PO? Reply YES or NO plus one-sentence reason.",
        "prompt": (
            "PO508084: 12 months brand services @ £5,000/month. Total £60,000.\n"
            "INV-25-050: 1 month brand services. Total £6,000 (incl. tax).\n"
        ),
    },
    # -- layout understanding --
    {
        "capability": "layout_understanding",
        "expect_json": False,
        "system": (
            "Identify the region type of the following snippet. One word from: "
            "header, body, table, footer, signature."
        ),
        "prompt": (
            "PAYMENT INFORMATION\nBriard Bank\nAccount Name: AquariusTeam\n"
            "Account No.: 103-854-3586"
        ),
    },
    # -- tool calling (format check) --
    {
        "capability": "tool_calling",
        "expect_json": False,
        "system": (
            "You have a tool retrieve_chunks(query: str, profile_id: str). "
            "Emit a <tool_call>...</tool_call> block in JSON with name/arguments."
        ),
        "prompt": "Find chunks about payment terms in profile 69e260f3.",
    },
    # -- citation discipline --
    {
        "capability": "citation_discipline",
        "expect_json": False,
        "system": (
            "Answer briefly. Every factual claim MUST be followed by "
            "[source: doc_id] where doc_id is provided in the context."
        ),
        "prompt": (
            "What is the total on this invoice?\n\n"
            "doc_id=DOC123: INVOICE INV-25-050. Subtotal £5,000. Total £6,000."
        ),
    },
    # -- domain recognition (hard pair) --
    {
        "capability": "domain_recognition",
        "expect_json": False,
        "system": (
            "Classify: invoice, purchase_order, quote, contract, statement. One word."
        ),
        "prompt": (
            "STATEMENT of ACCOUNT — as of 31 Mar 2026. Opening balance £5,000. Payments £3,000. "
            "Closing balance £2,000."
        ),
    },
]


def run_pilot(output_path: str = "finetune_artifacts/v5/pilot_report.json") -> Dict[str, Any]:
    logger.info("pilot: %d prompts across %d capabilities",
                len(PILOT), len({p["capability"] for p in PILOT}))

    results: List[Dict[str, Any]] = []
    t_start = time.monotonic()
    # Use the lighter-weight teachers only in pilot: V3 + Ollama + Nemotron
    # HF teacher is skipped here because loading the 14B model takes ~2 min
    # and the pilot is meant to be fast. Full 100K run uses all four.
    teachers = [_call_vllm, _call_ollama]

    for idx, row in enumerate(PILOT, start=1):
        t_row = time.monotonic()
        v = vote(
            prompt=row["prompt"],
            expect_json=bool(row.get("expect_json")),
            system_prompt=row.get("system") or "",
            max_tokens=400,
            temperature=0.1,
            call_nemotron_on_tie=True,
            teacher_callers=teachers,
        )
        elapsed = time.monotonic() - t_row
        logger.info(
            "[%2d/%d] cap=%s acc=%s conf=%s agree=%d/%d nemotron=%s elapsed=%.1fs reason=%r",
            idx, len(PILOT), row["capability"], v.accepted, v.confidence,
            v.agreement_count, v.primary_voice_count, v.nemotron_tiebreak,
            elapsed, v.reason,
        )
        results.append({
            "row_id": idx,
            "capability": row["capability"],
            "prompt_preview": row["prompt"][:100],
            "accepted": v.accepted,
            "confidence": v.confidence,
            "agreement": v.agreement_count,
            "primary_voices_present": v.primary_voice_count,
            "nemotron_tiebreak": v.nemotron_tiebreak,
            "reason": v.reason,
            "consensus": v.consensus,
            "teacher_responses": [
                {"teacher": r.teacher, "ok": r.ok, "parsed": r.parsed,
                 "error": r.error, "latency_s": round(r.latency_s, 2)}
                for r in v.teacher_responses
            ],
        })
    total_elapsed = time.monotonic() - t_start

    # Aggregate
    by_cap: Dict[str, Dict[str, int]] = defaultdict(lambda: {"accepted": 0, "quarantine": 0, "total": 0})
    for r in results:
        c = r["capability"]
        by_cap[c]["total"] += 1
        if r["accepted"]:
            by_cap[c]["accepted"] += 1
        else:
            by_cap[c]["quarantine"] += 1

    accepted = sum(1 for r in results if r["accepted"])
    tiebreak = sum(1 for r in results if r["nemotron_tiebreak"])
    acceptance_rate = accepted / max(len(results), 1)

    summary = {
        "total_rows": len(results),
        "accepted": accepted,
        "quarantined": len(results) - accepted,
        "acceptance_rate": round(acceptance_rate, 3),
        "nemotron_tiebreak_rate": round(tiebreak / max(len(results), 1), 3),
        "by_capability": {k: dict(v) for k, v in by_cap.items()},
        "capabilities_with_zero_accepted": [
            k for k, v in by_cap.items() if v["accepted"] == 0
        ],
        "total_elapsed_s": round(total_elapsed, 1),
        "pilot_gate_passed": (
            acceptance_rate >= 0.70
            and not [k for k, v in by_cap.items() if v["accepted"] == 0]
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if k == "by_capability":
            print(f"  by_capability:")
            for cap, stats in v.items():
                print(f"    {cap:26s}  accepted={stats['accepted']}/{stats['total']}")
        else:
            print(f"  {k}: {v}")
    print()
    print(f"Report written to {output_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="finetune_artifacts/v5/pilot_report.json")
    args = ap.parse_args()
    run_pilot(output_path=args.output)


if __name__ == "__main__":
    main()
