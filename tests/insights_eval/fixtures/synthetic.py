"""Synthetic fixtures for capability eval gates.

All content is fabricated. Per `feedback_no_customer_data_training.md`,
no customer data ever appears in adapter examples or eval fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SyntheticDoc:
    document_id: str
    domain: str
    text: str
    expected_anomalies: List[str] = field(default_factory=list)
    expected_gaps: List[str] = field(default_factory=list)
    expected_recommendations: List[str] = field(default_factory=list)


def synthetic_insurance_doc() -> SyntheticDoc:
    text = (
        "Policy number: SYN-INS-001\n"
        "Policyholder: Test Subject A\n"
        "Coverage: comprehensive automobile, $500 deductible.\n"
        "Premium: $1,800 / year. Effective 2026-01-01 to 2026-12-31.\n"
        "Excludes: flood damage, earthquake, racing events.\n"
        "Note: Liability limit $50,000 — well below state-recommended $100,000.\n"
    )
    return SyntheticDoc(
        document_id="SYN-INS-001",
        domain="insurance",
        text=text,
        expected_anomalies=[
            "Liability limit below state-recommended minimum",
        ],
        expected_gaps=[
            "No flood coverage",
            "No earthquake coverage",
        ],
        expected_recommendations=[
            "Increase liability limit to $100,000",
        ],
    )
