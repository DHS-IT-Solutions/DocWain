"""Generate a synthetic calibration corpus for AWQ quantization.

Per `feedback_no_customer_data_training.md`: synthetic only.

The corpus mixes domain-typical documents DocWain encounters, so the
quantization activation statistics reflect realistic inference traffic
across insurance, medical, HR, procurement, contract, and resume content.

Output: a single .jsonl with `{"text": "..."}` rows. autoawq's calibration
loader can consume this directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

OUT_PATH = Path("data/awq_calibration/calibration_v1.jsonl")
TARGET_SAMPLES = 256  # autoawq default is 128; use 256 for stability


_INSURANCE = """Policy number: SYN-INS-{i:04d}
Policyholder: Test Holder {i}
Coverage: comprehensive automobile, ${ded} deductible.
Premium: ${prem} / year. Effective 2026-{m:02d}-01 to 2026-12-31.
Excludes: flood damage, earthquake, racing events, intentional acts.
Liability limit: ${liab}. Roadside assistance: included.
Endorsements: rental reimbursement, glass coverage rider.
Renewal date: 2026-{rn:02d}-15."""

_MEDICAL = """Patient: Test Patient {i}
DOB: 19{yy:02d}-{m:02d}-{d:02d}. MRN: SYN-MED-{i:04d}
Vitals: BP {bp1}/{bp2}, HR {hr}, Temp 98.{t}F, SpO2 9{sp}%.
Chief complaint: {cc}.
Assessment: {ax}.
Plan: continue current medications, follow-up in 6 weeks for labs.
Active medications: lisinopril 10 mg daily, metformin 500 mg BID.
Allergies: {al}. Family history significant for type 2 diabetes."""

_HR = """Employee: Test Employee {i} (ID: SYN-EMP-{i:04d})
Position: Senior {role}. Department: {dept}. Manager: Test Manager.
Hire date: 2024-{m:02d}-{d:02d}. Status: active, full-time, exempt.
Compensation: base salary ${sal}/year, eligible for annual bonus.
PTO balance: {pto} days remaining. FMLA eligible. At-will employment.
Performance review cycle: annual, last review {rev_score}/5.
Probationary period: completed. Severance: per company policy."""

_PROCUREMENT = """RFP-{i:04d}
Buyer: SYN Corp. Vendor: Test Supplier {i}.
Item: industrial-grade {item}. Quantity: {qty} units. Unit price: ${up}.
Total: ${total}. Payment terms: Net {net} days. Lead time: {lt} days.
Incoterms: FOB Origin. SLA: 99.{sla}% uptime, response within {resp} hours.
MOQ: {moq} units. Early-pay discount: {ep}% if paid within 10 days.
Vendor consolidation note: alternate supplier registered as {alt}."""

_CONTRACT = """Service Agreement #{i:04d}
Effective: 2026-{m:02d}-{d:02d}. Term: {tt} months. Renewal: auto-renewing unless 60-day notice.
Parties: SYN Provider Inc and Test Client {i}.
Scope: managed services for {scope_field}. Deliverables specified in Exhibit A.
Fees: ${fee}/month, plus reasonable expenses pre-approved in writing.
Termination: either party may terminate for cause with 30 days written notice.
Indemnification: mutual, capped at fees paid in trailing 12 months.
Governing law: state of {state}. Confidentiality: 3 years post-termination."""

_RESUME = """Test Candidate {i}
Email: candidate{i}@example.com. Phone: 555-01{p}.
Summary: experienced {role} with {years} years across {sector}.
Skills: Python, SQL, distributed systems, project management.
Experience:
  Senior {role} at SYN Corp (2022-present): led team of {teamsz}.
  {role} at Prior Corp (2019-2022): scaled platform from {pre} to {post} users.
Education: BS Computer Science, State University, 20{ey:02d}.
Certifications: AWS Solutions Architect, PMP."""


def _sample_inurance(i: int) -> str:
    return _INSURANCE.format(
        i=i, ded=500 + (i % 5) * 100, prem=1500 + (i % 12) * 200,
        m=(i % 12) + 1, liab=25000 * (1 + (i % 4)),
        rn=((i * 3) % 12) + 1,
    )


def _sample_medical(i: int) -> str:
    return _MEDICAL.format(
        i=i, yy=60 + (i % 30), m=(i % 12) + 1, d=(i % 28) + 1,
        bp1=120 + (i % 30), bp2=70 + (i % 20), hr=60 + (i % 30),
        t=i % 10, sp=5 + (i % 5),
        cc=("chest pain" if i % 4 == 0 else "fatigue" if i % 4 == 1 else "headache" if i % 4 == 2 else "annual physical"),
        ax=("essential hypertension" if i % 3 == 0 else "type 2 diabetes mellitus" if i % 3 == 1 else "GERD"),
        al=("penicillin" if i % 3 == 0 else "no known drug allergies" if i % 3 == 1 else "shellfish"),
    )


def _sample_hr(i: int) -> str:
    roles = ["Software Engineer", "Product Manager", "Data Scientist", "Analyst"]
    depts = ["Engineering", "Product", "Data Platform", "Operations"]
    return _HR.format(
        i=i, role=roles[i % 4], dept=depts[i % 4],
        m=(i % 12) + 1, d=(i % 28) + 1,
        sal=80000 + (i % 12) * 10000, pto=5 + (i % 20),
        rev_score=3 + (i % 3),
    )


def _sample_procurement(i: int) -> str:
    items = ["server racks", "office chairs", "cleaning supplies", "raw materials"]
    return _PROCUREMENT.format(
        i=i, item=items[i % 4], qty=10 + (i % 100),
        up=50 + (i % 500), total=1000 + (i * 73),
        net=30 + (i % 4) * 15, lt=14 + (i % 30),
        sla=900 + (i % 99), resp=2 + (i % 24),
        moq=5 + (i % 50), ep=2 + (i % 5),
        alt=f"SYN-VENDOR-{i + 100:04d}",
    )


def _sample_contract(i: int) -> str:
    fields = ["data analytics", "cloud infrastructure", "security operations", "professional services"]
    states = ["CA", "NY", "TX", "WA"]
    return _CONTRACT.format(
        i=i, m=(i % 12) + 1, d=(i % 28) + 1, tt=12 + (i % 24),
        scope_field=fields[i % 4], fee=2000 + (i % 50) * 500,
        state=states[i % 4],
    )


def _sample_resume(i: int) -> str:
    roles = ["Software Engineer", "Data Scientist", "Product Manager", "Solutions Architect"]
    sectors = ["fintech", "healthcare tech", "e-commerce", "government tech"]
    return _RESUME.format(
        i=i, p=str(i)[:5].zfill(5),
        role=roles[i % 4], years=3 + (i % 15),
        sector=sectors[i % 4], teamsz=3 + (i % 12),
        pre=10000 + i * 100, post=100000 + i * 500,
        ey=(15 + i % 10),
    )


_GENERATORS = [_sample_inurance, _sample_medical, _sample_hr, _sample_procurement, _sample_contract, _sample_resume]


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples: List[str] = []
    for i in range(TARGET_SAMPLES):
        gen = _GENERATORS[i % len(_GENERATORS)]
        samples.append(gen(i))
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps({"text": s}) + "\n")
    print(f"Wrote {len(samples)} calibration samples to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
