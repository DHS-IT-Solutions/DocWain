"""Seed UAT demo profiles with rich, domain-specialized insights.

Generates synthetic documents across all 6 supported domains
(insurance, medical, hr, procurement, contract, resume), creates a
profile per scenario, and runs researcher v2 end-to-end so the UAT
team opens each profile to working insights / dashboard / actions /
proactive-injection on /api/ask.

Per `feedback_no_customer_data_training.md`: synthetic only.

Usage:
  python scripts/seed_uat_demo_profiles.py [--domains insurance,medical] [--per-domain 5]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Make `src.*` importable when this script is run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# All Insights Portal flags must be on; the running app process has them
# via systemd drop-in, but this script runs in its own process.
for _flag in (
    "INSIGHTS_TYPE_ANOMALY_ENABLED", "INSIGHTS_TYPE_GAP_ENABLED",
    "INSIGHTS_TYPE_COMPARISON_ENABLED", "INSIGHTS_TYPE_SCENARIO_ENABLED",
    "INSIGHTS_TYPE_TREND_ENABLED", "INSIGHTS_TYPE_RECOMMENDATION_ENABLED",
    "INSIGHTS_TYPE_CONFLICT_ENABLED", "INSIGHTS_TYPE_PROJECTION_ENABLED",
    "INSIGHTS_CITATION_ENFORCEMENT_ENABLED", "ADAPTER_AUTO_DETECT_ENABLED",
    "ADAPTER_GENERIC_FALLBACK_ENABLED",
):
    os.environ.setdefault(_flag, "true")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("seed-uat")


_INSURANCE_SCENARIOS = [
    {
        "title": "Underinsured driver with low liability",
        "doc": """Policy number: UAT-INS-001
Policyholder: Demo Subject A
Coverage: comprehensive automobile, $1,000 deductible.
Premium: $1,650 / year. Effective 2026-01-01 to 2026-12-31.
Excludes: flood damage, earthquake, racing events, intentional acts, normal wear.
Liability limit: $25,000 bodily injury per person, $50,000 per accident, $25,000 property damage.
Roadside assistance: included up to $100/incident.
Endorsements: rental reimbursement ($30/day, 30 days max).
Renewal date: 2026-12-15.""",
    },
    {
        "title": "High-net-worth homeowner without umbrella",
        "doc": """Policy number: UAT-INS-002
Policyholder: Demo Subject B (high-net-worth)
Coverage: HO-3 dwelling $1,200,000; personal property $600,000; loss of use $240,000; personal liability $300,000; medical to others $5,000.
Premium: $2,950 / year. Deductible: $2,500 standard, $5,000 wind/hail.
Excludes: flood, earthquake, sinkhole, intentional acts, business pursuits, war.
Endorsements: scheduled jewelry $25,000; identity theft $25,000.
Renewal date: 2026-09-30. No umbrella policy on file.""",
    },
    {
        "title": "Renewal due in 14 days",
        "doc": """Policy number: UAT-INS-003
Policyholder: Demo Subject C
Coverage: comprehensive automobile, $500 deductible.
Premium: $2,200 / year (up 15% from prior term).
Effective 2025-05-10 to 2026-05-10.
Liability limit: $100,000 / $300,000 / $50,000.
Excludes: flood, earthquake, racing.
Renewal date: 2026-05-10. Notice-of-renewal sent 2026-04-12.""",
    },
    {
        "title": "Multi-vehicle policy with one underinsured driver",
        "doc": """Policy number: UAT-INS-004
Policyholder: Demo Subject D (3-driver household)
Coverage:
  - Driver 1 (primary): comprehensive + collision, $500 ded, $250K liability.
  - Driver 2 (spouse): comprehensive + collision, $500 ded, $250K liability.
  - Driver 3 (teenager): liability ONLY, $25,000 / $50,000 / $25,000.
Premium: $4,200 / year combined.
Excludes: flood, earthquake, racing, drag racing, off-road.
Renewal date: 2026-08-01.""",
    },
    {
        "title": "Business policy missing cyber rider",
        "doc": """Policy number: UAT-INS-005
Policyholder: Demo Tech Co (50 employees, SaaS company)
Coverage: BOP general liability $1M / $2M aggregate; property $500K; business interruption 12 months.
Premium: $7,800 / year. Deductible: $1,000.
Excludes: pollution, professional services E&O, cyber events, employment practices, war.
Effective 2026-03-01 to 2027-03-01.""",
    },
]


_MEDICAL_SCENARIOS = [
    {
        "title": "Diabetic with rising A1C and missing follow-up",
        "doc": """Patient: Demo Patient A
DOB: 1965-04-12. MRN: UAT-MED-001
Visit date: 2026-04-20.
Vitals: BP 148/92, HR 84, Temp 98.4F, SpO2 97%, BMI 32.
Chief complaint: routine follow-up.
Assessment: Type 2 diabetes mellitus (E11) — A1C 8.4 (was 7.6 last visit, was 6.9 prior).
  Essential hypertension (I10) — uncontrolled despite lisinopril 10mg.
Active medications: metformin 500mg BID, lisinopril 10mg daily.
Allergies: NKDA.
Plan: continue current meds. Patient declined dose escalation. No follow-up scheduled.""",
    },
    {
        "title": "Patient with allergy and contraindicated medication",
        "doc": """Patient: Demo Patient B
DOB: 1978-09-03. MRN: UAT-MED-002
Visit date: 2026-04-15.
Vitals: BP 122/78, HR 72, Temp 99.1F.
Chief complaint: sinus infection.
Allergies: PENICILLIN (severe, anaphylaxis 2018).
Active medications: amoxicillin 500mg TID (started today by ER physician).
Plan: 7-day antibiotic course, return if symptoms persist.""",
    },
    {
        "title": "Polypharmacy in elderly patient",
        "doc": """Patient: Demo Patient C
DOB: 1942-01-22. MRN: UAT-MED-003
Visit date: 2026-04-10.
Vitals: BP 138/82, HR 68, eGFR 42.
Chief complaint: medication review.
Diagnoses: Hypertension (I10), Type 2 diabetes (E11), Atrial fibrillation, GERD (K21), Hyperlipidemia.
Active medications: lisinopril 20mg daily, metformin 1000mg BID, warfarin 3mg daily, omeprazole 40mg daily, atorvastatin 40mg daily, aspirin 81mg daily, ibuprofen 400mg PRN.
Allergies: sulfa drugs.
Plan: continue current. Lab in 6 months.""",
    },
    {
        "title": "Hypertension trending up across visits",
        "doc": """Patient: Demo Patient D
MRN: UAT-MED-004 (visit 4 of 4 in record)
Visit dates and vitals (most-recent first):
  2026-04-22: BP 156/96, HR 82, eGFR 78
  2026-01-15: BP 148/92, HR 80, eGFR 81
  2025-10-08: BP 142/88, HR 78, eGFR 82
  2025-06-12: BP 132/82, HR 72, eGFR 84
Active medications: lisinopril 10mg daily (started 2025-06-12).
Plan: same.""",
    },
]


_HR_SCENARIOS = [
    {
        "title": "Engineer approaching 1-year cliff with no review scheduled",
        "doc": """Employee: Demo Employee A (ID: UAT-EMP-001)
Position: Senior Software Engineer. Department: Engineering. Manager: Demo Manager.
Hire date: 2025-05-12 (probationary period: 90 days, ended 2025-08-10).
Status: active, full-time, exempt.
Compensation: base salary $145,000/year.
Equity grant: 4,000 RSUs vesting over 4 years with 1-year cliff (cliff date 2026-05-12).
PTO balance: 22 days remaining (max accrual 25 days; capped near limit).
At-will employment.
Annual performance review cycle: due May. No review scheduled yet.""",
    },
    {
        "title": "Offer letter vs employment contract mismatch",
        "doc": """OFFER LETTER (signed 2024-02-15, effective 2024-03-01):
Employee: Demo Employee B
Position: Director of Marketing
Base: $180,000/year. Sign-on bonus: $25,000.
Benefits: medical, dental, vision; 25 PTO days/year.
Equity: 8,000 RSUs over 4 years.

EMPLOYMENT CONTRACT (signed 2024-02-28, effective 2024-03-01):
Position: Senior Marketing Manager
Base: $175,000/year.
Benefits: medical, dental; 20 PTO days/year.
Equity: 6,000 RSUs over 4 years.
Non-compete: 12 months post-termination, nationwide.""",
    },
    {
        "title": "Severance package with non-compete",
        "doc": """Severance Agreement
Employee: Demo Employee C (ID: UAT-EMP-003), terminated 2026-04-30.
Years of service: 7 years. Final base salary: $210,000.
Severance: 12 weeks base = $48,461.
Continued health benefits: 12 weeks under COBRA (employer pays 50%).
Equity: vested as of termination date; no acceleration.
Non-compete: 12 months in same industry, US-only.
Non-solicitation of clients/employees: 12 months.
Release of claims: comprehensive (incl. ADEA, Title VII, FMLA).
Confidentiality of separation: 5 years.""",
    },
]


_PROCUREMENT_SCENARIOS = [
    {
        "title": "SaaS contract with auto-renewal trap",
        "doc": """RFP-UAT-001
Buyer: Demo Corp.
Vendor: ProvCloud Inc.
Service: managed cloud infrastructure (Tier 2).
Contract value: $24,000/month, $288,000/year.
Initial term: 36 months (2024-06-01 to 2027-05-31).
Auto-renewal: 12 months unless 90-day non-renewal notice given.
Payment: Net 45 days. Early-pay discount: 1.5% if paid within 10 days.
SLA: 99.5% uptime; penalty 5% credit per breach point per month.
Lead time on new instances: 14 days.
MOQ: $20,000/month minimum.""",
    },
    {
        "title": "Duplicate vendor — fragmented spend",
        "doc": """Vendor Contract Set:
Vendor A: "ACME Office Supplies" (vendor_id ACME-001)
  - PO 2026-Q1 #4521: stationery, $4,200, Net 30, no discount.
Vendor B: "ACME Supplies LLC" (vendor_id ACME-002)
  - PO 2026-Q1 #4537: stationery, $3,800, Net 30, no discount.
Vendor C: "Acme Office Solutions" (vendor_id ACME-003)
  - PO 2026-Q1 #4612: stationery, $5,100, Net 30, 2% early-pay if paid in 10.
Same physical address listed for all three.""",
    },
    {
        "title": "SLA-breach exposure",
        "doc": """Master Service Agreement
Vendor: HostNow Inc. Customer: Demo Corp.
Service: data center colocation.
Term: 24 months from 2025-09-01.
Monthly fee: $48,000.
SLA targets: 99.99% power uptime, 99.95% network uptime, 4-hour response.
Penalty schedule: 10% credit per 0.01% below uptime SLA per month.
Reported actual uptime past 6 months: 99.85%, 99.91%, 99.99%, 99.93%, 99.88%, 99.92%.
Early-termination fee: 6 months remaining fees.""",
    },
]


_CONTRACT_SCENARIOS = [
    {
        "title": "MSA with one-sided indemnification",
        "doc": """Master Service Agreement (MSA)
Effective Date: 2026-01-15.
Parties: Demo Corp ("Client") and Solutions Co ("Provider").
Term: 24 months, auto-renewing 12-month terms unless notice 60 days prior.

Section 8 (Indemnification):
Client shall indemnify, defend, and hold harmless Provider, its officers,
employees, and affiliates from any third-party claims arising from
Client's use of the services, including but not limited to data
liability, regulatory penalties, and reputational claims.

Section 12 (Limitation of Liability):
Provider's total liability under this Agreement shall not exceed the
fees paid by Client during the immediately preceding 30 days.

Governing Law: State of Delaware. Disputes: AAA arbitration in Wilmington.""",
    },
    {
        "title": "License agreement with broad IP transfer",
        "doc": """Software License Agreement
Effective: 2026-03-01.
Parties: Demo Corp (Licensee), DevWorks Inc (Licensor).
License: perpetual, non-exclusive, non-transferable.
Section 5 (IP):
"All improvements, modifications, derivatives, suggestions, and feedback
provided by Licensee shall become the sole and exclusive property of
Licensor without compensation. This includes but is not limited to bug
reports, feature suggestions, performance benchmarks, and use cases."
Fees: $120,000 perpetual + $24,000/year support.""",
    },
]


_RESUME_SCENARIOS = [
    {
        "title": "Senior engineer candidate with employment gap",
        "doc": """RESUME — Demo Candidate A
Email: candidate-a@example.com. Phone: 555-0142.
Summary: Senior Software Engineer with 11 years experience in distributed systems and cloud infrastructure.

Experience:
  Senior Software Engineer, BigTech Co (2022-08 to 2024-03): led team of 6 engineers; designed event-streaming platform at 2M events/sec.
  [GAP from 2024-03 to 2025-09]
  Staff Software Engineer, ScaleUp Inc (2025-09 to present): architecting kubernetes-native deployment system.
  Software Engineer, MidCo (2017 to 2022): full-stack development.
  Junior Software Engineer, StartCo (2013 to 2017): backend development.

Education: BS Computer Science, State University, 2013.
Skills: Python, Go, Kubernetes, AWS, system design.""",
    },
    {
        "title": "Candidate with skill claims unbacked by experience",
        "doc": """RESUME — Demo Candidate B
Summary: ML Engineer with 4 years experience.
Skills (claimed): TensorFlow, PyTorch, JAX, distributed training, MLOps, LLM fine-tuning, RAG systems, vector databases, Kubernetes, Terraform, AWS, GCP, Azure, Snowflake, Databricks, Airflow, Spark, dbt, Looker, Tableau.

Experience:
  Data Analyst, Demo Corp (2021 to present): SQL queries on financial data; weekly reporting in Looker.
  Intern, University Lab (2020 to 2021): Python scripts for survey analysis.

Education: BS Statistics, 2020.""",
    },
    {
        "title": "Strong candidate for senior leadership role",
        "doc": """RESUME — Demo Candidate C
Summary: VP of Engineering with 16 years experience scaling org from 8 to 240 engineers.

Experience:
  VP of Engineering, ScaleCo (2019 to present):
    - Grew engineering org from 32 to 240 (5 directors, 28 managers).
    - Drove platform reliability from 99.5% to 99.99%.
    - Led migration from monolith to microservices over 18 months.
  Director of Engineering, BigCo (2015 to 2019).
  Engineering Manager, MidCo (2012 to 2015).
  Senior Software Engineer, StartCo (2008 to 2012).

Education: MS Computer Science, Top University, 2008. BS Computer Science, 2006.
Certifications: AWS Solutions Architect Professional. SAFe Agilist.""",
    },
]


_DOMAIN_SCENARIOS = {
    "insurance": _INSURANCE_SCENARIOS,
    "medical": _MEDICAL_SCENARIOS,
    "hr": _HR_SCENARIOS,
    "procurement": _PROCUREMENT_SCENARIOS,
    "contract": _CONTRACT_SCENARIOS,
    "resume": _RESUME_SCENARIOS,
}


def seed_one_profile(domain: str, scenario: dict, *, suffix: str) -> Dict:
    """Run researcher v2 directly on the synthetic doc and persist insights."""
    from src.api.insights_wiring import wire_insights_portal
    wire_insights_portal()
    from src.tasks.researcher_v2 import run_researcher_v2_for_doc

    profile_id = f"uat-demo-{domain}-{suffix}"
    document_id = f"UAT-{domain.upper()[:3]}-{suffix}"

    t0 = time.time()
    result = run_researcher_v2_for_doc(
        document_id=document_id,
        profile_id=profile_id,
        subscription_id="uat-demo-sub",
        document_text=scenario["doc"],
        domain_hint="generic",  # let auto-detect kick in
    )
    elapsed = time.time() - t0
    logger.info("[%s] %s -> %s in %.1fs", domain, scenario["title"][:60], result, elapsed)
    return {"profile_id": profile_id, "scenario": scenario["title"],
            "result": result, "elapsed_s": elapsed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="insurance,medical,hr,procurement,contract,resume")
    parser.add_argument("--per-domain", type=int, default=10**9,
                        help="Limit scenarios per domain (default: all)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    jobs = []
    for d in domains:
        scenarios = _DOMAIN_SCENARIOS.get(d, [])[: args.per_domain]
        for i, sc in enumerate(scenarios):
            jobs.append((d, sc, f"{i+1:02d}-{uuid.uuid4().hex[:6]}"))

    logger.info("Seeding %d profiles across %s with %d workers",
                len(jobs), domains, args.workers)

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(seed_one_profile, d, sc, suffix=sf): (d, sc["title"])
                   for (d, sc, sf) in jobs}
        for fut in as_completed(futures):
            d, title = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                logger.exception("[%s] %s -> ERROR: %s", d, title, exc)
                results.append({"profile_id": "?", "scenario": title, "error": str(exc)})

    total_elapsed = time.time() - t0
    written = sum(r.get("result", {}).get("written", 0) for r in results if "result" in r)
    failed = sum(1 for r in results if "error" in r)
    logger.info("Done: %d profiles seeded with %d total insights in %.1fs (%d failed)",
                len(results) - failed, written, total_elapsed, failed)

    print()
    print("Seeded profiles:")
    for r in sorted(results, key=lambda x: x.get("profile_id", "")):
        if "error" in r:
            print(f"  ERROR  {r['scenario']}: {r['error']}")
        else:
            n = r.get("result", {}).get("written", 0)
            print(f"  {n:2d} insights  {r['profile_id']}  ({r['scenario']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
