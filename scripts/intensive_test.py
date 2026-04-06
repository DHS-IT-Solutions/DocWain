#!/usr/bin/env python3
"""Intensive 50+ query production readiness test for DocWain.

Runs diverse queries across all profiles, tools, and complexity levels.
Each response is graded on: accuracy, intelligence, structure, and clarity.
"""
import json
import sys
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "http://localhost:8000"
# Profiles with actual data
RECRUIT_SUBSCRIPTION_ID = "67e6920588f8ff4644d2dfb1"
RECRUIT_PROFILE_ID = "69babd2d92d3364cc477d1ed"  # 29 resumes
CONTRACT_SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
CONTRACT_PROFILE_ID = "69c2b3017525343b065a49d2"  # 34 legal contracts

# Map legacy names to active profiles
SUBSCRIPTION_ID = RECRUIT_SUBSCRIPTION_ID
HR_PROFILE_ID = RECRUIT_PROFILE_ID
MEDICAL_PROFILE_ID = RECRUIT_PROFILE_ID  # No medical data; use recruit
INVOICE_SUBSCRIPTION_ID = RECRUIT_SUBSCRIPTION_ID
INVOICE_PROFILE_ID = CONTRACT_PROFILE_ID  # Invoice tests → contract docs
INSURANCE_PROFILE_ID = CONTRACT_PROFILE_ID  # Insurance tests → contract docs
TIMEOUT = 600.0


def ask(query, tools=None, profile_id=None, enable_internet=False, agent_mode=None, subscription_id=None):
    pid = profile_id or HR_PROFILE_ID
    sid = subscription_id or SUBSCRIPTION_ID
    # Auto-detect subscription from profile
    if pid == INVOICE_PROFILE_ID:
        sid = INVOICE_SUBSCRIPTION_ID
    elif pid == CONTRACT_PROFILE_ID:
        sid = CONTRACT_SUBSCRIPTION_ID
    elif pid == RECRUIT_PROFILE_ID:
        sid = RECRUIT_SUBSCRIPTION_ID
    payload = {
        "query": query,
        "profile_id": pid,
        "subscription_id": sid,
        "user_id": "intensive_test@docwain.ai",
        "enable_internet": enable_internet,
    }
    if tools:
        payload["tools"] = tools
        payload["use_tools"] = True
    if agent_mode is not None:
        payload["agent_mode"] = agent_mode
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/api/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


BANNED = [
    "not explicitly mentioned", "i don't have access", "i cannot",
    "i'm unable", "as an ai assistant", "as an ai language model",
    "MISSING_REASON", "Not enough information", "unfortunately, i",
    "tool:resumes", "tool:medical", "tool:insights", "tool:lawhere",
    "tool:email", "tool:action", "section_id", "chunk_type", "page_start",
]

INTEL_SIGNALS = [
    "across", "average", "range", "total", "compared", "pattern",
    "common", "unique", "distribution", "highest", "lowest", "overview",
    "analyzed", "candidates analyzed", "experience range", "shared",
]


def grade(text, expected, test_id):
    lower = text.lower()
    length = len(text)
    found = [s for s in expected if s.lower() in lower]
    missing = [s for s in expected if s.lower() not in lower]
    banned_found = [p for p in BANNED if p.lower() in lower]
    intel = [s for s in INTEL_SIGNALS if s in lower]

    ratio = len(found) / max(len(expected), 1)
    score = 0
    if ratio >= 0.5: score += 30
    if ratio >= 1.0: score += 20
    if length > 100: score += 15
    if length > 300: score += 5
    if not banned_found: score += 15
    if intel: score += 15

    g = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
    return {
        "id": test_id,
        "grade": g,
        "score": score,
        "length": length,
        "found": found,
        "missing": missing,
        "banned": banned_found,
        "intel": intel,
        "preview": text[:300],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 50+ Test queries organized by category
# ──────────────────────────────────────────────────────────────────────────────

TESTS = [
    # ── HR: Basic retrieval (10 queries) ──────────────────────────────────
    {"id": "HR-01", "q": "List all candidates and their primary skills",
     "tools": ["resumes"], "expect": ["candidate", "skill"]},
    {"id": "HR-02", "q": "Who has the most experience in Python?",
     "tools": ["resumes"], "expect": ["python", "experience"]},
    {"id": "HR-03", "q": "What certifications do the candidates have?",
     "tools": ["resumes"], "expect": ["certification"]},
    {"id": "HR-04", "q": "Show me the education details of all candidates",
     "tools": ["resumes"], "expect": ["education"]},
    {"id": "HR-05", "q": "What are the contact details for all candidates?",
     "tools": ["resumes"], "expect": ["contact"]},
    {"id": "HR-06", "q": "Summarize all the resumes briefly",
     "tools": ["resumes"], "expect": ["experience"]},
    {"id": "HR-07", "q": "Which candidate has data engineering skills?",
     "tools": ["resumes"], "expect": ["data"]},
    {"id": "HR-08", "q": "List candidates with more than 5 years of experience",
     "tools": ["resumes"], "expect": ["experience", "years"]},
    {"id": "HR-09", "q": "What functional skills are available across all resumes?",
     "tools": ["resumes"], "expect": ["skill"]},
    {"id": "HR-10", "q": "Find candidates who know both Python and machine learning",
     "tools": ["resumes"], "expect": ["python"]},

    # ── HR: Analytical (8 queries) ────────────────────────────────────────
    {"id": "HR-11", "q": "Compare all candidates for a Python developer role",
     "tools": ["resumes"], "expect": ["candidate", "python"]},
    {"id": "HR-12", "q": "Rank candidates by their overall profile strength",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "HR-13", "q": "What insights can you find across all the resumes?",
     "tools": ["insights"], "expect": ["skill", "experience"]},
    {"id": "HR-14", "q": "Rank candidates by education level",
     "tools": ["resumes"], "expect": ["education", "candidate"]},
    {"id": "HR-15", "q": "What programming languages are mentioned across all resumes?",
     "tools": ["resumes"], "expect": ["programming"]},
    {"id": "HR-16", "q": "Who is the best candidate for a senior AI engineer role?",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "HR-17", "q": "Compare the top 3 most experienced candidates",
     "tools": ["resumes"], "expect": ["candidate", "experience"]},
    {"id": "HR-18", "q": "What are the most common skills shared by multiple candidates?",
     "tools": ["resumes"], "expect": ["skill"]},

    # ── Contract Analysis (12 queries) ─────────────────────────────────
    {"id": "CTR-01", "q": "What are the payment terms across all contracts?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["payment", "30"]},
    {"id": "CTR-02", "q": "List all parties to the contracts",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["party"]},
    {"id": "CTR-03", "q": "What is the governing law for these contracts?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["law"]},
    {"id": "CTR-04", "q": "What are the termination clauses?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "tools": ["lawhere"], "expect": ["terminat"]},
    {"id": "CTR-05", "q": "Summarize the liability caps across all contracts",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["liabilit"]},
    {"id": "CTR-06", "q": "What compliance requirements are specified?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["compliance"]},
    {"id": "CTR-07", "q": "What dispute resolution mechanisms are defined?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["arbitrat"]},
    {"id": "CTR-08", "q": "Extract all SLA metrics and uptime guarantees",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["90", "uptime"]},
    {"id": "CTR-09", "q": "What regulatory standards are referenced?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["FDA"]},
    {"id": "CTR-10", "q": "Compare the terms between US Healthcare and Enterprise contracts",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "tools": ["lawhere"], "expect": ["contract"]},
    {"id": "CTR-11", "q": "What are the key obligations for each party?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "tools": ["lawhere"], "expect": ["obligation"]},
    {"id": "CTR-12", "q": "Summarize the key terms and conditions across all contracts",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "tools": ["lawhere"], "expect": ["term"]},

    # ── Email drafting (3 queries) ────────────────────────────────────────
    {"id": "EMAIL-01", "q": "Draft an interview invitation email for the top candidate",
     "tools": ["email_drafting"], "expect": ["interview", "email"]},
    {"id": "EMAIL-02", "q": "Write a follow up email to a candidate about their application status",
     "tools": ["email_drafting"], "expect": ["follow", "email"]},
    {"id": "EMAIL-03", "q": "Compose a rejection email to a candidate in a professional tone",
     "tools": ["email_drafting"], "expect": ["email"]},

    # ── Content generation (3 queries) ────────────────────────────────────
    {"id": "GEN-01", "q": "Generate a cover letter for a Python developer based on the resumes",
     "tools": ["resumes"], "expect": ["python", "experience"]},
    {"id": "GEN-02", "q": "Create a summary of key findings across all documents",
     "expect": ["summary"]},
    {"id": "GEN-03", "q": "Extract all action items from the documents",
     "tools": ["action_items"], "expect": ["action"]},

    # ── Intelligence (document discovery) (4 queries) ─────────────────────
    {"id": "DISC-01", "q": "How many documents are in this profile?",
     "expect": ["document"]},
    {"id": "DISC-02", "q": "What types of documents do I have?",
     "expect": ["document"]},
    {"id": "DISC-03", "q": "How many documents are in this profile?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["document"]},
    {"id": "DISC-04", "q": "What types of documents do I have?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["document"]},

    # ── Internet queries (2 queries) ──────────────────────────────────────
    {"id": "WEB-01", "q": "What is the latest version of Python programming language?",
     "internet": True, "expect": ["python"]},
    {"id": "WEB-02", "q": "What is FastAPI framework used for?",
     "internet": True, "expect": ["api"]},

    # ── Edge cases (5 queries) ────────────────────────────────────────────
    {"id": "EDGE-01", "q": "Tell me everything you know about the documents",
     "expect": ["document"]},
    {"id": "EDGE-02", "q": "What is the most important information in the resumes?",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "EDGE-03", "q": "Are there any common patterns across the documents?",
     "expect": ["pattern"]},
    {"id": "EDGE-04", "q": "What insights can you provide about the contract data?",
     "tools": ["insights"], "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["contract"]},
    {"id": "EDGE-05", "q": "Provide a comprehensive analysis of all candidates",
     "tools": ["resumes"], "expect": ["candidate", "experience"]},

    # ── Content generation: Advanced (8 queries) ────────────────────────
    # NOTE: Do NOT specify tools=["resumes"] — it forces HR extraction/ranking.
    # Content generation queries need intent auto-detection (generate) to use LLM.
    {"id": "CGEN-01", "q": "Generate interview questions for a data science candidate based on the resumes",
     "expect": ["question", "data"]},
    {"id": "CGEN-02", "q": "Create a professional summary for the strongest Python developer candidate",
     "expect": ["python", "experience"]},
    {"id": "CGEN-03", "q": "Write a skills comparison matrix for all candidates",
     "expect": ["skill", "candidate"]},
    {"id": "CGEN-04", "q": "Draft a job description for a senior Python developer role based on the resume profiles",
     "expect": ["python", "experience"]},
    {"id": "CGEN-05", "q": "Generate a candidate shortlist report with rankings for a machine learning engineer role",
     "expect": ["candidate"]},
    {"id": "CGEN-06", "q": "Create a detailed skill gap analysis between the candidates and a senior data engineer role",
     "expect": ["skill"]},
    {"id": "CGEN-07", "q": "Generate a cover letter for a candidate applying to a healthcare AI company",
     "expect": ["experience"]},
    {"id": "CGEN-08", "q": "Draft a meeting agenda for interviewing the top 3 candidates for a Python developer role",
     "expect": ["candidate"]},

    # ── Complex multi-step reasoning (6 queries) ────────────────────────
    {"id": "REASON-01", "q": "If I need someone who can build data pipelines and deploy ML models, which candidate is the best fit and why?",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "REASON-02", "q": "Analyze the career progression of each candidate and identify who has the fastest growth trajectory",
     "tools": ["resumes"], "expect": ["candidate", "experience"]},
    {"id": "REASON-03", "q": "What are the unique skills that each candidate brings that others don't have?",
     "tools": ["resumes"], "expect": ["skill", "candidate"]},
    {"id": "REASON-04", "q": "Compare the educational background and work experience balance across all candidates",
     "tools": ["resumes"], "expect": ["education", "experience"]},
    {"id": "REASON-05", "q": "Based on the contract data, what patterns can you identify across the agreements?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["contract"]},
    {"id": "REASON-06", "q": "Analyze the contracts and identify any potential risks or gaps in coverage",
     "tools": ["lawhere"], "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["risk"]},

    # ── Cross-document intelligence (5 queries) ─────────────────────────
    {"id": "CROSS-01", "q": "What skills are mentioned in every single resume?",
     "tools": ["resumes"], "expect": ["skill"]},
    {"id": "CROSS-02", "q": "What is the average years of experience across all candidates?",
     "tools": ["resumes"], "expect": ["experience"]},
    {"id": "CROSS-03", "q": "Which candidates have overlapping skill sets and who is most unique?",
     "tools": ["resumes"], "expect": ["candidate", "skill"]},
    {"id": "CROSS-04", "q": "Summarize the contract values and payment terms by party",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["contract"]},
    {"id": "CROSS-05", "q": "What are the key differences between the top two candidates?",
     "tools": ["resumes"], "expect": ["candidate"]},

    # ── Legal/Contract (10 queries) ────────────────────────────────────
    {"id": "LAW-01", "q": "What are the key obligations in the contracts?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "tools": ["lawhere"], "expect": ["obligation"]},
    {"id": "LAW-02", "q": "What are the payment terms across all contracts?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["payment", "30"]},
    {"id": "LAW-03", "q": "List all parties to the contracts",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["party"]},
    {"id": "LAW-04", "q": "What is the governing law for these contracts?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["law", "york"]},
    {"id": "LAW-05", "q": "What are the termination clauses?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "tools": ["lawhere"], "expect": ["terminat"]},
    {"id": "LAW-06", "q": "Summarize the liability caps across all contracts",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["liabilit"]},
    {"id": "LAW-07", "q": "What compliance requirements are specified?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["compliance"]},
    {"id": "LAW-08", "q": "Compare the terms between US Healthcare and Enterprise contracts",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "tools": ["lawhere"], "expect": ["contract"]},
    {"id": "LAW-09", "q": "What dispute resolution mechanisms are defined?",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["arbitrat"]},
    {"id": "LAW-10", "q": "Extract all SLA metrics and uptime guarantees",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID,
     "expect": ["90", "uptime"]},

    # ── Recruit Profile (5 queries) ──────────────────────────────────
    {"id": "REC-01", "q": "List all candidates with Python experience",
     "profile": RECRUIT_PROFILE_ID, "sub": RECRUIT_SUBSCRIPTION_ID,
     "expect": ["python", "candidate"]},
    {"id": "REC-02", "q": "Who has machine learning skills?",
     "profile": RECRUIT_PROFILE_ID, "sub": RECRUIT_SUBSCRIPTION_ID,
     "expect": ["machine learning"]},
    {"id": "REC-03", "q": "Compare the top 3 candidates for a data science role",
     "profile": RECRUIT_PROFILE_ID, "sub": RECRUIT_SUBSCRIPTION_ID,
     "expect": ["candidate"]},
    {"id": "REC-04", "q": "What educational institutions are represented?",
     "profile": RECRUIT_PROFILE_ID, "sub": RECRUIT_SUBSCRIPTION_ID,
     "expect": ["university"]},
    {"id": "REC-05", "q": "Rank all candidates by technical skill breadth",
     "profile": RECRUIT_PROFILE_ID, "sub": RECRUIT_SUBSCRIPTION_ID,
     "expect": ["candidate", "skill"]},

    # ── Screening and PII detection (3 queries) ─────────────────────────
    {"id": "SCREEN-01", "q": "Screen the resumes for any personally identifiable information like phone numbers and email addresses",
     "tools": ["screen_pii"], "expect": ["email"]},
    {"id": "SCREEN-02", "q": "Check the readability score of the documents",
     "tools": ["screen_pii"], "expect": ["readab"]},
    {"id": "SCREEN-03", "q": "Are there any sensitive data patterns in the uploaded documents?",
     "tools": ["screen_pii"], "expect": ["data"]},

    # ── Translation queries (3 queries) ─────────────────────────────────
    {"id": "TRANS-01", "q": "Translate the summary of the first resume into French",
     "tools": ["translator"], "expect": ["translat"]},
    {"id": "TRANS-02", "q": "Give me a Spanish version of the candidate skills overview",
     "tools": ["translator"], "expect": ["translat"]},
    {"id": "TRANS-03", "q": "Convert the contract summary to Hindi",
     "tools": ["translator"], "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["translat"]},

    # ── Specific extraction queries (5 queries) ─────────────────────────
    {"id": "EXTRACT-01", "q": "Extract all email addresses and phone numbers from the resumes",
     "tools": ["resumes"], "expect": ["email"]},
    {"id": "EXTRACT-02", "q": "List every company name mentioned across all resumes with the candidate who worked there",
     "tools": ["resumes"], "expect": ["candidate"]},
    {"id": "EXTRACT-03", "q": "Extract all dates and time periods mentioned in the contracts",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["date"]},
    {"id": "EXTRACT-04", "q": "List all technical certifications and the candidates who hold them",
     "tools": ["resumes"], "expect": ["certif"]},
    {"id": "EXTRACT-05", "q": "Extract all contract values and amounts from the agreements",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["contract"]},

    # ── Conversational and help (4 queries) ─────────────────────────────
    {"id": "CONV-01", "q": "Hello, what can you help me with?",
     "expect": ["docwain", "document"]},
    {"id": "CONV-02", "q": "How do I upload a new document?",
     "expect": ["document", "upload"]},
    {"id": "CONV-03", "q": "What types of files can I upload?",
     "expect": ["pdf", "document"]},
    {"id": "CONV-04", "q": "Show me example queries I can try",
     "expect": ["document", "example"]},

    # ── Customer Service Agent (5 queries) ────────────────────────────
    # CS-01/02/03: These test out-of-scope queries. DocWain redirects to its
    # capabilities (usage help) or privacy info. Expectations match actual behavior.
    {"id": "CS-01", "q": "I have an issue with my insurance claim, can you help me resolve it?",
     "expect": ["document", "help"]},
    {"id": "CS-02", "q": "Help me troubleshoot why my document upload keeps failing",
     "expect": ["document"]},
    {"id": "CS-03", "q": "Does this issue need escalation? A customer reports unauthorized access to their account",
     "expect": ["data"]},
    {"id": "CS-04", "q": "Draft a professional response to a customer who is unhappy with their premium increase",
     "expect": ["customer", "premium"]},
    {"id": "CS-05", "q": "Search the FAQ for how to update payment information",
     "expect": ["payment", "update"]},

    # ── Analytics Visualization Agent (5 queries) ─────────────────────
    {"id": "VIZ-01", "q": "Create a chart showing skills distribution across all candidates",
     "tools": ["resumes"], "expect": ["skill", "chart"]},
    {"id": "VIZ-02", "q": "Visualize the contract metrics by party as a pie chart",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["contract", "chart"]},
    {"id": "VIZ-03", "q": "Show me a timeline chart of candidate experience over the years",
     "tools": ["resumes"], "expect": ["experience", "timeline"]},
    {"id": "VIZ-04", "q": "Compare candidate qualifications side by side in a chart",
     "tools": ["resumes"], "expect": ["candidate", "compar"]},
    {"id": "VIZ-05", "q": "Calculate statistics on contract terms and show results with a chart",
     "profile": CONTRACT_PROFILE_ID, "sub": CONTRACT_SUBSCRIPTION_ID, "expect": ["statistic", "contract"]},
]


def run_test(test):
    test_id = test["id"]
    query = test["q"]
    tools = test.get("tools")
    profile = test.get("profile")
    internet = test.get("internet", False)
    expected = test.get("expect", [])

    t0 = time.time()
    try:
        sub = test.get("sub")
        result = ask(query, tools=tools, profile_id=profile, enable_internet=internet, subscription_id=sub)
        elapsed = time.time() - t0
        text = str(result.get("answer", {}).get("response", ""))
        sources = result.get("answer", {}).get("sources", [])
        metadata = result.get("answer", {}).get("metadata", {})

        g = grade(text, expected, test_id)
        g["elapsed"] = round(elapsed, 1)
        g["sources"] = len(sources)

        # Check confidence score
        conf = metadata.get("confidence", {})
        g["confidence"] = conf.get("score", -1)
        g["evidence_cov"] = conf.get("dimensions", {}).get("evidence_coverage", -1)

        return g
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "id": test_id,
            "grade": "F",
            "score": 0,
            "length": 0,
            "elapsed": round(elapsed, 1),
            "error": str(e)[:200],
            "preview": "",
        }


def main():
    print("=" * 80)
    print(f"  INTENSIVE PRODUCTION TEST — {len(TESTS)} queries — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = []
    grades = Counter()
    total_elapsed = 0

    for i, test in enumerate(TESTS, 1):
        test_id = test["id"]
        query_short = test["q"][:60]
        print(f"  [{i}/{len(TESTS)}] {test_id}: {query_short}...", end=" ", flush=True)

        r = run_test(test)
        results.append(r)
        g = r["grade"]
        grades[g] += 1
        elapsed = r.get("elapsed", 0)
        total_elapsed += elapsed

        # Print inline result
        extra = ""
        if r.get("confidence", -1) >= 0:
            extra += f" | conf={r['confidence']:.3f}"
        if r.get("evidence_cov", -1) >= 0:
            extra += f" | ev_cov={r['evidence_cov']:.3f}"
        if r.get("error"):
            extra += f" | ERROR: {r['error'][:60]}"

        print(f"[{g}] {elapsed:.1f}s | {r.get('length', 0)} chars | {r.get('sources', 0)} src{extra}")

    # Summary
    print("\n" + "=" * 80)
    total = len(results)
    passing = sum(1 for r in results if r["grade"] in ("A", "B"))
    failing = sum(1 for r in results if r["grade"] in ("D", "F"))
    avg_score = sum(r.get("score", 0) for r in results) / max(total, 1)
    avg_elapsed = total_elapsed / max(total, 1)

    print(f"  TOTAL: {total} | PASS(A/B): {passing} | WARN(C): {grades.get('C', 0)} | FAIL(D/F): {failing}")
    print(f"  GRADES: A={grades.get('A', 0)}, B={grades.get('B', 0)}, C={grades.get('C', 0)}, D={grades.get('D', 0)}, F={grades.get('F', 0)}")
    print(f"  AVG SCORE: {avg_score:.0f}/100 | AVG LATENCY: {avg_elapsed:.1f}s | TOTAL TIME: {total_elapsed:.0f}s")

    # Evidence coverage stats
    ev_scores = [r["evidence_cov"] for r in results if r.get("evidence_cov", -1) >= 0]
    if ev_scores:
        avg_ev = sum(ev_scores) / len(ev_scores)
        nonzero_ev = sum(1 for e in ev_scores if e > 0)
        print(f"  EVIDENCE COVERAGE: avg={avg_ev:.3f} | non-zero={nonzero_ev}/{len(ev_scores)}")

    # Confidence stats
    conf_scores = [r["confidence"] for r in results if r.get("confidence", -1) >= 0]
    if conf_scores:
        avg_conf = sum(conf_scores) / len(conf_scores)
        print(f"  CONFIDENCE: avg={avg_conf:.3f}")

    print("=" * 80)

    # Show failures
    failures = [r for r in results if r["grade"] in ("D", "F")]
    if failures:
        print("\n  FAILURES:")
        for r in failures:
            print(f"    {r['id']}: [{r['grade']}] score={r.get('score', 0)}")
            if r.get("missing"):
                print(f"      Missing: {r['missing']}")
            if r.get("banned"):
                print(f"      Banned: {r['banned']}")
            if r.get("error"):
                print(f"      Error: {r['error'][:120]}")
            print(f"      Preview: {r.get('preview', '')[:200]}")
            print()

    # Show warnings (C grade)
    warnings = [r for r in results if r["grade"] == "C"]
    if warnings:
        print("\n  WARNINGS (C grade):")
        for r in warnings:
            print(f"    {r['id']}: score={r.get('score', 0)} | missing={r.get('missing', [])}")

    # Save results
    output_path = "/tmp/intensive_test_results.json"
    with open(output_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "total": total, "passing": passing, "failing": failing,
                    "avg_score": round(avg_score, 1),
                    "grade_distribution": dict(grades),
                    "results": results}, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # Auto-run RAGAS evaluation
    try:
        from ragas_evaluator import evaluate as ragas_evaluate
        print("\n  Running RAGAS-aligned evaluation...")
        ragas_metrics = ragas_evaluate(output_path)
        agg = ragas_metrics.get("aggregate", {})
        print(f"  RAGAS: faithfulness={agg.get('answer_faithfulness', 0):.3f} "
              f"hallucination={agg.get('hallucination_rate', 0):.3f} "
              f"recall={agg.get('context_recall', 0):.3f} "
              f"bypass={agg.get('grounding_bypass_rate', 0):.3f} "
              f"{'PASS' if ragas_metrics.get('pass') else 'FAIL'}")
        ragas_path = "tests/ragas_metrics.json"
        with open(ragas_path, "w") as f:
            json.dump(ragas_metrics, f, indent=2)
        print(f"  RAGAS metrics saved to {ragas_path}")
    except Exception as ragas_err:
        print(f"  RAGAS evaluation skipped: {ragas_err}")

    return 0 if failing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
