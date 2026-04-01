#!/usr/bin/env python3
"""End-user testing script for DocWain intelligence and accuracy evaluation."""

import json
import time
import requests
from dataclasses import dataclass, field

BASE = "http://localhost:8000"

# Subscription IDs
HR_SUB_ID = "67e6920588f8ff4644d2dfb1"        # HR resumes + invoices
LEGAL_SUB_ID = "67fde0754e36c00b14cea7f5"      # Legal contracts
SUB_ID = HR_SUB_ID  # Default for most tests

# Profile mappings — verified against Qdrant embeddings 2026-03-26
RESUME_PROFILE_1 = "69babd2d92d3364cc477d1ed"  # "Recruit" — 20 resumes, 174 chunks
RESUME_PROFILE_2 = "69bb7f4a92d3364cc477d913"  # "Recruit-Pavan" — 27 resumes, 129 chunks
RESUME_PROFILE_3 = "69babd2d92d3364cc477d1ed"  # Same as PROFILE_1 (no third resume profile)
INVOICE_PROFILE = "69bd2c345f03e6f10885f9ed"   # "INV" — 10 invoice PDFs, 135 chunks
INVOICE_PROFILE_2 = "69bb944092d3364cc477df48"  # "Invoices" — 10 invoice PDFs, 74 chunks
CONTRACT_PROFILE = "69c2b3017525343b065a49d2"   # "Contract" — 33 legal docs, 441 chunks

@dataclass
class TestResult:
    category: str
    query: str
    profile: str
    response: str = ""
    sources: list = field(default_factory=list)
    grounded: bool = False
    context_found: bool = False
    latency_ms: float = 0
    error: str = ""
    score: int = 0  # 0-10 manual scoring
    issues: list = field(default_factory=list)


def ask_query(profile_id: str, query: str, sub_id: str = None, timeout: int = 120) -> dict:
    """Send a query via /api/profiles/{profile_id}/query"""
    url = f"{BASE}/api/profiles/{profile_id}/query"
    payload = {
        "subscription_id": sub_id or SUB_ID,
        "query": query,
        "top_k": 10,
    }
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return {"data": data, "latency_ms": latency, "error": None}
        else:
            return {"data": None, "latency_ms": latency, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
    except Exception as e:
        return {"data": None, "latency_ms": (time.time() - start) * 1000, "error": str(e)}


def ask_main(profile_id: str, query: str, sub_id: str = None, debug: bool = True, timeout: int = 120) -> dict:
    """Send a query via /api/ask"""
    url = f"{BASE}/api/ask"
    payload = {
        "query": query,
        "profile_id": profile_id,
        "subscription_id": sub_id or SUB_ID,
        "debug": debug,
        "new_session": True,
    }
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return {"data": data, "latency_ms": latency, "error": None}
        else:
            return {"data": None, "latency_ms": latency, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
    except Exception as e:
        return {"data": None, "latency_ms": (time.time() - start) * 1000, "error": str(e)}


def extract_answer(result: dict) -> tuple:
    """Extract answer text, sources, grounded flag from result."""
    if result.get("error"):
        return "", [], False, False
    data = result.get("data", {})
    if not data:
        return "", [], False, False

    # Try nested answer structure
    answer = data.get("answer", data)
    if isinstance(answer, dict):
        text = answer.get("response", answer.get("answer", ""))
        sources = answer.get("sources", [])
        grounded = answer.get("grounded", False)
        context_found = answer.get("context_found", False)
    else:
        text = str(answer)
        sources = []
        grounded = False
        context_found = False

    return text, sources, grounded, context_found


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    # --- FACTUAL (Resume) ---
    {
        "category": "factual",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "What is Srianish Rameshwaran's educational qualification?",
        "expect": "specific degree/university info",
    },
    {
        "category": "factual",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "What are Manoj Prabakaran's skills?",
        "expect": "list of technical/professional skills",
    },
    {
        "category": "factual",
        "profile": RESUME_PROFILE_2,
        "sub_id": HR_SUB_ID,
        "query": "What is Dhanush P's work experience?",
        "expect": "job titles, companies, durations",
    },

    # --- CONTACT/DETAIL ---
    {
        "category": "contact",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "What is Navin E's email and phone number?",
        "expect": "email address and phone number",
    },
    {
        "category": "contact",
        "profile": RESUME_PROFILE_2,
        "sub_id": HR_SUB_ID,
        "query": "Provide contact details for Harini S",
        "expect": "email, phone, address if available",
    },

    # --- COMPARISON ---
    {
        "category": "comparison",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "Compare Srianish Rameshwaran and Kavipriya's skills and experience",
        "expect": "side-by-side comparison, structured format",
    },
    {
        "category": "comparison",
        "profile": RESUME_PROFILE_2,
        "sub_id": HR_SUB_ID,
        "query": "Compare Dhanush P and Aruna T's qualifications",
        "expect": "structured comparison",
    },

    # --- RANKING ---
    {
        "category": "ranking",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "Rank all candidates by years of experience",
        "expect": "ordered list with experience years",
    },
    {
        "category": "ranking",
        "profile": RESUME_PROFILE_2,
        "sub_id": HR_SUB_ID,
        "query": "Who is the most qualified candidate for a data science role?",
        "expect": "ranked recommendation with reasoning",
    },

    # --- SUMMARY ---
    {
        "category": "summary",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "Give me a summary of all candidates in this profile",
        "expect": "brief overview of each candidate",
    },
    {
        "category": "summary",
        "profile": CONTRACT_PROFILE,
        "sub_id": LEGAL_SUB_ID,
        "query": "Summarize the key terms across all contracts",
        "expect": "summary of contract terms and conditions",
    },

    # --- INVOICE ---
    {
        "category": "factual_invoice",
        "profile": INVOICE_PROFILE,
        "sub_id": HR_SUB_ID,
        "query": "What is the total amount on the Fresh Foods Co invoice?",
        "expect": "specific monetary amount",
    },
    {
        "category": "factual_invoice",
        "profile": INVOICE_PROFILE,
        "sub_id": HR_SUB_ID,
        "query": "List all line items from the Stellar Retail invoices",
        "expect": "itemized list with quantities and prices",
    },
    {
        "category": "comparison_invoice",
        "profile": INVOICE_PROFILE,
        "sub_id": HR_SUB_ID,
        "query": "Compare the totals across all invoices",
        "expect": "comparison table/list of invoice totals",
    },

    # --- CROSS-DOCUMENT ---
    {
        "category": "cross_document",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "Which candidates have Python experience?",
        "expect": "list from multiple resumes",
    },
    {
        "category": "cross_document",
        "profile": RESUME_PROFILE_2,
        "sub_id": HR_SUB_ID,
        "query": "What technologies are common across all candidates?",
        "expect": "aggregated tech skills from multiple docs",
    },

    # --- ANALYTICS ---
    {
        "category": "analytics",
        "profile": RESUME_PROFILE_2,
        "sub_id": HR_SUB_ID,
        "query": "How many candidates have more than 3 years of experience?",
        "expect": "count with names",
    },

    # --- GENERATE ---
    {
        "category": "generate",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "Write a professional summary for Srianish Rameshwaran based on his resume",
        "expect": "well-written paragraph summarizing candidate",
    },

    # --- EDGE CASES ---
    {
        "category": "edge_no_info",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "What is the company's revenue?",
        "expect": "graceful not-found message, not hallucinated",
    },
    {
        "category": "edge_ambiguous",
        "profile": RESUME_PROFILE_2,
        "sub_id": HR_SUB_ID,
        "query": "Tell me about the candidate",
        "expect": "clarification or list all candidates",
    },
    {
        "category": "edge_typo",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "Wht is Srianish's educaton?",
        "expect": "should still understand and answer about education",
    },

    # --- CONVERSATIONAL ---
    {
        "category": "conversational",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "Hello",
        "expect": "greeting response, not a document search",
    },
    {
        "category": "conversational",
        "profile": RESUME_PROFILE_1,
        "sub_id": HR_SUB_ID,
        "query": "What can you do?",
        "expect": "capability description",
    },

    # --- LEGAL/CONTRACT (replaces medical — no medical data available) ---
    {
        "category": "legal",
        "profile": CONTRACT_PROFILE,
        "sub_id": LEGAL_SUB_ID,
        "query": "What are the payment terms in the US Healthcare contracts?",
        "expect": "specific payment terms from contract documents",
    },
    {
        "category": "legal",
        "profile": CONTRACT_PROFILE,
        "sub_id": LEGAL_SUB_ID,
        "query": "What liability clauses are specified across the contracts?",
        "expect": "liability terms or appropriate not-found",
    },
]


def evaluate_response(test: dict, answer: str, sources: list, grounded: bool, context_found: bool) -> tuple:
    """Auto-evaluate response quality. Returns (score, issues)."""
    issues = []
    score = 5  # Start at middle

    if not answer or len(answer.strip()) < 10:
        return 1, ["Empty or near-empty response"]

    # Check for hallucination indicators
    if "edge_no_info" in test["category"] and context_found:
        # Should NOT have found context for irrelevant queries
        if "revenue" not in answer.lower() and "not" not in answer.lower() and "couldn't" not in answer.lower():
            pass  # ok
        elif any(w in answer.lower() for w in ["revenue is", "company earns", "annual revenue"]):
            issues.append("HALLUCINATION: fabricated answer for irrelevant query")
            score -= 4

    # Check grounding
    if test["category"] not in ("conversational", "edge_no_info", "generate", "edge_ambiguous"):
        if not grounded:
            issues.append("Response not grounded in evidence")
            score -= 1

    # Check structure for comparison/ranking
    if "comparison" in test["category"]:
        if "|" not in answer and "**" not in answer:
            issues.append("Comparison lacks structured format (no table or bold headers)")
            score -= 1

    if "ranking" in test["category"]:
        if not any(c.isdigit() for c in answer[:200]):
            issues.append("Ranking lacks numbered ordering")
            score -= 1

    # Check length appropriateness
    if test["category"] in ("summary", "cross_document", "comparison"):
        if len(answer) < 100:
            issues.append(f"Response too short for {test['category']} ({len(answer)} chars)")
            score -= 1

    # Check for LLM preamble leaks
    preamble_patterns = ["based on my analysis", "after reviewing", "upon examination", "i have reviewed"]
    for p in preamble_patterns:
        if p in answer.lower()[:100]:
            issues.append(f"LLM preamble leak: '{p}'")
            score -= 1
            break

    # Check for metadata leaks
    metadata_patterns = ["chunk_type", "section_id", "embed_pipeline", "canonical_json"]
    for m in metadata_patterns:
        if m in answer.lower():
            issues.append(f"Metadata leak: '{m}'")
            score -= 2
            break

    # Check for repetition
    sentences = answer.split(". ")
    if len(sentences) > 3:
        seen = set()
        for s in sentences:
            key = s.strip().lower()[:80]
            if key in seen and len(key) > 30:
                issues.append("Contains repeated sentences")
                score -= 1
                break
            seen.add(key)

    # Bonus for good structure
    if any(marker in answer for marker in ["|", "**", "- ", "1.", "•"]):
        score += 1

    # Bonus for sources
    if sources and len(sources) > 0:
        score += 1

    # Cap score
    score = max(1, min(10, score))

    return score, issues


def run_tests():
    results = []

    print("=" * 80)
    print("DocWain End-User Intelligence & Accuracy Test")
    print("=" * 80)
    print(f"\nRunning {len(TEST_CASES)} test queries...\n")

    for i, test in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {test['category'].upper()}: {test['query'][:60]}...")

        # Use /api/ask for main pipeline testing
        result = ask_main(test["profile"], test["query"],
                          sub_id=test.get("sub_id"), debug=True, timeout=180)

        answer, sources, grounded, context_found = extract_answer(result)
        latency = result["latency_ms"]
        error = result.get("error", "")

        score, issues = evaluate_response(test, answer, sources, grounded, context_found)

        tr = TestResult(
            category=test["category"],
            query=test["query"],
            profile=test["profile"],
            response=answer[:500] if answer else "",
            sources=sources[:5] if sources else [],
            grounded=grounded,
            context_found=context_found,
            latency_ms=latency,
            error=error or "",
            score=score,
            issues=issues,
        )
        results.append(tr)

        status = "OK" if not error else "ERR"
        grnd = "G" if grounded else "U"
        print(f"  [{status}] Score: {score}/10 | {grnd} | {latency:.0f}ms | {len(answer)} chars")
        if issues:
            for iss in issues:
                print(f"  ⚠ {iss}")
        if error:
            print(f"  ERROR: {error[:200]}")
        # Show first 200 chars of answer
        if answer:
            preview = answer[:200].replace("\n", " ")
            print(f"  → {preview}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"scores": [], "issues": [], "errors": 0, "latencies": []}
        categories[cat]["scores"].append(r.score)
        categories[cat]["issues"].extend(r.issues)
        categories[cat]["latencies"].append(r.latency_ms)
        if r.error:
            categories[cat]["errors"] += 1

    total_score = 0
    total_count = 0
    all_issues = []

    for cat, data in sorted(categories.items()):
        avg_score = sum(data["scores"]) / len(data["scores"])
        avg_lat = sum(data["latencies"]) / len(data["latencies"])
        total_score += sum(data["scores"])
        total_count += len(data["scores"])
        all_issues.extend(data["issues"])

        print(f"\n  {cat:25s}  Avg Score: {avg_score:.1f}/10  Avg Latency: {avg_lat:.0f}ms  Errors: {data['errors']}")
        if data["issues"]:
            for iss in data["issues"]:
                print(f"    ⚠ {iss}")

    overall = total_score / total_count if total_count else 0
    print(f"\n  {'OVERALL':25s}  Avg Score: {overall:.1f}/10  Total Issues: {len(all_issues)}")

    # Save detailed results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(results),
        "overall_score": round(overall, 2),
        "total_issues": len(all_issues),
        "results": [
            {
                "category": r.category,
                "query": r.query,
                "response_preview": r.response[:300],
                "grounded": r.grounded,
                "context_found": r.context_found,
                "latency_ms": round(r.latency_ms),
                "score": r.score,
                "issues": r.issues,
                "error": r.error,
            }
            for r in results
        ],
        "all_issues": all_issues,
        "category_summary": {
            cat: {
                "avg_score": round(sum(d["scores"]) / len(d["scores"]), 2),
                "avg_latency_ms": round(sum(d["latencies"]) / len(d["latencies"])),
                "issues": d["issues"],
            }
            for cat, d in categories.items()
        },
    }

    with open("tests/enduser_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to tests/enduser_test_results.json")
    return results


if __name__ == "__main__":
    run_tests()
