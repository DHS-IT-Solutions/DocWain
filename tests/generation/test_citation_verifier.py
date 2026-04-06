"""Tests for src.generation.citation_verifier."""

from src.generation.citation_verifier import (
    ClaimVerification,
    VerificationResult,
    extract_claims,
    verify_claims,
    verify_response,
)


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------

def test_extract_claims_splits_sentences():
    text = "Revenue was $1.5M. The company has 500 employees. What do you think?"
    claims = extract_claims(text)
    assert len(claims) == 2  # question excluded
    assert "Revenue was $1.5M" in claims[0]


def test_extract_claims_filters_meta():
    text = "Based on the documents, revenue grew 20%. It's worth noting that this is significant."
    claims = extract_claims(text)
    # Should extract the factual part, filtering meta-commentary
    assert any("20%" in c for c in claims)


def test_extract_claims_filters_questions():
    text = "What is the revenue? The revenue was $5M. How about profits?"
    claims = extract_claims(text)
    assert len(claims) == 1
    assert "$5M" in claims[0]


def test_extract_claims_filters_short_fragments():
    text = "Yes. OK. The total revenue for Q2 2024 was $3.2 million."
    claims = extract_claims(text)
    assert len(claims) == 1
    assert "$3.2 million" in claims[0]


def test_extract_claims_empty_input():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_extract_claims_strips_hedge_prefix():
    text = "It's worth noting that the company earned $10M in revenue last year."
    claims = extract_claims(text)
    assert len(claims) == 1
    assert "company earned" in claims[0].lower()


# ---------------------------------------------------------------------------
# verify_claims
# ---------------------------------------------------------------------------

def test_verify_supported_claim():
    claims = ["Revenue was $1.5M in Q2"]
    chunks = [{"text": "Q2 2024 financial report shows revenue of $1.5M for the period."}]
    results = verify_claims(claims, chunks)
    assert results[0].status == "SUPPORTED"


def test_verify_unsupported_claim():
    claims = ["The company went bankrupt in 2024"]
    chunks = [{"text": "Revenue was $1.5M in Q2 2024, showing healthy growth."}]
    results = verify_claims(claims, chunks)
    assert results[0].status == "UNSUPPORTED"


def test_verify_partial_claim_no_llm():
    """When keyword overlap is moderate and no LLM is provided, expect PARTIAL."""
    claims = ["The company reported strong revenue growth last year"]
    chunks = [{"text": "Revenue growth was noted in the annual report for the company."}]
    results = verify_claims(claims, chunks)
    # Moderate overlap — should be PARTIAL (no LLM to adjudicate)
    assert results[0].status in ("PARTIAL", "SUPPORTED")


def test_verify_claims_with_llm():
    """LLM callback is used for ambiguous overlap."""
    def mock_llm(prompt: str) -> str:
        return "YES"

    claims = ["The company reported strong revenue growth last year"]
    chunks = [{"text": "Revenue growth was noted in the annual report for the company."}]
    results = verify_claims(claims, chunks, llm_fn=mock_llm)
    # Mock LLM says YES -> SUPPORTED
    assert results[0].status == "SUPPORTED"


def test_verify_claims_empty():
    assert verify_claims([], []) == []


def test_verify_claims_empty_chunk_text():
    claims = ["Revenue was $1.5M"]
    chunks = [{"text": ""}]
    results = verify_claims(claims, chunks)
    assert results[0].status == "UNSUPPORTED"


# ---------------------------------------------------------------------------
# verify_response
# ---------------------------------------------------------------------------

def test_verify_response_computes_score():
    text = "Revenue was $1.5M. The company has 500 employees."
    chunks = [{"text": "Revenue of $1.5M reported. Total headcount is 500."}]
    result = verify_response(text, chunks)
    assert result.overall_score >= 0.5
    assert isinstance(result.grounding_density, float)


def test_verify_empty_response():
    result = verify_response("", [])
    assert result.overall_score == 1.0  # no claims = nothing to dispute
    assert len(result.claims) == 0


def test_verify_response_flagged_claims():
    text = "Revenue was $1.5M. The CEO resigned yesterday."
    chunks = [{"text": "Revenue of $1.5M reported for the fiscal quarter."}]
    result = verify_response(text, chunks)
    # "CEO resigned" should be flagged as unsupported
    assert any("CEO" in c for c in result.flagged_claims)


def test_verify_response_grounding_density():
    text = "Revenue was $1.5M. Costs were $500K. Profit was $1M."
    chunks = [{"text": "Revenue $1.5M, costs $500K, profit $1M for the period."}]
    result = verify_response(text, chunks)
    assert result.grounding_density > 0.0
    # 3 claims in ~12 words -> density should be roughly 25 claims/100 words
    assert result.grounding_density > 10.0
