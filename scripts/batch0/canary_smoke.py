#!/usr/bin/env python
"""Batch-0 canary smoke: POST 10 canned queries to a live DocWain API,
assert each response is non-empty, grounded where applicable, and carries
DocWain persona on greeting/identity.

Usage:

    DOCWAIN_SMOKE_PROFILE=<profile_id> \\
    DOCWAIN_SMOKE_SUB=<subscription_id> \\
    python scripts/batch0/canary_smoke.py [--base-url http://localhost:8000] [--auth <token>]

Exits 0 if all 10 pass; exits 1 and prints a compact report otherwise.

Intended use:
- Before Batch-0 merge: run against port 8000 to document today's
  regression (expect failures — these are the "before" snapshot).
- After Batch-0 merge: run again on port 8000 (now serving Batch-0 code).
  Compare the outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

CANNED_QUERIES = [
    ("greet",     "Hello",                        False, True),
    ("identity",  "Who are you?",                 False, True),
    ("lookup",    "What is the invoice total on the most recent invoice?", True,  False),
    ("list",      "List all documents I have uploaded.",      True,  False),
    ("count",     "How many documents do I have uploaded?",   True,  False),
    ("extract",   "Extract line items from the latest invoice.", True, False),
    ("summarize", "Summarize the most recent document.",      True,  False),
    ("compare",   "Compare the two most recent documents.",   True,  False),
    ("analyze",   "What does the data suggest about spending trends?", True, False),
    ("timeline",  "Show the timeline of documents received.", True,  False),
]


@dataclass
class QueryResult:
    intent: str
    query: str
    ok: bool
    reason: str
    duration_s: float
    response_head: str
    grounded: Optional[bool]
    context_found: Optional[bool]
    http_status: int


def post_ask(base_url: str, body: dict, auth_token: Optional[str], timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/ask",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if auth_token:
        req.add_header("Authorization", f"Bearer {auth_token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
        return e.code, {"_error_body": err_body}


def run_one(base_url, auth_token, profile_id, sub_id, intent, query, needs_grounding, needs_persona) -> QueryResult:
    # The request model QuestionRequest requires:
    # - query (required)
    # - profile_id (required)
    # - subscription_id (required)
    # Optional: user_id, session_id, document_id, tool_hint, etc.
    body = {
        "query": query,
        "profile_id": profile_id,
        "subscription_id": sub_id,
    }
    t0 = time.monotonic()
    try:
        status, payload = post_ask(base_url, body, auth_token, timeout=180)
    except Exception as exc:
        return QueryResult(intent, query, False, f"exception: {exc}", time.monotonic() - t0, "", None, None, 0)
    duration = time.monotonic() - t0

    # The response model AskResponse wraps AnswerPayload in an "answer" field
    # with keys: response, sources, grounded, context_found, metadata
    answer_obj = payload.get("answer", {})
    resp_text = (answer_obj.get("response") or "")
    grounded = answer_obj.get("grounded")
    ctx = answer_obj.get("context_found")

    reasons = []
    ok = True
    if status != 200:
        ok = False
        reasons.append(f"http_status={status}")
    if not resp_text:
        ok = False
        reasons.append("empty_response")
    if "I'm having trouble" in resp_text:
        ok = False
        reasons.append("static_fallback_surfaced")
    if needs_grounding and ctx is False:
        ok = False
        reasons.append("no_context_found")
    if needs_persona and "DocWain" not in resp_text:
        ok = False
        reasons.append("persona_missing")
    return QueryResult(
        intent, query, ok, ",".join(reasons) or "pass", duration,
        resp_text[:160].replace("\n", " "), grounded, ctx, status,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("DOCWAIN_SMOKE_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--auth", default=os.environ.get("DOCWAIN_SMOKE_AUTH"))
    args = ap.parse_args()
    profile_id = os.environ.get("DOCWAIN_SMOKE_PROFILE")
    sub_id = os.environ.get("DOCWAIN_SMOKE_SUB")
    if not profile_id or not sub_id:
        print("error: DOCWAIN_SMOKE_PROFILE and DOCWAIN_SMOKE_SUB must be set", file=sys.stderr)
        return 2

    print(f"Canary base_url={args.base_url} profile={profile_id[:8]}... sub={sub_id[:8]}...")
    print(f"{'INTENT':<10s} {'OK':<3s} {'STATUS':<6s} {'T(s)':<6s} REASON")
    print("-" * 90)
    failures = 0
    for intent, query, needs_g, needs_p in CANNED_QUERIES:
        r = run_one(args.base_url, args.auth, profile_id, sub_id, intent, query, needs_g, needs_p)
        ok_s = "YES" if r.ok else "NO"
        print(f"{r.intent:<10s} {ok_s:<3s} {r.http_status:<6d} {r.duration_s:<6.2f} {r.reason}")
        print(f"           grounded={r.grounded} ctx={r.context_found}  head={r.response_head!r}")
        if not r.ok:
            failures += 1
    print("-" * 90)
    print(f"TOTAL: {10 - failures}/10 passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
