"""Baseline CLI — orchestrates end-to-end evaluation and writes snapshot.

Usage:
    python -m scripts.sme_eval.run_baseline \\
        --eval-dir tests/sme_evalset_v1/queries \\
        --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \\
        --out tests/sme_metrics_baseline_$(date +%Y-%m-%d).json \\
        --api-base-url http://localhost:8000 \\
        --skip-llm-judge           # optional, skips sme_persona_consistency
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import yaml

from scripts.sme_eval.aggregate import aggregate_latency_per_intent
from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.metrics.cross_doc_integration_rate import CrossDocIntegrationRate
from scripts.sme_eval.metrics.insight_novelty import InsightNovelty
from scripts.sme_eval.metrics.ragas_wrapper import RagasMetrics
from scripts.sme_eval.metrics.recommendation_groundedness import RecommendationGroundedness
from scripts.sme_eval.metrics.sme_artifact_hit_rate import SmeArtifactHitRate
from scripts.sme_eval.metrics.sme_persona_consistency import SmePersonaConsistency
from scripts.sme_eval.metrics.verified_removal_rate import VerifiedRemovalRate
from scripts.sme_eval.query_runner import QueryRunner, RunnerConfig
from scripts.sme_eval.result_store import JsonlResultStore
from scripts.sme_eval.schema import BaselineSnapshot, EvalQuery, EvalResult, MetricResult


DEFAULT_METRICS_NON_LLM: tuple[type[Metric], ...] = (
    RagasMetrics,
    RecommendationGroundedness,
    CrossDocIntegrationRate,
    InsightNovelty,
    VerifiedRemovalRate,
    SmeArtifactHitRate,
)

DEFAULT_METRICS = DEFAULT_METRICS_NON_LLM + (SmePersonaConsistency,)


def load_queries_from_yaml(path: Path) -> list[EvalQuery]:
    """Load and validate queries from one domain YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    queries_raw = raw.get("queries", [])
    return [EvalQuery(**q) for q in queries_raw]


def load_all_queries(eval_dir: Path) -> list[EvalQuery]:
    all_q: list[EvalQuery] = []
    for yaml_path in sorted(Path(eval_dir).glob("*.yaml")):
        all_q.extend(load_queries_from_yaml(yaml_path))
    return all_q


def _current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _docwain_llm_judge(prompt: str) -> float:
    """Judge via DocWain's own LLM gateway, keeping eval self-contained.

    Hits a chat completion endpoint compatible with OpenAI's /v1/chat/completions
    (per src/serving/vllm_manager.py). Extracts the first integer 0..5.
    """
    import re

    import httpx

    url = os.environ.get("DOCWAIN_LLM_URL", "http://localhost:8100/v1/chat/completions")
    model = os.environ.get("DOCWAIN_LLM_MODEL", "docwain-fast")
    resp = httpx.post(
        url,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 4,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    m = re.search(r"[0-5]", text)
    if not m:
        raise ValueError(f"Could not parse judge output: {text!r}")
    return float(m.group(0))


def compose_snapshot(
    results: Iterable[EvalResult],
    *,
    run_id: str,
    git_sha: str,
    api_base_url: str,
    judge_fn: Callable[[str], float] | None = None,
) -> BaselineSnapshot:
    results = list(results)
    counter = Counter(r.query.profile_domain for r in results)

    # Non-LLM metrics — always run
    ragas = RagasMetrics().compute(results)
    rec_ground = RecommendationGroundedness().compute(results)
    xdoc = CrossDocIntegrationRate().compute(results)
    novelty = InsightNovelty().compute(results)
    verif_rem = VerifiedRemovalRate().compute(results)
    art_hit = SmeArtifactHitRate().compute(results)

    sme_metrics: dict[str, MetricResult] = {
        rec_ground.metric_name: rec_ground,
        xdoc.metric_name: xdoc,
        novelty.metric_name: novelty,
        verif_rem.metric_name: verif_rem,
        art_hit.metric_name: art_hit,
    }

    # LLM-judge metric — opt-in
    if judge_fn is not None:
        persona = SmePersonaConsistency(judge_fn=judge_fn).compute(results)
        sme_metrics[persona.metric_name] = persona
    else:
        sme_metrics["sme_persona_consistency"] = MetricResult(
            metric_name="sme_persona_consistency",
            value=0.0,
            details={"skipped": True, "reason": "judge_fn not provided"},
        )

    # Latency per intent
    per_intent = aggregate_latency_per_intent(results)
    p50 = {intent: s["p50"] for intent, s in per_intent.items()}
    p95 = {intent: s["p95"] for intent, s in per_intent.items()}
    p99 = {intent: s["p99"] for intent, s in per_intent.items()}

    return BaselineSnapshot(
        run_id=run_id,
        captured_at=datetime.utcnow(),
        git_sha=git_sha,
        api_base_url=api_base_url,
        num_queries=len(results),
        per_domain_counts=dict(counter),
        ragas=ragas.details,
        sme_metrics=sme_metrics,
        latency_p50_per_intent=p50,
        latency_p95_per_intent=p95,
        latency_p99_per_intent=p99,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocWain SME eval baseline runner")
    parser.add_argument("--eval-dir", type=Path, default=Path("tests/sme_evalset_v1/queries"))
    parser.add_argument("--fixtures", type=Path,
                        default=Path("tests/sme_evalset_v1/fixtures/test_profiles.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results-jsonl", type=Path,
                        default=Path("tests/sme_results.jsonl"))
    parser.add_argument("--api-base-url", default=os.environ.get("DOCWAIN_API_URL",
                                                                 "http://localhost:8000"))
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit # of queries (debug)")
    args = parser.parse_args(argv)

    # Load queries
    queries = load_all_queries(args.eval_dir)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print(f"ERROR: no queries found under {args.eval_dir}", file=sys.stderr)
        return 2

    # Override subscription/profile IDs from fixtures (queries ship with placeholder IDs;
    # fixtures file supplies the real ones at run time)
    fixtures = yaml.safe_load(args.fixtures.read_text(encoding="utf-8"))
    sub_id = fixtures["test_subscription"]["subscription_id"]
    domain_to_profile = {
        dom: cfg["profile_id"] for dom, cfg in fixtures["profiles"].items()
    }
    queries = [
        q.model_copy(update={
            "subscription_id": sub_id,
            "profile_id": domain_to_profile.get(q.profile_domain, q.profile_id),
        })
        for q in queries
    ]
    # Rationale: EvalQuery is a pydantic BaseModel; in-place attribute assignment
    # would skip validation. model_copy(update=...) reruns validation.

    # Run
    run_id = f"baseline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    store = JsonlResultStore(args.results_jsonl)
    config = RunnerConfig(base_url=args.api_base_url)

    print(f"[run_baseline] running {len(queries)} queries; run_id={run_id}")
    with QueryRunner(config) as runner:
        for i, q in enumerate(queries, 1):
            result = runner.run_one(q, run_id=run_id)
            store.append(result)
            if i % 25 == 0 or i == len(queries):
                print(f"  … {i}/{len(queries)} done")

    # Compose snapshot
    judge_fn = None if args.skip_llm_judge else _docwain_llm_judge
    results = list(store.iter_run(run_id))
    snapshot = compose_snapshot(
        results,
        run_id=run_id,
        git_sha=_current_git_sha(),
        api_base_url=args.api_base_url,
        judge_fn=judge_fn,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(snapshot.model_dump_json(indent=2))
    print(f"[run_baseline] snapshot written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
