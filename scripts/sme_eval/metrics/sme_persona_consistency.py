"""sme_persona_consistency metric.

Uses an LLM judge to rate (0–5) how well the response's voice matches a
reference persona for the profile's domain. Phase 0 uses hand-written
reference personas; Phase 4 will swap them for the actual adapter personas.

The judge_fn dependency is injected so tests don't call a real LLM.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult

# Phase 0 reference personas — will be replaced by adapter personas in Phase 4.
_REFERENCE_PERSONAS: dict[str, str] = {
    "finance": (
        "A senior financial analyst: direct, quantitative, hedged with uncertainty "
        "bounds. Cites absolute values behind every percentage. Distinguishes "
        "explicit facts from inferences."
    ),
    "legal": (
        "A senior legal counsel: precise, explicit about obligations and dates, "
        "careful to distinguish authoritative text from commentary. Flags ambiguity "
        "rather than smoothing over it."
    ),
    "hr": (
        "A seasoned HR business partner: professional, policy-anchored, respectful "
        "of privacy. Balances individual facts with organizational context."
    ),
    "medical": (
        "A careful clinical informationist: strictly evidence-grounded, explicit "
        "about differential possibilities, never prescribes. Caveats confidently."
    ),
    "it_support": (
        "A senior support engineer: structured symptom→cause→fix flow, precise "
        "about systems and conditions, explicit about assumptions."
    ),
    "generic": (
        "A domain-agnostic subject-matter expert: clear, evidence-grounded, "
        "explicitly hedged; distinguishes explicit facts from inferences."
    ),
}

_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a DocWain response matches a target persona.

Target persona for domain '{domain}':
{persona}

Response under evaluation:
{response}

Rate the persona match on a 0-5 scale:
- 0: Voice is entirely wrong for the persona (e.g., casual chatter where formal expertise is expected)
- 1: Major mismatches in tone, hedging, or specificity
- 2: Partial match with clear gaps
- 3: Acceptable match with minor issues
- 4: Good match; minor polish needed
- 5: Excellent match — reads as though written by the persona

Return ONLY a single number (0, 1, 2, 3, 4, or 5). No explanation.
"""


def default_judge_fn(prompt: str) -> float:
    """Default implementation — must be wired to DocWain LLM gateway.

    Left as a stub so the metric module imports cleanly even without a
    gateway available. The baseline CLI injects the real judge_fn when
    running; tests inject a mock.
    """
    raise NotImplementedError(
        "default_judge_fn is a stub; inject a real judge_fn from the baseline CLI"
    )


class SmePersonaConsistency(Metric):
    name = "sme_persona_consistency"

    def __init__(self, judge_fn: Callable[[str], float] = default_judge_fn):
        self._judge_fn = judge_fn

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        results = list(results)
        if not results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={"num_judged": 0, "num_failed": 0},
            )

        scores: list[float] = []
        failures: list[dict] = []
        for r in results:
            persona = _REFERENCE_PERSONAS.get(
                r.query.profile_domain, _REFERENCE_PERSONAS["generic"]
            )
            prompt = _JUDGE_PROMPT_TEMPLATE.format(
                domain=r.query.profile_domain,
                persona=persona,
                response=r.response_text[:2000],  # cap for judge input
            )
            try:
                score = float(self._judge_fn(prompt))
                scores.append(max(0.0, min(5.0, score)))
            except Exception as e:
                failures.append({"query_id": r.query.query_id, "error": str(e)})

        avg = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            metric_name=self.name,
            value=avg,
            details={
                "num_judged": len(scores),
                "num_failed": len(failures),
                "failures": failures,
                "scale_max": 5.0,
            },
        )
