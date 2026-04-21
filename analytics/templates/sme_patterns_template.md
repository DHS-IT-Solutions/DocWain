# DocWain SME Patterns — {{ report.period_start.strftime('%Y-%m') }}

Generated {{ generated_at }} from {{ report.num_query_runs }} query runs and {{ report.num_synth_runs }} synthesis runs in the window {{ report.period_start.strftime('%Y-%m-%d') }} → {{ report.period_end.strftime('%Y-%m-%d') }}.

## Executive summary

- Success clusters surfaced: **{{ report.successes | length }}**
- Failure clusters surfaced: **{{ report.failures | length }}**
- Artifact utility rows: **{{ report.artifact_utility | length }}**
- Persona performance rows: **{{ report.persona_effect | length }}**
- Training candidates for sub-project F: **{{ report.training_candidates | length }}**
{% if report.rollback_links %}
- Rollback post-mortems this month: **{{ report.rollback_links | length }}**
{%- endif %}

Phase 6 produces evidence for future retraining. No retraining is triggered automatically; sub-project F remains a separate, human-gated project.

## Success patterns

{% if report.successes %}
{% for c in report.successes %}
### {{ c.cluster_id }}

- **Size:** {{ c.size }}
- **Domain / intent:** {{ c.profile_domain or 'any' }} / {{ c.primary_intent or 'any' }}
- **Subscriptions:** {{ c.subscription_ids | join(', ') or 'n/a' }}
- **Signal score:** {{ '%.2f' | format(c.signal_score) }}
- **Summary:** {{ c.short_description }}
- **Evidence:**
{% for k, v in c.evidence.items() %}  - `{{ k }}`: {{ v }}
{% endfor %}

{% endfor %}
{% else %}
_No success clusters surfaced this month._
{% endif %}

## Failure patterns

{% if report.failures %}
{% for c in report.failures %}
### {{ c.cluster_id }}

- **Size:** {{ c.size }}
- **Domain / intent:** {{ c.profile_domain or 'any' }} / {{ c.primary_intent or 'any' }}
- **Subscriptions:** {{ c.subscription_ids | join(', ') or 'n/a' }}
- **Severity score:** {{ '%.2f' | format(c.signal_score) }}
- **Summary:** {{ c.short_description }}
- **Fingerprint samples:** {{ c.fingerprint_samples | join(', ') or 'n/a' }}
- **Evidence:**
{% for k, v in c.evidence.items() %}  - `{{ k }}`: {{ v }}
{% endfor %}

{% endfor %}
{% else %}
_No failure clusters surfaced this month._
{% endif %}

## Artifact utility

{% if report.artifact_utility %}
| Layer | Retrieval rate | Citation rate | Dead-weight? |
|---|---|---|---|
{% for c in report.artifact_utility %}| {{ c.evidence.layer }} | {{ '%.0f%%' | format(c.evidence.retrieval_rate * 100) }} | {{ '%.0f%%' | format(c.evidence.citation_rate * 100) }} | {{ 'yes' if c.evidence.dead_weight_flag else 'no' }} |
{% endfor %}
{% else %}
_No artifact utility rows._
{% endif %}

## Persona performance

{% if report.persona_effect %}
| Persona | Domain | Proxy score | Baseline | Regression? | Queries |
|---|---|---|---|---|---|
{% for c in report.persona_effect %}| {{ c.evidence.persona_role }} | {{ c.profile_domain }} | {{ '%.2f' | format(c.evidence.sme_score_proxy) }} | {{ '%.2f' | format(c.evidence.domain_baseline) }} | {{ 'yes' if c.evidence.regression_flag else 'no' }} | {{ c.evidence.queries }} |
{% endfor %}
{% else %}
_No persona rows._
{% endif %}

## Training candidates

Failure patterns that have stabilized across ≥2 months become candidates for sub-project F. Candidates listed here are evidence; sub-project F is triggered by human decision after review.

{% if report.training_candidates %}
{% for tc in report.training_candidates %}
### {{ tc.candidate_id }}

- **Months present:** {{ tc.months_present }}
- **Total volume:** {{ tc.total_volume }}
- **Stabilization score:** {{ '%.2f' | format(tc.stabilization_score) }}
- **Dominant intent / domain:** {{ tc.dominant_intent }} / {{ tc.dominant_domain }}
- **Cluster ids:** {{ tc.cluster_ids | join(', ') }}
- **Summary:** {{ tc.short_description }}

{% endfor %}
{% else %}
_No stabilized candidates this month._
{% endif %}

{% if report.rollback_links %}
## Rollback post-mortems

{% for link in report.rollback_links %}- [{{ link }}]({{ link }})
{% endfor %}
{% endif %}
