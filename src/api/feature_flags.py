"""Feature flag registry for the Insights Portal.

Single source of truth for all 25 capability flags. Every flag defaults
to False; production enablement is staged per Section 14.4 of the spec.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


FLAG_NAMES = (
    "INSIGHTS_TYPE_ANOMALY_ENABLED",
    "INSIGHTS_TYPE_GAP_ENABLED",
    "INSIGHTS_TYPE_COMPARISON_ENABLED",
    "INSIGHTS_TYPE_SCENARIO_ENABLED",
    "INSIGHTS_TYPE_TREND_ENABLED",
    "INSIGHTS_TYPE_RECOMMENDATION_ENABLED",
    "INSIGHTS_TYPE_CONFLICT_ENABLED",
    "INSIGHTS_TYPE_PROJECTION_ENABLED",
    "ACTIONS_ARTIFACT_ENABLED",
    "ACTIONS_FORM_FILL_ENABLED",
    "ACTIONS_PLAN_ENABLED",
    "ACTIONS_REMINDER_ENABLED",
    "ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED",
    "KB_BUNDLED_ENABLED",
    "KB_EXTERNAL_ENABLED",
    "INSIGHTS_CITATION_ENFORCEMENT_ENABLED",
    "REFRESH_ON_UPLOAD_ENABLED",
    "INSIGHTS_PROACTIVE_INJECTION",
    "REFRESH_SCHEDULED_ENABLED",
    "REFRESH_INCREMENTAL_ENABLED",
    "WATCHLIST_ENABLED",
    "ADAPTER_AUTO_DETECT_ENABLED",
    "ADAPTER_BLOB_LOADING_ENABLED",
    "ADAPTER_GENERIC_FALLBACK_ENABLED",
    "VIZ_ENABLED",
    "INSIGHTS_DASHBOARD_ENABLED",
)


@dataclass(frozen=True)
class FeatureFlags:
    overrides: Dict[str, bool] = field(default_factory=dict)


def is_enabled(name: str, flags: FeatureFlags) -> bool:
    if name not in FLAG_NAMES:
        raise KeyError(name)
    if name in flags.overrides:
        return bool(flags.overrides[name])
    env_value = os.environ.get(name)
    if env_value is not None:
        return env_value.strip().lower() in ("1", "true", "yes", "on")
    return False
