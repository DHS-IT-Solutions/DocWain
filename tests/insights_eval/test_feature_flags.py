from src.api.feature_flags import FeatureFlags, is_enabled, FLAG_NAMES


def test_all_25_flags_registered():
    expected = {
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
    }
    assert set(FLAG_NAMES) == expected


def test_all_flags_default_false():
    flags = FeatureFlags()
    for name in FLAG_NAMES:
        assert is_enabled(name, flags) is False, f"{name} should default to False"


def test_is_enabled_unknown_flag_raises():
    import pytest
    with pytest.raises(KeyError):
        is_enabled("BOGUS_FLAG", FeatureFlags())
