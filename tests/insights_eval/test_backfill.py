from unittest.mock import MagicMock


def test_backfill_iterates_profiles_idempotent():
    from scripts.insights_backfill import backfill_profiles

    profiles = [{"profile_id": f"P{i}"} for i in range(3)]

    runner = MagicMock(return_value={"status": "ok"})
    fetch = MagicMock(return_value=profiles)

    result = backfill_profiles(
        fetch_profiles=fetch,
        run_for_profile=runner,
        subscription_id="S",
    )
    assert result["processed"] == 3
    assert runner.call_count == 3

    runner2 = MagicMock(return_value={"status": "skipped_already_done"})
    result2 = backfill_profiles(
        fetch_profiles=fetch,
        run_for_profile=runner2,
        subscription_id="S",
    )
    assert result2["processed"] == 3
