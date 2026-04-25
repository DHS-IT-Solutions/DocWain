"""DocWain Celery application — task queue for document processing pipeline."""

import os
from celery import Celery

from src.docwain.logging_config import apply_log_hygiene
apply_log_hygiene()


def _build_redis_url(db: int = 0) -> str:
    """Build Redis URL from existing Config.Redis settings."""
    try:
        from src.api.config import Config
        host = Config.Redis.HOST
        port = Config.Redis.PORT
        ssl = Config.Redis.SSL
        pwd = getattr(Config.Redis, 'PASSWORD', '') or getattr(Config.Redis, 'KEY', '')
        scheme = 'rediss' if ssl else 'redis'
        if pwd:
            return f"{scheme}://:{pwd}@{host}:{port}/{db}"
        return f"{scheme}://{host}:{port}/{db}"
    except Exception:
        return f"redis://localhost:6379/{db}"


app = Celery("docwain")

app.config_from_object({
    "broker_url": os.getenv("CELERY_BROKER_URL", _build_redis_url(0)),
    "result_backend": os.getenv("CELERY_RESULT_BACKEND", _build_redis_url(1)),
    "broker_use_ssl": {"ssl_cert_reqs": __import__('ssl').CERT_NONE},
    "redis_backend_use_ssl": {"ssl_cert_reqs": __import__('ssl').CERT_NONE},

    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "timezone": "UTC",
    "enable_utc": True,

    # Queue definitions
    "task_queues": {
        "extraction_queue": {"exchange": "extraction", "routing_key": "extraction"},
        "screening_queue": {"exchange": "screening", "routing_key": "screening"},
        "kg_queue": {"exchange": "kg", "routing_key": "kg"},
        "embedding_queue": {"exchange": "embedding", "routing_key": "embedding"},
        "researcher_queue": {"exchange": "researcher", "routing_key": "researcher"},
        "backfill_queue": {"exchange": "backfill", "routing_key": "backfill"},
        "profile_intelligence_queue": {"exchange": "profile_intelligence", "routing_key": "profile_intelligence"},
        "researcher_v2_queue": {"exchange": "researcher_v2", "routing_key": "researcher_v2"},
        "researcher_refresh_queue": {"exchange": "researcher_refresh", "routing_key": "researcher_refresh"},
        "actions_queue": {"exchange": "actions", "routing_key": "actions"},
    },

    "task_routes": {
        "src.tasks.extraction.extract_document": {"queue": "extraction_queue"},
        "src.tasks.screening.screen_document": {"queue": "screening_queue"},
        "src.tasks.kg.build_knowledge_graph": {"queue": "kg_queue"},
        "src.tasks.kg_extract.run_knowledge_extraction": {"queue": "kg_queue"},
        "src.tasks.embedding.embed_document": {"queue": "embedding_queue"},
        "src.tasks.researcher.run_researcher_agent": {"queue": "researcher_queue"},
        "src.tasks.backfill.backfill_kg_refs": {"queue": "backfill_queue"},
        "src.tasks.profile_intelligence.generate_profile_intelligence_task": {"queue": "profile_intelligence_queue"},
        "src.tasks.profile_intelligence_refresh.profile_intelligence_weekly_refresh": {"queue": "profile_intelligence_queue"},
        # Insights Portal v2
        "src.tasks.researcher_v2.run_researcher_v2_for_doc_task": {"queue": "researcher_v2_queue"},
        "src.tasks.researcher_v2.run_researcher_v2_for_profile_task": {"queue": "researcher_v2_queue"},
        "src.tasks.researcher_v2_refresh.refresh_for_new_doc_task": {"queue": "researcher_refresh_queue"},
        "src.tasks.researcher_v2_refresh.refresh_scheduled_pass_task": {"queue": "researcher_refresh_queue"},
    },

    # Reliability
    "task_acks_late": True,
    "task_reject_on_worker_lost": True,
    "task_time_limit": 1800,
    "task_soft_time_limit": 1500,

    # Retry
    "task_default_retry_delay": 60,
    "task_max_retries": 3,

    # Monitoring
    "worker_send_task_events": True,
    "task_send_sent_event": True,
})

# Weekend Researcher refresh beat schedule (flag-gated).
def _build_researcher_beat_schedule():
    try:
        from src.api.config import Config
        from celery.schedules import crontab
        researcher_cfg = getattr(Config, "Researcher", None)
        if not researcher_cfg or not getattr(researcher_cfg, "WEEKEND_REFRESH_ENABLED", False):
            return {}
        cron_expr = getattr(researcher_cfg, "WEEKEND_REFRESH_CRON", "0 3 * * 0")
        parts = cron_expr.split()
        if len(parts) != 5:
            return {}
        minute, hour, day_of_month, month_of_year, day_of_week = parts
        return {
            "researcher-weekly-refresh": {
                "task": "src.tasks.researcher_refresh.researcher_weekly_refresh",
                "schedule": crontab(
                    minute=minute, hour=hour,
                    day_of_week=day_of_week,
                    day_of_month=day_of_month,
                    month_of_year=month_of_year,
                ),
                "options": {"queue": "researcher_queue"},
            }
        }
    except Exception:
        return {}


def _build_profile_intelligence_beat_schedule():
    """Weekly profile-intelligence refresh — flag-gated.

    Default cadence: Sunday 04:00 UTC (= 09:30 IST), well outside business hours.
    """
    try:
        from src.api.config import Config
        from celery.schedules import crontab
        cfg = getattr(Config, "ProfileIntelligence", None)
        if not cfg or not getattr(cfg, "WEEKLY_REFRESH_ENABLED", False):
            return {}
        cron_expr = getattr(cfg, "WEEKLY_REFRESH_CRON", "0 4 * * 0")
        parts = cron_expr.split()
        if len(parts) != 5:
            return {}
        minute, hour, day_of_month, month_of_year, day_of_week = parts
        return {
            "profile-intelligence-weekly-refresh": {
                "task": "src.tasks.profile_intelligence_refresh.profile_intelligence_weekly_refresh",
                "schedule": crontab(
                    minute=minute, hour=hour,
                    day_of_week=day_of_week,
                    day_of_month=day_of_month,
                    month_of_year=month_of_year,
                ),
                "options": {"queue": "profile_intelligence_queue"},
            }
        }
    except Exception:
        return {}


try:
    _beat = _build_researcher_beat_schedule()
    _beat.update(_build_profile_intelligence_beat_schedule())
    app.conf.beat_schedule = _beat
except Exception:
    pass

app.autodiscover_tasks(["src.tasks.extraction", "src.tasks.screening",
                         "src.tasks.kg", "src.tasks.kg_extract",
                         "src.tasks.embedding",
                         "src.tasks.backfill", "src.tasks.researcher",
                         "src.tasks.researcher_refresh",
                         "src.tasks.profile_intelligence",
                         "src.tasks.profile_intelligence_refresh"])
