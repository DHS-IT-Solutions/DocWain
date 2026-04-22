"""Render PatternReport to Markdown via Jinja2.

The template lives in ``analytics/templates/`` so the format can be tuned
without touching Python. The renderer never mutates the report; it writes to
a target path and returns it.

Memory rule: use ``datetime.now(timezone.utc)`` — never ``datetime.utcnow()``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from scripts.sme_patterns.schema import PatternReport

_TEMPLATE_NAME = "sme_patterns_template.md"


def _default_templates_dir() -> Path:
    # scripts/sme_patterns/report/renderer.py  ->  repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "analytics" / "templates"


def render_pattern_report(
    report: PatternReport,
    out_path: Path | str,
    *,
    templates_dir: Path | str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render and return the written path as a string."""
    templates_dir = Path(templates_dir or _default_templates_dir())
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    template = env.get_template(_TEMPLATE_NAME)
    text = template.render(
        report=report,
        generated_at=(generated_at or datetime.now(timezone.utc)).isoformat(
            timespec="seconds"
        ),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return str(out)
