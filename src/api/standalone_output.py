"""Output format conversion for standalone extraction results.

Converts structured data (table, entities, summary) to csv, markdown, html, or
returns the original dict for json (passthrough).
"""
import csv
import io
from typing import Any, Dict, Union


# ---------------------------------------------------------------------------
# HTML helper
# ---------------------------------------------------------------------------

def _esc(val: Any) -> str:
    """Minimal HTML escaping for &, <, >."""
    s = str(val)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s


# ---------------------------------------------------------------------------
# Table converters
# ---------------------------------------------------------------------------

def _table_to_csv(data: Dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    for table in data.get("tables", []):
        writer.writerow(table.get("headers", []))
        for row in table.get("rows", []):
            writer.writerow(row)
    return buf.getvalue()


def _table_to_markdown(data: Dict[str, Any]) -> str:
    parts = []
    for table in data.get("tables", []):
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        parts.append(header_row)
        parts.append(separator)
        for row in rows:
            parts.append("| " + " | ".join(str(c) for c in row) + " |")
        parts.append("")
    return "\n".join(parts)


def _table_to_html(data: Dict[str, Any]) -> str:
    parts = []
    for table in data.get("tables", []):
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        caption = table.get("caption", "")
        parts.append("<table>")
        if caption:
            parts.append(f"  <caption>{_esc(caption)}</caption>")
        parts.append("  <thead>")
        parts.append("    <tr>" + "".join(f"<th>{_esc(h)}</th>" for h in headers) + "</tr>")
        parts.append("  </thead>")
        parts.append("  <tbody>")
        for row in rows:
            parts.append("    <tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>")
        parts.append("  </tbody>")
        parts.append("</table>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entities converters
# ---------------------------------------------------------------------------

def _entities_to_csv(data: Dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["text", "type", "page", "confidence"])
    for ent in data.get("entities", []):
        writer.writerow([
            ent.get("text", ""),
            ent.get("type", ""),
            ent.get("page", ""),
            ent.get("confidence", ""),
        ])
    return buf.getvalue()


def _entities_to_markdown(data: Dict[str, Any]) -> str:
    lines = ["## Entities", ""]
    for ent in data.get("entities", []):
        text = ent.get("text", "")
        etype = ent.get("type", "")
        page = ent.get("page", "")
        conf = ent.get("confidence", "")
        lines.append(f"- **{text}** ({etype}) — page {page}, confidence {conf}")
    return "\n".join(lines)


def _entities_to_html(data: Dict[str, Any]) -> str:
    parts = ["<dl>"]
    for ent in data.get("entities", []):
        text = _esc(ent.get("text", ""))
        etype = _esc(ent.get("type", ""))
        page = _esc(ent.get("page", ""))
        conf = _esc(ent.get("confidence", ""))
        parts.append(f"  <dt>{text}</dt>")
        parts.append(f"  <dd>Type: {etype}, Page: {page}, Confidence: {conf}</dd>")
    parts.append("</dl>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Summary converters
# ---------------------------------------------------------------------------

def _summary_to_csv(data: Dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["title", "summary", "key_points"])
    for section in data.get("sections", []):
        key_points = section.get("key_points", [])
        writer.writerow([
            section.get("title", ""),
            section.get("summary", ""),
            "; ".join(key_points),
        ])
    return buf.getvalue()


def _summary_to_markdown(data: Dict[str, Any]) -> str:
    parts = []
    for section in data.get("sections", []):
        title = section.get("title", "")
        summary = section.get("summary", "")
        key_points = section.get("key_points", [])
        parts.append(f"## {title}")
        parts.append("")
        parts.append(summary)
        parts.append("")
        parts.append("**Key Points:**")
        for point in key_points:
            parts.append(f"- {point}")
        parts.append("")
    return "\n".join(parts)


def _summary_to_html(data: Dict[str, Any]) -> str:
    parts = []
    for section in data.get("sections", []):
        title = _esc(section.get("title", ""))
        summary = _esc(section.get("summary", ""))
        key_points = section.get("key_points", [])
        parts.append("<section>")
        parts.append(f"  <h2>{title}</h2>")
        parts.append(f"  <p>{summary}</p>")
        parts.append("  <ul>")
        for point in key_points:
            parts.append(f"    <li>{_esc(point)}</li>")
        parts.append("  </ul>")
        parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_CONVERTERS = {
    ("table", "csv"):       _table_to_csv,
    ("table", "markdown"):  _table_to_markdown,
    ("table", "html"):      _table_to_html,
    ("entities", "csv"):    _entities_to_csv,
    ("entities", "markdown"): _entities_to_markdown,
    ("entities", "html"):   _entities_to_html,
    ("summary", "csv"):     _summary_to_csv,
    ("summary", "markdown"): _summary_to_markdown,
    ("summary", "html"):    _summary_to_html,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_output(
    data: Dict[str, Any],
    mode: str,
    output_format: str,
) -> Union[str, Dict[str, Any]]:
    """Convert structured extraction data to the requested output format.

    Args:
        data: Structured extraction result dict.
        mode: One of "table", "entities", or "summary".
        output_format: One of "json", "csv", "markdown", or "html".

    Returns:
        str for csv/markdown/html; original dict for json (passthrough).

    Raises:
        ValueError: If the (mode, output_format) combination is unsupported.
    """
    if output_format == "json":
        return data

    key = (mode, output_format)
    converter = _CONVERTERS.get(key)
    if converter is None:
        raise ValueError(
            f"Unsupported conversion: mode={mode!r}, output_format={output_format!r}. "
            f"Supported combinations: {sorted(_CONVERTERS.keys())}"
        )
    return converter(data)
