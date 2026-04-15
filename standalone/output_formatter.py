import csv
import io
import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def format_output(raw_llm_response: str, output_format: str) -> Any:
    parsed = _try_parse_json(raw_llm_response)

    if output_format == "json":
        if parsed is not None:
            return parsed
        return {"content": raw_llm_response}

    if output_format == "csv":
        return _to_csv(parsed, raw_llm_response)

    if output_format == "sections":
        if parsed is not None and "sections" in parsed:
            return parsed
        return {"sections": [{"title": "Document", "content": raw_llm_response}]}

    if output_format == "flatfile":
        return _to_flatfile(parsed, raw_llm_response)

    if output_format == "tables":
        if parsed is not None:
            if "tables" in parsed:
                return parsed["tables"]
            return parsed
        return {"content": raw_llm_response}

    return {"content": raw_llm_response}


def _try_parse_json(text: str) -> dict | list | None:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    match = _JSON_FENCE_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _to_csv(parsed: dict | None, raw: str) -> str:
    if parsed and "tables" in parsed:
        output = io.StringIO()
        writer = csv.writer(output)
        for table in parsed["tables"]:
            if "headers" in table:
                writer.writerow(table["headers"])
            if "rows" in table:
                for row in table["rows"]:
                    writer.writerow(row)
        return output.getvalue()
    return raw


def _to_flatfile(parsed: dict | None, raw: str) -> str:
    if parsed and isinstance(parsed, dict):
        lines = []
        for key, value in parsed.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            lines.append(f"{key}={value}")
        return "\n".join(lines)
    return raw
