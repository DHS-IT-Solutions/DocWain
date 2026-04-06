"""Native parsers for CSV/TSV and Excel files.

These bypass the OCR/deep-analysis pipeline entirely, producing structured
statistical profiles and representative samples instead of raw text.
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NATIVE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
_MAX_TEXT_CHARS = 50_000
_SAMPLING_THRESHOLD = 10_000
_SAMPLE_SIZE = 500  # total representative rows when sampling


def is_native_parseable(filename: str) -> bool:
    """Return True if the file extension indicates a natively parseable format."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in _NATIVE_EXTENSIONS


def parse_native(content: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Route to the appropriate native parser based on file extension.

    Returns None if the file is not natively parseable or parsing fails.
    """
    if not is_native_parseable(filename):
        return None
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext in (".csv", ".tsv"):
            return parse_csv(content, filename)
        elif ext in (".xlsx", ".xls"):
            return parse_excel(content, filename)
    except Exception:
        logger.exception("Native parser failed for %s, falling back to default pipeline", filename)
        return None
    return None


# ---------------------------------------------------------------------------
# CSV / TSV
# ---------------------------------------------------------------------------

def parse_csv(content: bytes, filename: str) -> Dict[str, Any]:
    """Parse a CSV/TSV file into a structured result dict."""
    sep = "\t" if filename.lower().endswith(".tsv") else ","
    df = pd.read_csv(io.BytesIO(content), sep=sep, low_memory=False)
    profile = _build_profile(df, filename)
    profile["parser"] = "native_csv"
    return profile


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def parse_excel(content: bytes, filename: str) -> Dict[str, Any]:
    """Parse an Excel workbook into a structured result dict."""
    sheets_dict = pd.read_excel(io.BytesIO(content), sheet_name=None)
    sheets: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for sheet_name, df in sheets_dict.items():
        sheet_profile = _build_profile(df, filename, sheet_name=str(sheet_name))
        sheets.append(sheet_profile)
        text_parts.append(sheet_profile.get("text", ""))

    combined_text = "\n\n".join(text_parts)
    if len(combined_text) > _MAX_TEXT_CHARS:
        combined_text = combined_text[:_MAX_TEXT_CHARS]

    return {
        "parser": "native_excel",
        "filename": filename,
        "sheet_count": len(sheets),
        "sheets": sheets,
        "text": combined_text,
        "full_text": combined_text,
        "texts": [combined_text],
        "native_parsed": True,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_profile(df: pd.DataFrame, filename: str, sheet_name: str | None = None) -> Dict[str, Any]:
    """Build a statistical profile + representative samples for a DataFrame."""
    columns = list(df.columns.astype(str))
    row_count = len(df)

    stat_profile = _statistical_profile(df)
    sample_rows = _representative_sample(df)
    text = _build_text_summary(df, filename, stat_profile, sample_rows, sheet_name=sheet_name)

    result: Dict[str, Any] = {
        "filename": filename,
        "row_count": row_count,
        "columns": columns,
        "statistical_profile": stat_profile,
        "sample_rows": sample_rows,
        "text": text,
        # Pipeline-compatible fields: downstream extraction_service expects
        # full_text and texts for sanitization, PII masking, and embedding.
        "full_text": text,
        "texts": [text],
        # Signal to skip in-extraction embedding — native-parsed data
        # will be properly chunked and embedded during the embedding stage.
        "native_parsed": True,
    }
    if sheet_name is not None:
        result["name"] = sheet_name
    return result


def _statistical_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute per-column statistics."""
    profile: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        col_str = str(col)

        # Try numeric first
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            profile[col_str] = {
                "dtype": "numeric",
                "min": _safe_scalar(clean.min()) if len(clean) else None,
                "max": _safe_scalar(clean.max()) if len(clean) else None,
                "mean": _safe_scalar(clean.mean()) if len(clean) else None,
                "std": _safe_scalar(clean.std()) if len(clean) else None,
                "median": _safe_scalar(clean.median()) if len(clean) else None,
                "null_count": int(series.isna().sum()),
            }
        elif pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            profile[col_str] = {
                "dtype": "datetime",
                "earliest": str(clean.min()) if len(clean) else None,
                "latest": str(clean.max()) if len(clean) else None,
                "null_count": int(series.isna().sum()),
            }
        else:
            # Treat as string/categorical
            clean = series.dropna().astype(str)
            value_counts = clean.value_counts()
            top_values = value_counts.head(10).to_dict()
            profile[col_str] = {
                "dtype": "string",
                "unique_count": int(clean.nunique()),
                "top_values": top_values,
                "null_count": int(series.isna().sum()),
            }
    return profile


def _representative_sample(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return representative rows: all if small, sampled otherwise."""
    row_count = len(df)
    if row_count == 0:
        return []

    if row_count <= _SAMPLING_THRESHOLD:
        return _df_to_records(df)

    # Stratified sample: head + tail + random middle + outlier rows
    n_head = min(50, row_count)
    n_tail = min(50, row_count)
    n_random = min(_SAMPLE_SIZE - n_head - n_tail - 50, row_count)  # leave room for outliers
    n_random = max(n_random, 0)

    head = df.head(n_head)
    tail = df.tail(n_tail)

    middle_indices = set(range(n_head, row_count - n_tail))
    if middle_indices and n_random > 0:
        rng = np.random.default_rng(42)
        random_idx = rng.choice(list(middle_indices), size=min(n_random, len(middle_indices)), replace=False)
        random_sample = df.iloc[sorted(random_idx)]
    else:
        random_sample = df.iloc[0:0]

    # Outlier rows: for each numeric column, grab rows at min/max
    outlier_indices = set()
    for col in df.select_dtypes(include=[np.number]).columns:
        clean = df[col].dropna()
        if len(clean) > 0:
            outlier_indices.add(clean.idxmin())
            outlier_indices.add(clean.idxmax())

    # Remove indices already covered
    existing_indices = set(head.index) | set(tail.index) | set(random_sample.index)
    outlier_indices -= existing_indices
    outlier_rows = df.loc[list(outlier_indices)[:50]] if outlier_indices else df.iloc[0:0]

    combined = pd.concat([head, random_sample, outlier_rows, tail]).drop_duplicates()
    return _df_to_records(combined)


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of dicts with JSON-safe values."""
    records = df.to_dict(orient="records")
    safe_records = []
    for rec in records:
        safe_rec = {}
        for k, v in rec.items():
            k_str = str(k)
            if pd.isna(v):
                safe_rec[k_str] = None
            elif isinstance(v, (np.integer,)):
                safe_rec[k_str] = int(v)
            elif isinstance(v, (np.floating,)):
                safe_rec[k_str] = float(v)
            elif isinstance(v, np.bool_):
                safe_rec[k_str] = bool(v)
            else:
                safe_rec[k_str] = v
            # Truncate very long string values
            if isinstance(safe_rec[k_str], str) and len(safe_rec[k_str]) > 500:
                safe_rec[k_str] = safe_rec[k_str][:500] + "..."
        safe_records.append(safe_rec)
    return safe_records


def _build_text_summary(
    df: pd.DataFrame,
    filename: str,
    stat_profile: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
    sheet_name: str | None = None,
) -> str:
    """Build a structured text summary suitable for embedding."""
    parts: List[str] = []
    title = f"Structured data: {filename}"
    if sheet_name:
        title += f" (sheet: {sheet_name})"
    parts.append(title)
    parts.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    parts.append(f"Column names: {', '.join(str(c) for c in df.columns)}")

    # Column profiles
    parts.append("\n--- Column Profiles ---")
    for col, stats in stat_profile.items():
        dtype = stats.get("dtype", "unknown")
        if dtype == "numeric":
            parts.append(
                f"  {col} (numeric): min={stats['min']}, max={stats['max']}, "
                f"mean={stats['mean']:.4f}, median={stats['median']}, std={stats['std']:.4f}"
                if stats['mean'] is not None
                else f"  {col} (numeric): all null"
            )
        elif dtype == "datetime":
            parts.append(f"  {col} (datetime): {stats['earliest']} to {stats['latest']}")
        else:
            top = list(stats.get("top_values", {}).keys())[:5]
            parts.append(f"  {col} (string): {stats['unique_count']} unique values, top: {top}")

    # Sample rows
    n_sample_display = min(20, len(sample_rows))
    if sample_rows:
        parts.append(f"\n--- Sample Rows ({n_sample_display} of {len(sample_rows)} representative) ---")
        for row in sample_rows[:n_sample_display]:
            parts.append(str(row))

    text = "\n".join(parts)
    if len(text) > _MAX_TEXT_CHARS:
        text = text[:_MAX_TEXT_CHARS]
    return text


def _safe_scalar(val: Any) -> Any:
    """Convert numpy scalars to Python native types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    return val
