# Performance & Pipeline Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the stuck extraction pipeline, harden document processing with HITL gates, optimize query latency to sub-2s for simple queries, improve retrieval accuracy, and add analytical intelligence.

**Architecture:** Phased delivery across 3 parallel tracks (latency, accuracy, ingestion) plus a pipeline redesign with two HITL gates and pre-computed knowledge. All heavy analysis happens at document processing time; query time is lightweight assembly + reasoning.

**Tech Stack:** Python 3.12, FastAPI, MongoDB, Qdrant, Neo4j, Redis, vLLM, sentence-transformers (bge-m3), pandas, ThreadPoolExecutor

**Spec:** `docs/superpowers/specs/2026-04-06-performance-pipeline-redesign.md`

---

## Phase 0: Extraction Hardening (Immediate — Unblock Stuck Pipeline)

### Task 1: Add Per-Document Extraction Timeout

**Files:**
- Modify: `src/api/extraction_service.py:1830-1924` (the batch loop in `extract_documents()`)
- Test: `tests/api/test_extraction_timeout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_extraction_timeout.py
import threading
import time
import pytest
from unittest.mock import patch, MagicMock


def _slow_extract(doc_id, doc_data, conn_data):
    """Simulate a stuck extraction that takes forever."""
    time.sleep(600)
    return {"document_id": doc_id, "status": "EXTRACTION_COMPLETED"}


def test_per_document_timeout_fires():
    """A document that exceeds DOC_EXTRACTION_TIMEOUT_SECONDS should be marked FAILED."""
    from src.api.extraction_service import _extract_single_with_timeout

    result = _extract_single_with_timeout(
        doc_id="test_doc_123",
        doc_data={"name": "slow.pdf"},
        conn_data={},
        timeout_seconds=2,
        extract_fn=_slow_extract,
    )
    assert result["status"] == "EXTRACTION_FAILED"
    assert "timeout" in result.get("error", "").lower()


def test_normal_extraction_completes_within_timeout():
    """A fast extraction should return normally."""
    def _fast_extract(doc_id, doc_data, conn_data):
        return {"document_id": doc_id, "status": "EXTRACTION_COMPLETED"}

    from src.api.extraction_service import _extract_single_with_timeout

    result = _extract_single_with_timeout(
        doc_id="test_doc_456",
        doc_data={"name": "fast.pdf"},
        conn_data={},
        timeout_seconds=30,
        extract_fn=_fast_extract,
    )
    assert result["status"] == "EXTRACTION_COMPLETED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_extraction_timeout.py -v`
Expected: FAIL with `ImportError` — `_extract_single_with_timeout` does not exist yet.

- [ ] **Step 3: Implement `_extract_single_with_timeout`**

Add this function in `src/api/extraction_service.py` just above `extract_documents()` (before line 1757):

```python
# Per-document extraction timeout
_DOC_EXTRACTION_TIMEOUT_SECONDS = int(os.getenv("DOC_EXTRACTION_TIMEOUT_SECONDS", "300"))  # 5 min default


def _extract_single_with_timeout(
    doc_id: str,
    doc_data: Dict[str, Any],
    conn_data: Dict[str, Any],
    timeout_seconds: int = _DOC_EXTRACTION_TIMEOUT_SECONDS,
    extract_fn=None,
) -> Dict[str, Any]:
    """Run extraction for a single document with a hard timeout.

    If the extraction exceeds timeout_seconds, returns a FAILED result
    instead of blocking the batch forever.
    """
    if extract_fn is None:
        extract_fn = _extract_from_connector

    result_holder: List[Dict[str, Any]] = []
    error_holder: List[Exception] = []

    def _run():
        try:
            result_holder.append(extract_fn(doc_id, doc_data, conn_data))
        except Exception as exc:
            error_holder.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logger.error(
            "Document %s extraction timed out after %ds — marking FAILED",
            doc_id, timeout_seconds,
        )
        return {
            "document_id": doc_id,
            "status": STATUS_EXTRACTION_FAILED,
            "error": f"Extraction timed out after {timeout_seconds}s",
        }

    if error_holder:
        exc = error_holder[0]
        if isinstance(exc, CredentialError):
            raise exc  # Propagate credential errors to batch level
        return {
            "document_id": doc_id,
            "status": STATUS_EXTRACTION_FAILED,
            "error": str(exc),
        }

    return result_holder[0] if result_holder else {
        "document_id": doc_id,
        "status": STATUS_EXTRACTION_FAILED,
        "error": "Extraction produced no result",
    }
```

- [ ] **Step 4: Wire timeout into batch loop**

In `extract_documents()`, replace line 1876:
```python
# OLD:
res = _extract_from_connector(doc_id, doc_info.get("dataDict", {}), doc_info.get("connDict", {}))

# NEW:
res = _extract_single_with_timeout(
    doc_id,
    doc_info.get("dataDict", {}),
    doc_info.get("connDict", {}),
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_extraction_timeout.py -v`
Expected: PASS (both tests)

- [ ] **Step 6: Commit**

```bash
git add tests/api/test_extraction_timeout.py src/api/extraction_service.py
git commit -m "fix: add per-document extraction timeout (5min default)"
```

---

### Task 2: Add Stale Batch Lock Auto-Release

**Files:**
- Modify: `src/api/extraction_service.py:1678-1714` (lock functions)
- Modify: `src/api/extraction_service.py:1757-1773` (lock check in `extract_documents()`)
- Test: `tests/api/test_extraction_timeout.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_extraction_timeout.py`:

```python
def test_stale_lock_auto_released():
    """If a batch lock is older than STALE_LOCK_THRESHOLD, acquire_batch_lock should reclaim it."""
    from src.api.extraction_service import _acquire_batch_lock, _release_batch_lock, _BATCH_LOCK_TTL_SECONDS

    # Acquire a lock
    lock1 = _acquire_batch_lock("test_sub_stale")
    assert lock1 is not None

    # Second acquire should fail (lock held)
    lock2 = _acquire_batch_lock("test_sub_stale")
    assert lock2 is None

    # Clean up
    _release_batch_lock(lock1)


def test_stale_lock_recovery_with_force():
    """Force-release should allow re-acquisition."""
    from src.api.extraction_service import _acquire_batch_lock, _force_release_batch_lock

    lock1 = _acquire_batch_lock("test_sub_force")
    assert lock1 is not None

    # Force release
    _force_release_batch_lock("test_sub_force")

    # Now acquire should succeed
    lock2 = _acquire_batch_lock("test_sub_force")
    assert lock2 is not None

    # Clean up
    from src.api.extraction_service import _release_batch_lock
    _release_batch_lock(lock2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_extraction_timeout.py::test_stale_lock_recovery_with_force -v`
Expected: FAIL — `_force_release_batch_lock` does not exist.

- [ ] **Step 3: Implement `_force_release_batch_lock`**

Add after `_release_batch_lock` in `src/api/extraction_service.py`:

```python
_STALE_LOCK_THRESHOLD_SECONDS = int(os.getenv("STALE_LOCK_THRESHOLD_SECONDS", "900"))  # 15 min


def _force_release_batch_lock(subscription_id: str) -> bool:
    """Force-release a batch lock for a subscription. Used for stale lock recovery."""
    lock_key = f"docwain:batch_extraction:{subscription_id}"
    try:
        from src.api.dw_newron import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            redis_client.delete(lock_key)
            logger.warning("Force-released stale batch lock for subscription %s", subscription_id)
            return True
    except Exception:
        pass
    from src.utils.idempotency import _MEMORY_LOCKS, _MEMORY_LOCK
    with _MEMORY_LOCK:
        _MEMORY_LOCKS.pop(lock_key, None)
    logger.warning("Force-released stale batch lock (in-memory) for subscription %s", subscription_id)
    return True
```

- [ ] **Step 4: Update `extract_documents()` to report lock age**

In `extract_documents()`, update the "already_running" return block (lines 1768-1773) to include lock age and force-release instructions:

```python
if not batch_lock_key:
    # Check lock age for stale lock detection
    lock_age = _get_batch_lock_age(effective_sub)
    stale = lock_age is not None and lock_age > _STALE_LOCK_THRESHOLD_SECONDS
    if stale:
        logger.warning(
            "Batch lock for subscription %s is stale (age=%ds > threshold=%ds). Auto-releasing.",
            effective_sub, lock_age, _STALE_LOCK_THRESHOLD_SECONDS,
        )
        _force_release_batch_lock(effective_sub)
        batch_lock_key = _acquire_batch_lock(effective_sub)
    if not batch_lock_key:
        logger.info(
            "Batch extraction already in progress for subscription %s; rejecting duplicate request",
            effective_sub,
        )
        return {
            "status": "already_running",
            "message": f"Extraction is already running for subscription {effective_sub}. "
                       "Please wait for the current batch to complete.",
            "lock_age_seconds": lock_age,
            "documents": [],
        }
```

Add the helper `_get_batch_lock_age` near the lock functions:

```python
def _get_batch_lock_age(subscription_id: str) -> Optional[float]:
    """Get how long the batch lock has been held (seconds). Returns None if no lock."""
    lock_key = f"docwain:batch_extraction:{subscription_id}"
    try:
        from src.api.dw_newron import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            ttl = redis_client.ttl(lock_key)
            if ttl and ttl > 0:
                return _BATCH_LOCK_TTL_SECONDS - ttl
            return None
    except Exception:
        pass
    # In-memory fallback
    from src.utils.idempotency import _MEMORY_LOCKS, _MEMORY_LOCK
    with _MEMORY_LOCK:
        expiry = _MEMORY_LOCKS.get(lock_key)
        if expiry:
            return _BATCH_LOCK_TTL_SECONDS - (expiry - time.time())
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_extraction_timeout.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add tests/api/test_extraction_timeout.py src/api/extraction_service.py
git commit -m "fix: auto-release stale batch extraction locks after 15min"
```

---

### Task 3: Add CSV/Excel Native Parser Path

**Files:**
- Create: `src/extraction/native_parsers.py`
- Modify: `src/api/dataHandler.py:469` (in `fileProcessor()` — add early return for CSV/Excel)
- Test: `tests/extraction/test_native_parsers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/extraction/test_native_parsers.py
import io
import pytest


def test_csv_native_parse_returns_structured_result():
    """CSV files should be parsed with pandas, not OCR."""
    from src.extraction.native_parsers import parse_csv

    csv_content = b"name,age,salary\nAlice,30,80000\nBob,25,75000\nCharlie,35,90000\n"
    result = parse_csv(csv_content, filename="test.csv")

    assert result["parser"] == "native_csv"
    assert result["row_count"] == 3
    assert result["columns"] == ["name", "age", "salary"]
    assert "statistical_profile" in result
    assert result["statistical_profile"]["age"]["min"] == 25
    assert result["statistical_profile"]["age"]["max"] == 35
    assert "sample_rows" in result
    assert len(result["sample_rows"]) == 3  # all rows (< 10K threshold)


def test_csv_large_file_uses_sampling():
    """CSVs with >10K rows should sample instead of including all rows."""
    from src.extraction.native_parsers import parse_csv

    # Generate a 15K row CSV
    header = b"id,value\n"
    rows = b"".join(f"{i},{i*1.5}\n".encode() for i in range(15000))
    csv_content = header + rows

    result = parse_csv(csv_content, filename="big.csv")

    assert result["row_count"] == 15000
    # Sample should be much smaller than full dataset
    assert len(result["sample_rows"]) < 1000
    assert "statistical_profile" in result
    assert result["statistical_profile"]["value"]["mean"] is not None


def test_excel_native_parse():
    """Excel files should be parsed with openpyxl, not OCR."""
    from src.extraction.native_parsers import parse_excel
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Product", "Price", "Quantity"])
    ws.append(["Widget", 9.99, 100])
    ws.append(["Gadget", 19.99, 50])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    result = parse_excel(buf.read(), filename="test.xlsx")

    assert result["parser"] == "native_excel"
    assert result["sheet_count"] == 1
    assert result["sheets"][0]["name"] == "Sheet1"
    assert result["sheets"][0]["row_count"] == 2
    assert result["sheets"][0]["columns"] == ["Product", "Price", "Quantity"]


def test_csv_text_representation_not_raw():
    """The text field should be a structured summary, not raw CSV content."""
    from src.extraction.native_parsers import parse_csv

    csv_content = b"name,age\nAlice,30\nBob,25\n"
    result = parse_csv(csv_content, filename="test.csv")

    # Text should NOT be the raw CSV (which would be megabytes for large files)
    assert result["text"] != "name,age\nAlice,30\nBob,25\n"
    # Text should contain structured description
    assert "columns" in result["text"].lower() or "name" in result["text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/extraction/test_native_parsers.py -v`
Expected: FAIL — `src.extraction.native_parsers` does not exist.

- [ ] **Step 3: Implement native parsers**

```python
# src/extraction/native_parsers.py
"""Native parsers for structured file formats (CSV, Excel).

These bypass OCR/vision pipelines entirely and use pandas/openpyxl
for fast, accurate extraction of structured data.
"""
import io
import math
from typing import Any, Dict, List, Optional

import pandas as pd

_SAMPLE_THRESHOLD = 10_000  # rows above which we sample instead of including all
_SAMPLE_SIZE = 500  # representative rows to keep
_MAX_TEXT_CHARS = 50_000  # max chars for the text representation


def _statistical_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Build per-column statistical profile."""
    profile = {}
    for col in df.columns:
        col_profile: Dict[str, Any] = {"dtype": str(df[col].dtype), "null_count": int(df[col].isna().sum())}
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            col_profile.update({
                "min": None if math.isnan(desc.get("min", float("nan"))) else float(desc["min"]),
                "max": None if math.isnan(desc.get("max", float("nan"))) else float(desc["max"]),
                "mean": None if math.isnan(desc.get("mean", float("nan"))) else round(float(desc["mean"]), 4),
                "std": None if math.isnan(desc.get("std", float("nan"))) else round(float(desc["std"]), 4),
                "median": None if math.isnan(desc.get("50%", float("nan"))) else float(desc["50%"]),
            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                col_profile.update({
                    "earliest": str(non_null.min()),
                    "latest": str(non_null.max()),
                })
        else:
            col_profile["unique_count"] = int(df[col].nunique())
            top = df[col].value_counts().head(5)
            col_profile["top_values"] = {str(k): int(v) for k, v in top.items()}
        profile[col] = col_profile
    return profile


def _sample_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Sample representative rows: head + tail + random + outliers."""
    if len(df) <= _SAMPLE_THRESHOLD:
        return df.head(500).to_dict(orient="records")

    parts = []
    parts.append(df.head(50))  # first 50
    parts.append(df.tail(50))  # last 50

    # Random sample from middle
    middle = df.iloc[50:-50]
    if len(middle) > 0:
        n_random = min(300, len(middle))
        parts.append(middle.sample(n=n_random, random_state=42))

    # Outliers from numeric columns
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        outliers = df[(df[col] < q1) | (df[col] > q99)]
        if len(outliers) > 0:
            parts.append(outliers.head(50))

    combined = pd.concat(parts).drop_duplicates().head(_SAMPLE_SIZE)
    return combined.to_dict(orient="records")


def _build_text_representation(
    filename: str,
    columns: List[str],
    row_count: int,
    profile: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
) -> str:
    """Build a structured text summary for embedding (NOT raw CSV content)."""
    lines = [
        f"Structured data file: {filename}",
        f"Rows: {row_count}, Columns: {len(columns)}",
        f"Column names: {', '.join(columns)}",
        "",
        "Column profiles:",
    ]
    for col, stats in profile.items():
        parts = [f"  {col} ({stats['dtype']})"]
        if "min" in stats and stats["min"] is not None:
            parts.append(f"range=[{stats['min']}, {stats['max']}], mean={stats['mean']}")
        if "unique_count" in stats:
            parts.append(f"unique={stats['unique_count']}")
        if "top_values" in stats:
            top = ", ".join(f"{k}({v})" for k, v in list(stats["top_values"].items())[:3])
            parts.append(f"top: {top}")
        lines.append(" — ".join(parts))

    lines.append("")
    lines.append(f"Sample data ({min(len(sample_rows), 20)} of {row_count} rows):")
    for row in sample_rows[:20]:
        row_str = ", ".join(f"{k}={v}" for k, v in row.items())
        lines.append(f"  {row_str}")

    text = "\n".join(lines)
    return text[:_MAX_TEXT_CHARS]


def parse_csv(content: bytes, filename: str) -> Dict[str, Any]:
    """Parse CSV using pandas. Returns structured result with profile and samples."""
    df = pd.read_csv(io.BytesIO(content), low_memory=False)

    # Attempt date parsing on likely date columns
    for col in df.columns:
        if any(hint in col.lower() for hint in ("date", "time", "timestamp", "created", "updated")):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass

    columns = list(df.columns)
    row_count = len(df)
    profile = _statistical_profile(df)
    samples = _sample_rows(df)
    text = _build_text_representation(filename, columns, row_count, profile, samples)

    return {
        "parser": "native_csv",
        "filename": filename,
        "row_count": row_count,
        "columns": columns,
        "statistical_profile": profile,
        "sample_rows": samples,
        "text": text,
    }


def parse_excel(content: bytes, filename: str) -> Dict[str, Any]:
    """Parse Excel using openpyxl/pandas. Returns structured result per sheet."""
    xls = pd.ExcelFile(io.BytesIO(content))
    sheets = []
    all_text_parts = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if df.empty:
            continue
        columns = list(df.columns)
        row_count = len(df)
        profile = _statistical_profile(df)
        samples = _sample_rows(df)
        text = _build_text_representation(f"{filename}:{sheet_name}", columns, row_count, profile, samples)
        sheets.append({
            "name": sheet_name,
            "row_count": row_count,
            "columns": columns,
            "statistical_profile": profile,
            "sample_rows": samples,
            "text": text,
        })
        all_text_parts.append(text)

    return {
        "parser": "native_excel",
        "filename": filename,
        "sheet_count": len(sheets),
        "sheets": sheets,
        "text": "\n\n".join(all_text_parts)[:_MAX_TEXT_CHARS],
    }


def is_native_parseable(filename: str) -> bool:
    """Check if a file should use native parsing instead of OCR/deep analysis."""
    lower = filename.lower()
    return lower.endswith((".csv", ".tsv", ".xlsx", ".xls"))


def parse_native(content: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Route to the appropriate native parser, or return None if not supported."""
    lower = filename.lower()
    if lower.endswith((".csv", ".tsv")):
        return parse_csv(content, filename)
    elif lower.endswith((".xlsx", ".xls")):
        return parse_excel(content, filename)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/extraction/test_native_parsers.py -v`
Expected: ALL PASS

- [ ] **Step 5: Wire native parser into `fileProcessor`**

In `src/api/dataHandler.py`, add an early return at the top of `fileProcessor()` (around line 469) for CSV/Excel files:

```python
# At the top of fileProcessor(), before existing extraction logic:
from src.extraction.native_parsers import is_native_parseable, parse_native

if isinstance(doc_content, bytes) and is_native_parseable(file_path):
    native_result = parse_native(doc_content, file_path)
    if native_result is not None:
        logger.info("Using native parser for %s (parser=%s, rows=%s)",
                     file_path, native_result["parser"], native_result.get("row_count", "N/A"))
        return {file_path: native_result}
```

- [ ] **Step 6: Commit**

```bash
git add src/extraction/native_parsers.py tests/extraction/test_native_parsers.py src/api/dataHandler.py
git commit -m "feat: native CSV/Excel parser bypassing OCR pipeline"
```

---

### Task 4: Add Neo4j Circuit Breaker

**Files:**
- Create: `src/kg/circuit_breaker.py`
- Modify: `src/kg/neo4j_store.py` (wrap `run_query` with circuit breaker)
- Test: `tests/kg/test_circuit_breaker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/kg/test_circuit_breaker.py
import time
import pytest


def test_circuit_opens_after_consecutive_failures():
    from src.kg.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=5)

    assert cb.state == "closed"

    # 3 consecutive failures should open the circuit
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()

    assert cb.state == "open"
    assert cb.should_allow_request() is False


def test_circuit_resets_on_success():
    from src.kg.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=5)

    cb.record_failure()
    cb.record_failure()
    cb.record_success()  # resets counter

    assert cb.state == "closed"
    assert cb.should_allow_request() is True


def test_circuit_half_opens_after_recovery_timeout():
    from src.kg.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=1)

    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"

    time.sleep(1.1)  # wait for recovery timeout

    assert cb.state == "half_open"
    assert cb.should_allow_request() is True  # allow one probe


def test_circuit_closes_after_successful_probe():
    from src.kg.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=1)

    cb.record_failure()
    cb.record_failure()
    cb.record_failure()

    time.sleep(1.1)
    assert cb.state == "half_open"

    cb.record_success()
    assert cb.state == "closed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/kg/test_circuit_breaker.py -v`
Expected: FAIL — `src.kg.circuit_breaker` does not exist.

- [ ] **Step 3: Implement circuit breaker**

```python
# src/kg/circuit_breaker.py
"""Circuit breaker for Neo4j connections.

States:
  closed    → requests flow normally, failures counted
  open      → requests blocked, skip KG enrichment
  half_open → one probe request allowed to test recovery
"""
import threading
import time
from typing import Literal

from src.observability.logging import get_logger

logger = get_logger(__name__)

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout_seconds: int = 300):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state: State = "closed"
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = "half_open"
            return self._state

    def should_allow_request(self) -> bool:
        current = self.state
        if current == "closed":
            return True
        if current == "half_open":
            return True  # allow one probe
        return False  # open

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state in ("half_open", "open"):
                logger.info("Neo4j circuit breaker CLOSED — connection recovered")
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                if self._state != "open":
                    logger.warning(
                        "Neo4j circuit breaker OPEN — %d consecutive failures, "
                        "skipping KG enrichment for %ds",
                        self._failure_count, self._recovery_timeout,
                    )
                self._state = "open"


# Singleton instance for Neo4j
neo4j_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=300)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/kg/test_circuit_breaker.py -v`
Expected: ALL PASS

- [ ] **Step 5: Wire circuit breaker into Neo4jStore.run_query**

In `src/kg/neo4j_store.py`, wrap `run_query` (around line 70):

```python
from src.kg.circuit_breaker import neo4j_breaker

# In run_query method:
def run_query(self, query: str, parameters: dict = None):
    if not neo4j_breaker.should_allow_request():
        return []  # circuit open, skip silently
    try:
        result = self._execute_query(query, parameters)
        neo4j_breaker.record_success()
        return result
    except Exception as exc:
        neo4j_breaker.record_failure()
        raise
```

- [ ] **Step 6: Commit**

```bash
git add src/kg/circuit_breaker.py tests/kg/test_circuit_breaker.py src/kg/neo4j_store.py
git commit -m "feat: Neo4j circuit breaker — auto-skip KG after 3 failures"
```

---

### Task 5: Add HITL Pipeline Statuses

**Files:**
- Modify: `src/api/statuses.py` (add new statuses)
- Test: `tests/api/test_statuses.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_statuses.py
def test_hitl_statuses_exist():
    from src.api import statuses

    # New HITL gate statuses
    assert statuses.PIPELINE_AWAITING_REVIEW_1 == "AWAITING_REVIEW_1"
    assert statuses.PIPELINE_AWAITING_REVIEW_2 == "AWAITING_REVIEW_2"
    assert statuses.PIPELINE_REJECTED == "REJECTED"
    assert statuses.PIPELINE_PROCESSING_IN_PROGRESS == "PROCESSING_IN_PROGRESS"
    assert statuses.PIPELINE_PROCESSING_COMPLETED == "PROCESSING_COMPLETED"
    assert statuses.PIPELINE_PROCESSING_FAILED == "PROCESSING_FAILED"

    # They should be in ALL_STATUSES
    assert "AWAITING_REVIEW_1" in statuses.ALL_STATUSES
    assert "AWAITING_REVIEW_2" in statuses.ALL_STATUSES
    assert "REJECTED" in statuses.ALL_STATUSES
    assert "PROCESSING_IN_PROGRESS" in statuses.ALL_STATUSES
    assert "PROCESSING_COMPLETED" in statuses.ALL_STATUSES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_statuses.py -v`
Expected: FAIL — attributes do not exist.

- [ ] **Step 3: Add new statuses**

In `src/api/statuses.py`, add after line 13:

```python
# HITL gate statuses
PIPELINE_AWAITING_REVIEW_1 = "AWAITING_REVIEW_1"  # Post-extraction, awaiting human approval
PIPELINE_AWAITING_REVIEW_2 = "AWAITING_REVIEW_2"  # Post-screening, awaiting human approval
PIPELINE_REJECTED = "REJECTED"  # Human rejected at either gate
PIPELINE_PROCESSING_IN_PROGRESS = "PROCESSING_IN_PROGRESS"  # Embedding + KG + intelligence
PIPELINE_PROCESSING_COMPLETED = "PROCESSING_COMPLETED"  # Full pipeline complete
PIPELINE_PROCESSING_FAILED = "PROCESSING_FAILED"  # Processing stage failed
```

Add them to `ALL_STATUSES`:

```python
ALL_STATUSES = {
    # ... existing entries ...
    PIPELINE_AWAITING_REVIEW_1,
    PIPELINE_AWAITING_REVIEW_2,
    PIPELINE_REJECTED,
    PIPELINE_PROCESSING_IN_PROGRESS,
    PIPELINE_PROCESSING_COMPLETED,
    PIPELINE_PROCESSING_FAILED,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_statuses.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/statuses.py tests/api/test_statuses.py
git commit -m "feat: add HITL pipeline statuses (AWAITING_REVIEW_1/2, REJECTED, PROCESSING)"
```

---

### Task 6: Wire HITL Gate 1 — Auto-transition to AWAITING_REVIEW_1

**Files:**
- Modify: `src/api/extraction_service.py:1884-1897` (after successful extraction)
- Test: `tests/api/test_extraction_timeout.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_extraction_timeout.py`:

```python
def test_extraction_transitions_to_awaiting_review():
    """After successful extraction, document status should move to AWAITING_REVIEW_1."""
    from unittest.mock import patch, MagicMock
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_1

    with patch("src.api.extraction_service.update_document_fields") as mock_update:
        from src.api.extraction_service import _transition_to_awaiting_review
        _transition_to_awaiting_review("test_doc_789")
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[0][0] == "test_doc_789"
        fields = call_args[0][1]
        assert fields["status"] == PIPELINE_AWAITING_REVIEW_1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_extraction_timeout.py::test_extraction_transitions_to_awaiting_review -v`
Expected: FAIL — `_transition_to_awaiting_review` does not exist.

- [ ] **Step 3: Implement the transition function**

Add in `src/api/extraction_service.py` near the status functions:

```python
from src.api.statuses import PIPELINE_AWAITING_REVIEW_1, PIPELINE_AWAITING_REVIEW_2, PIPELINE_REJECTED, PIPELINE_PROCESSING_IN_PROGRESS


def _transition_to_awaiting_review(doc_id: str) -> None:
    """After extraction completes, move document to AWAITING_REVIEW_1 for HITL gate."""
    update_document_fields(doc_id, {
        "status": PIPELINE_AWAITING_REVIEW_1,
        "awaiting_review_1_at": time.time(),
    })
    update_stage(doc_id, "extraction", {
        "status": "COMPLETED",
        "completed_at": time.time(),
        "awaiting_review": True,
    })
    logger.info("Document %s moved to AWAITING_REVIEW_1 (HITL gate 1)", doc_id)
```

- [ ] **Step 4: Wire into batch extraction success path**

In `extract_documents()`, after the extraction success block (after line 1897), add:

```python
# After: emit_progress(doc_id, "extraction", 0.20, ...)
# Add:
try:
    _transition_to_awaiting_review(doc_id)
except Exception:
    logger.warning("Failed to transition doc %s to AWAITING_REVIEW_1", doc_id, exc_info=True)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_extraction_timeout.py::test_extraction_transitions_to_awaiting_review -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/extraction_service.py tests/api/test_extraction_timeout.py
git commit -m "feat: auto-transition to AWAITING_REVIEW_1 after extraction"
```

---

### Task 7: Add HITL Review API Endpoints

**Files:**
- Create: `src/api/hitl_review.py`
- Modify: `src/main.py` (register new routes)
- Test: `tests/api/test_hitl_review.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_hitl_review.py
import pytest
from unittest.mock import patch, MagicMock


def test_approve_review_1_transitions_to_screening():
    """Approving at HITL Gate 1 should trigger screening."""
    from src.api.hitl_review import approve_review_gate_1
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_1, PIPELINE_SCREENING_IN_PROGRESS

    with patch("src.api.hitl_review._get_document_status", return_value=PIPELINE_AWAITING_REVIEW_1), \
         patch("src.api.hitl_review.update_document_fields") as mock_update, \
         patch("src.api.hitl_review._trigger_screening") as mock_screen:

        result = approve_review_gate_1("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "approved"
        mock_screen.assert_called_once_with("doc_123")


def test_approve_review_1_rejects_wrong_status():
    """Cannot approve if document is not in AWAITING_REVIEW_1."""
    from src.api.hitl_review import approve_review_gate_1

    with patch("src.api.hitl_review._get_document_status", return_value="EXTRACTION_IN_PROGRESS"):
        result = approve_review_gate_1("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "error"
        assert "AWAITING_REVIEW_1" in result["message"]


def test_reject_document():
    """Rejecting at either gate should mark document REJECTED."""
    from src.api.hitl_review import reject_document
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_1, PIPELINE_REJECTED

    with patch("src.api.hitl_review._get_document_status", return_value=PIPELINE_AWAITING_REVIEW_1), \
         patch("src.api.hitl_review.update_document_fields") as mock_update:

        result = reject_document("doc_123", reviewer="muthu@docwain.com", reason="Poor extraction quality")
        assert result["status"] == "rejected"
        call_fields = mock_update.call_args[0][1]
        assert call_fields["status"] == PIPELINE_REJECTED
        assert call_fields["rejection_reason"] == "Poor extraction quality"


def test_approve_review_2_transitions_to_processing():
    """Approving at HITL Gate 2 should trigger the heavy processing phase."""
    from src.api.hitl_review import approve_review_gate_2
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_2, PIPELINE_PROCESSING_IN_PROGRESS

    with patch("src.api.hitl_review._get_document_status", return_value=PIPELINE_AWAITING_REVIEW_2), \
         patch("src.api.hitl_review.update_document_fields") as mock_update, \
         patch("src.api.hitl_review._trigger_processing") as mock_proc:

        result = approve_review_gate_2("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "approved"
        mock_proc.assert_called_once_with("doc_123")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_hitl_review.py -v`
Expected: FAIL — `src.api.hitl_review` does not exist.

- [ ] **Step 3: Implement HITL review module**

```python
# src/api/hitl_review.py
"""HITL (Human-in-the-Loop) review gates for document processing pipeline.

Gate 1: After extraction — human reviews extracted content before screening.
Gate 2: After screening — human reviews screening results before processing (embedding + KG + intelligence).
"""
import time
from typing import Any, Dict

from src.api.document_status import update_document_fields, update_stage
from src.api.statuses import (
    PIPELINE_AWAITING_REVIEW_1,
    PIPELINE_AWAITING_REVIEW_2,
    PIPELINE_REJECTED,
    PIPELINE_SCREENING_IN_PROGRESS,
    PIPELINE_PROCESSING_IN_PROGRESS,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)

_REVIEWABLE_GATE_1 = {PIPELINE_AWAITING_REVIEW_1}
_REVIEWABLE_GATE_2 = {PIPELINE_AWAITING_REVIEW_2}
_REJECTABLE = {PIPELINE_AWAITING_REVIEW_1, PIPELINE_AWAITING_REVIEW_2}


def _get_document_status(doc_id: str) -> str:
    from src.api.document_status import get_documents_collection
    coll = get_documents_collection()
    if coll is None:
        return ""
    from bson import ObjectId
    doc = coll.find_one({"_id": ObjectId(doc_id)}, {"status": 1})
    if not doc:
        doc = coll.find_one({"_id": doc_id}, {"status": 1})
    return doc.get("status", "") if doc else ""


def _trigger_screening(doc_id: str) -> None:
    """Trigger the screening pipeline for a document."""
    update_document_fields(doc_id, {
        "status": PIPELINE_SCREENING_IN_PROGRESS,
        "screening_started_at": time.time(),
    })
    update_stage(doc_id, "screening", {"status": "IN_PROGRESS", "started_at": time.time()})
    logger.info("Document %s: screening triggered by HITL approval", doc_id)
    try:
        from src.api.screening_service import run_screening
        run_screening(doc_id)
    except Exception as exc:
        logger.error("Screening failed for %s: %s", doc_id, exc, exc_info=True)
        from src.api.statuses import PIPELINE_SCREENING_FAILED
        update_document_fields(doc_id, {"status": PIPELINE_SCREENING_FAILED})
        update_stage(doc_id, "screening", {"status": "FAILED", "error": str(exc)})


def _trigger_processing(doc_id: str) -> None:
    """Trigger the heavy processing phase (embedding + KG + intelligence)."""
    update_document_fields(doc_id, {
        "status": PIPELINE_PROCESSING_IN_PROGRESS,
        "processing_started_at": time.time(),
    })
    update_stage(doc_id, "processing", {"status": "IN_PROGRESS", "started_at": time.time()})
    logger.info("Document %s: processing triggered by HITL approval", doc_id)
    # Processing implementation will be added in Phase 2b
    # For now, fall through to existing embedding pipeline
    try:
        from src.api.extraction_service import _run_embedding_pipeline
        _run_embedding_pipeline(doc_id)
    except Exception as exc:
        logger.error("Processing failed for %s: %s", doc_id, exc, exc_info=True)
        from src.api.statuses import PIPELINE_PROCESSING_FAILED
        update_document_fields(doc_id, {"status": PIPELINE_PROCESSING_FAILED})
        update_stage(doc_id, "processing", {"status": "FAILED", "error": str(exc)})


def approve_review_gate_1(doc_id: str, reviewer: str) -> Dict[str, Any]:
    """Approve document at HITL Gate 1 (post-extraction). Triggers screening."""
    current = _get_document_status(doc_id)
    if current not in _REVIEWABLE_GATE_1:
        return {
            "status": "error",
            "message": f"Document must be in {PIPELINE_AWAITING_REVIEW_1} to approve at Gate 1 (current: {current})",
        }

    update_document_fields(doc_id, {
        "review_1_approved_by": reviewer,
        "review_1_approved_at": time.time(),
    })
    logger.info("Document %s: HITL Gate 1 approved by %s", doc_id, reviewer)

    _trigger_screening(doc_id)
    return {"status": "approved", "gate": 1, "document_id": doc_id, "next_stage": "screening"}


def approve_review_gate_2(doc_id: str, reviewer: str) -> Dict[str, Any]:
    """Approve document at HITL Gate 2 (post-screening). Triggers processing."""
    current = _get_document_status(doc_id)
    if current not in _REVIEWABLE_GATE_2:
        return {
            "status": "error",
            "message": f"Document must be in {PIPELINE_AWAITING_REVIEW_2} to approve at Gate 2 (current: {current})",
        }

    update_document_fields(doc_id, {
        "review_2_approved_by": reviewer,
        "review_2_approved_at": time.time(),
    })
    logger.info("Document %s: HITL Gate 2 approved by %s", doc_id, reviewer)

    _trigger_processing(doc_id)
    return {"status": "approved", "gate": 2, "document_id": doc_id, "next_stage": "processing"}


def reject_document(doc_id: str, reviewer: str, reason: str) -> Dict[str, Any]:
    """Reject document at either HITL gate. Document will not proceed."""
    current = _get_document_status(doc_id)
    if current not in _REJECTABLE:
        return {
            "status": "error",
            "message": f"Document must be in a review state to reject (current: {current})",
        }

    update_document_fields(doc_id, {
        "status": PIPELINE_REJECTED,
        "rejected_by": reviewer,
        "rejected_at": time.time(),
        "rejection_reason": reason,
    })
    logger.info("Document %s: REJECTED by %s — %s", doc_id, reviewer, reason)
    return {"status": "rejected", "document_id": doc_id, "reason": reason}


def request_reextraction(doc_id: str, reviewer: str, reason: str = "") -> Dict[str, Any]:
    """Request re-extraction at HITL Gate 1. Resets to UNDER_REVIEW for fresh extraction."""
    current = _get_document_status(doc_id)
    if current not in _REVIEWABLE_GATE_1:
        return {
            "status": "error",
            "message": f"Document must be in {PIPELINE_AWAITING_REVIEW_1} to request re-extraction (current: {current})",
        }

    from src.api.statuses import STATUS_UNDER_REVIEW
    update_document_fields(doc_id, {
        "status": STATUS_UNDER_REVIEW,
        "reextraction_requested_by": reviewer,
        "reextraction_requested_at": time.time(),
        "reextraction_reason": reason,
    })
    logger.info("Document %s: re-extraction requested by %s", doc_id, reviewer)
    return {"status": "reextraction_requested", "document_id": doc_id}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_hitl_review.py -v`
Expected: ALL PASS

- [ ] **Step 5: Register routes in main.py**

In `src/main.py`, add near the other route handlers:

```python
# HITL Review endpoints
@app.post("/api/review/{doc_id}/approve/gate1")
async def hitl_approve_gate_1(doc_id: str, request: Request):
    body = await request.json()
    reviewer = body.get("reviewer", "unknown")
    from src.api.hitl_review import approve_review_gate_1
    return approve_review_gate_1(doc_id, reviewer=reviewer)

@app.post("/api/review/{doc_id}/approve/gate2")
async def hitl_approve_gate_2(doc_id: str, request: Request):
    body = await request.json()
    reviewer = body.get("reviewer", "unknown")
    from src.api.hitl_review import approve_review_gate_2
    return approve_review_gate_2(doc_id, reviewer=reviewer)

@app.post("/api/review/{doc_id}/reject")
async def hitl_reject(doc_id: str, request: Request):
    body = await request.json()
    reviewer = body.get("reviewer", "unknown")
    reason = body.get("reason", "")
    from src.api.hitl_review import reject_document
    return reject_document(doc_id, reviewer=reviewer, reason=reason)

@app.post("/api/review/{doc_id}/reextract")
async def hitl_reextract(doc_id: str, request: Request):
    body = await request.json()
    reviewer = body.get("reviewer", "unknown")
    reason = body.get("reason", "")
    from src.api.hitl_review import request_reextraction
    return request_reextraction(doc_id, reviewer=reviewer, reason=reason)
```

- [ ] **Step 6: Commit**

```bash
git add src/api/hitl_review.py tests/api/test_hitl_review.py src/main.py
git commit -m "feat: HITL review API endpoints (approve/reject/reextract at gate 1 and 2)"
```

---

### Task 8: Wire HITL Gate 2 — Screening Completion Transitions to AWAITING_REVIEW_2

**Files:**
- Modify: `src/api/screening_service.py:68-94` (after screening completes)
- Test: `tests/api/test_hitl_review.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_hitl_review.py`:

```python
def test_screening_completion_transitions_to_awaiting_review_2():
    """After screening completes, document should move to AWAITING_REVIEW_2."""
    from src.api.hitl_review import transition_to_awaiting_review_2
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_2

    with patch("src.api.hitl_review.update_document_fields") as mock_update:
        transition_to_awaiting_review_2("doc_456")
        call_fields = mock_update.call_args[0][1]
        assert call_fields["status"] == PIPELINE_AWAITING_REVIEW_2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_hitl_review.py::test_screening_completion_transitions_to_awaiting_review_2 -v`
Expected: FAIL — `transition_to_awaiting_review_2` does not exist.

- [ ] **Step 3: Add transition function to hitl_review.py**

Append to `src/api/hitl_review.py`:

```python
def transition_to_awaiting_review_2(doc_id: str) -> None:
    """After screening completes, move to AWAITING_REVIEW_2 for HITL Gate 2."""
    update_document_fields(doc_id, {
        "status": PIPELINE_AWAITING_REVIEW_2,
        "awaiting_review_2_at": time.time(),
    })
    update_stage(doc_id, "screening", {
        "status": "COMPLETED",
        "completed_at": time.time(),
        "awaiting_review": True,
    })
    logger.info("Document %s moved to AWAITING_REVIEW_2 (HITL gate 2)", doc_id)
```

- [ ] **Step 4: Wire into screening completion in screening_service.py**

In `src/api/screening_service.py`, after the `promote_to_screening_completed()` function (around line 68), add a call to transition:

```python
# After screening is promoted to COMPLETED, add:
try:
    from src.api.hitl_review import transition_to_awaiting_review_2
    transition_to_awaiting_review_2(doc_id)
except Exception:
    logger.warning("Failed to transition doc %s to AWAITING_REVIEW_2", doc_id, exc_info=True)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/api/test_hitl_review.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/hitl_review.py src/api/screening_service.py tests/api/test_hitl_review.py
git commit -m "feat: auto-transition to AWAITING_REVIEW_2 after screening"
```

---

### Task 9: Add KG Index Loading at Server Startup

**Files:**
- Create: `src/kg/kg_cache.py`
- Modify: `src/api/app_lifespan.py:206-216` (add KG cache warming after Neo4j init)
- Test: `tests/kg/test_kg_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/kg/test_kg_cache.py
import pytest
from unittest.mock import MagicMock, patch


def test_kg_cache_loads_entity_catalog():
    from src.kg.kg_cache import KGCache

    mock_store = MagicMock()
    mock_store.run_query.return_value = [
        {"name": "Acme Corp", "type": "ORG", "doc_count": 5},
        {"name": "John Smith", "type": "PERSON", "doc_count": 3},
    ]

    cache = KGCache()
    cache.warm(neo4j_store=mock_store)

    assert cache.entity_count == 2
    entities = cache.get_entity_catalog()
    assert any(e["name"] == "Acme Corp" for e in entities)
    assert any(e["name"] == "John Smith" for e in entities)


def test_kg_cache_survives_neo4j_failure():
    from src.kg.kg_cache import KGCache

    mock_store = MagicMock()
    mock_store.run_query.side_effect = Exception("Neo4j down")

    cache = KGCache()
    cache.warm(neo4j_store=mock_store)

    # Should not raise, just have empty catalog
    assert cache.entity_count == 0
    assert cache.is_warmed is False


def test_kg_cache_entity_lookup():
    from src.kg.kg_cache import KGCache

    mock_store = MagicMock()
    mock_store.run_query.return_value = [
        {"name": "Acme Corp", "type": "ORG", "doc_count": 5},
    ]

    cache = KGCache()
    cache.warm(neo4j_store=mock_store)

    match = cache.lookup_entity("Acme Corp")
    assert match is not None
    assert match["type"] == "ORG"

    no_match = cache.lookup_entity("Nonexistent")
    assert no_match is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/kg/test_kg_cache.py -v`
Expected: FAIL — `src.kg.kg_cache` does not exist.

- [ ] **Step 3: Implement KG cache**

```python
# src/kg/kg_cache.py
"""In-memory KG index cache for fast entity lookups at query time.

Loaded at server startup, refreshed every 15 minutes.
Falls back gracefully if Neo4j is unavailable.
"""
import threading
import time
from typing import Any, Dict, List, Optional

from src.observability.logging import get_logger

logger = get_logger(__name__)

_REFRESH_INTERVAL_SECONDS = 900  # 15 minutes

_ENTITY_CATALOG_QUERY = """
MATCH (e:Entity)
WITH e.name AS name, labels(e) AS types, e.doc_count AS doc_count
RETURN name, head([t IN types WHERE t <> 'Entity']) AS type, coalesce(doc_count, 1) AS doc_count
ORDER BY doc_count DESC
LIMIT 10000
"""

_RELATIONSHIP_SCHEMA_QUERY = """
MATCH ()-[r]->()
WITH type(r) AS rel_type, count(*) AS freq
RETURN rel_type, freq
ORDER BY freq DESC
LIMIT 100
"""


class KGCache:
    def __init__(self):
        self._entities: List[Dict[str, Any]] = []
        self._entity_index: Dict[str, Dict[str, Any]] = {}  # name -> entity
        self._relationship_schema: List[Dict[str, Any]] = []
        self._is_warmed = False
        self._last_refresh: float = 0.0
        self._lock = threading.Lock()

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def is_warmed(self) -> bool:
        return self._is_warmed

    def warm(self, neo4j_store) -> None:
        """Load KG index from Neo4j into memory."""
        try:
            entities = neo4j_store.run_query(_ENTITY_CATALOG_QUERY)
            rel_schema = neo4j_store.run_query(_RELATIONSHIP_SCHEMA_QUERY)

            with self._lock:
                self._entities = [dict(e) for e in entities] if entities else []
                self._entity_index = {e["name"]: e for e in self._entities}
                self._relationship_schema = [dict(r) for r in rel_schema] if rel_schema else []
                self._is_warmed = True
                self._last_refresh = time.time()

            logger.info(
                "KG cache warmed: %d entities, %d relationship types",
                len(self._entities), len(self._relationship_schema),
            )
        except Exception as exc:
            logger.warning("KG cache warming failed (Neo4j unavailable): %s", exc)
            self._is_warmed = False

    def get_entity_catalog(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entities)

    def get_relationship_schema(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._relationship_schema)

    def lookup_entity(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._entity_index.get(name)

    def needs_refresh(self) -> bool:
        return time.time() - self._last_refresh > _REFRESH_INTERVAL_SECONDS


# Singleton
kg_cache = KGCache()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/kg/test_kg_cache.py -v`
Expected: ALL PASS

- [ ] **Step 5: Wire into app startup**

In `src/api/app_lifespan.py`, after the Neo4j/GraphAugmenter initialization (line 216), add:

```python
    # Warm KG entity cache for fast query-time lookups
    if graph_augmenter:
        try:
            from src.kg.kg_cache import kg_cache
            kg_cache.warm(neo4j_store=neo4j_store)
        except Exception as exc:
            logger.warning("KG cache warming skipped: %s", exc)
```

- [ ] **Step 6: Commit**

```bash
git add src/kg/kg_cache.py tests/kg/test_kg_cache.py src/api/app_lifespan.py
git commit -m "feat: KG entity cache loaded at startup, refreshed every 15min"
```

---

## Phase 1: Evaluation Harness

### Task 10: Create Evaluation Framework

**Files:**
- Create: `eval/test_bank.json`
- Create: `eval/eval_runner.py`
- Create: `eval/eval_report.py`
- Test: `tests/eval/test_eval_runner.py`

- [ ] **Step 1: Create test bank schema with seed cases**

```python
# eval/test_bank.json
[
  {
    "id": "eval_001",
    "category": "simple_lookup",
    "query": "What is the total revenue for Q2 2024?",
    "profile_id": null,
    "subscription_id": null,
    "expected_facts": ["total revenue", "Q2 2024"],
    "negative_facts": ["Q1 2024", "projected"],
    "expected_doc_ids": [],
    "notes": "Template — populate with real doc IDs after first extraction run"
  },
  {
    "id": "eval_002",
    "category": "multi_doc",
    "query": "How has the salary changed between the 2023 and 2024 reports?",
    "profile_id": null,
    "subscription_id": null,
    "expected_facts": ["salary", "2023", "2024", "change"],
    "negative_facts": [],
    "expected_doc_ids": [],
    "notes": "Template — requires two documents from different periods"
  },
  {
    "id": "eval_003",
    "category": "table",
    "query": "What are the line items in the invoice?",
    "profile_id": null,
    "subscription_id": null,
    "expected_facts": ["line item", "amount"],
    "negative_facts": [],
    "expected_doc_ids": [],
    "notes": "Template — tests table extraction accuracy"
  },
  {
    "id": "eval_004",
    "category": "analytical",
    "query": "Based on the financial data, what areas need improvement?",
    "profile_id": null,
    "subscription_id": null,
    "expected_facts": ["analysis", "recommendation"],
    "negative_facts": [],
    "expected_doc_ids": [],
    "notes": "Template — tests analytical intelligence"
  }
]
```

- [ ] **Step 2: Write the eval runner**

```python
# eval/eval_runner.py
"""Evaluation runner for DocWain RAG quality metrics.

Usage:
    python -m eval.eval_runner --test-bank eval/test_bank.json --output eval/results/
"""
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List


def load_test_bank(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def run_single_eval(case: Dict[str, Any], ask_fn) -> Dict[str, Any]:
    """Run a single evaluation case and collect metrics."""
    start = time.time()
    try:
        response = ask_fn(
            query=case["query"],
            profile_id=case.get("profile_id"),
            subscription_id=case.get("subscription_id"),
        )
    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "status": "error",
            "error": str(exc),
            "latency_ms": round((time.time() - start) * 1000),
        }

    latency_ms = round((time.time() - start) * 1000)
    answer_text = response.get("answer", "") if isinstance(response, dict) else str(response)
    retrieved_ids = response.get("source_doc_ids", []) if isinstance(response, dict) else []

    # Fact presence check
    facts_found = []
    facts_missing = []
    for fact in case.get("expected_facts", []):
        if fact.lower() in answer_text.lower():
            facts_found.append(fact)
        else:
            facts_missing.append(fact)

    # Negative fact check (hallucination)
    hallucinated = []
    for neg in case.get("negative_facts", []):
        if neg.lower() in answer_text.lower():
            hallucinated.append(neg)

    # Retrieval recall
    expected_docs = set(case.get("expected_doc_ids", []))
    retrieved_set = set(retrieved_ids)
    retrieval_recall = (
        len(expected_docs & retrieved_set) / len(expected_docs)
        if expected_docs else None
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "status": "ok",
        "latency_ms": latency_ms,
        "answer_length": len(answer_text),
        "facts_found": facts_found,
        "facts_missing": facts_missing,
        "fact_coverage": len(facts_found) / max(len(case.get("expected_facts", [])), 1),
        "hallucinated_facts": hallucinated,
        "hallucination_count": len(hallucinated),
        "retrieval_recall": retrieval_recall,
    }


def run_eval(test_bank_path: str, ask_fn, output_dir: str = "eval/results") -> Dict[str, Any]:
    """Run full evaluation suite."""
    cases = load_test_bank(test_bank_path)
    results = []
    for case in cases:
        result = run_single_eval(case, ask_fn)
        results.append(result)

    # Aggregate
    ok_results = [r for r in results if r["status"] == "ok"]
    summary = {
        "total_cases": len(cases),
        "successful": len(ok_results),
        "errors": len(results) - len(ok_results),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in ok_results) / max(len(ok_results), 1)),
        "avg_fact_coverage": round(sum(r["fact_coverage"] for r in ok_results) / max(len(ok_results), 1), 3),
        "total_hallucinations": sum(r["hallucination_count"] for r in ok_results),
        "by_category": {},
    }

    for cat in set(r["category"] for r in ok_results):
        cat_results = [r for r in ok_results if r["category"] == cat]
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in cat_results) / len(cat_results)),
            "avg_fact_coverage": round(sum(r["fact_coverage"] for r in cat_results) / len(cat_results), 3),
        }

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"{output_dir}/eval_{timestamp}.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    return summary
```

- [ ] **Step 3: Write the eval report tool**

```python
# eval/eval_report.py
"""Compare evaluation results across runs."""
import json
import sys
from pathlib import Path
from typing import List


def load_results(results_dir: str) -> List[dict]:
    results = []
    for f in sorted(Path(results_dir).glob("eval_*.json")):
        with open(f) as fh:
            data = json.load(fh)
            data["filename"] = f.name
            results.append(data)
    return results


def compare_latest(results_dir: str = "eval/results") -> None:
    runs = load_results(results_dir)
    if len(runs) < 1:
        print("No evaluation results found.")
        return

    latest = runs[-1]
    print(f"Latest run: {latest['filename']}")
    print(f"  Cases: {latest['summary']['total_cases']}")
    print(f"  Avg latency: {latest['summary']['avg_latency_ms']}ms")
    print(f"  Avg fact coverage: {latest['summary']['avg_fact_coverage']}")
    print(f"  Hallucinations: {latest['summary']['total_hallucinations']}")

    if len(runs) >= 2:
        prev = runs[-2]
        print(f"\nVs previous ({prev['filename']}):")
        lat_delta = latest['summary']['avg_latency_ms'] - prev['summary']['avg_latency_ms']
        cov_delta = latest['summary']['avg_fact_coverage'] - prev['summary']['avg_fact_coverage']
        hal_delta = latest['summary']['total_hallucinations'] - prev['summary']['total_hallucinations']
        print(f"  Latency: {'+' if lat_delta > 0 else ''}{lat_delta}ms")
        print(f"  Fact coverage: {'+' if cov_delta > 0 else ''}{cov_delta:.3f}")
        print(f"  Hallucinations: {'+' if hal_delta > 0 else ''}{hal_delta}")


if __name__ == "__main__":
    compare_latest(sys.argv[1] if len(sys.argv) > 1 else "eval/results")
```

- [ ] **Step 4: Write the test**

```python
# tests/eval/test_eval_runner.py
import json
import pytest


def test_run_single_eval_captures_facts():
    from eval.eval_runner import run_single_eval

    case = {
        "id": "test_001",
        "category": "simple_lookup",
        "query": "What is the revenue?",
        "expected_facts": ["revenue", "1.5M"],
        "negative_facts": ["projected"],
        "expected_doc_ids": [],
    }

    def mock_ask(query, profile_id=None, subscription_id=None):
        return {"answer": "The total revenue is 1.5M for the period.", "source_doc_ids": []}

    result = run_single_eval(case, mock_ask)

    assert result["status"] == "ok"
    assert result["fact_coverage"] == 1.0
    assert result["hallucination_count"] == 0
    assert "revenue" in result["facts_found"]
    assert "1.5M" in result["facts_found"]


def test_run_single_eval_detects_hallucination():
    from eval.eval_runner import run_single_eval

    case = {
        "id": "test_002",
        "category": "simple_lookup",
        "query": "What is the revenue?",
        "expected_facts": ["revenue"],
        "negative_facts": ["projected"],
        "expected_doc_ids": [],
    }

    def mock_ask(query, profile_id=None, subscription_id=None):
        return {"answer": "The projected revenue is $2M.", "source_doc_ids": []}

    result = run_single_eval(case, mock_ask)

    assert result["hallucination_count"] == 1
    assert "projected" in result["hallucinated_facts"]


def test_run_single_eval_handles_errors():
    from eval.eval_runner import run_single_eval

    case = {"id": "test_003", "category": "simple_lookup", "query": "test", "expected_facts": [], "negative_facts": [], "expected_doc_ids": []}

    def failing_ask(query, **kwargs):
        raise ConnectionError("LLM unavailable")

    result = run_single_eval(case, failing_ask)
    assert result["status"] == "error"
    assert "LLM unavailable" in result["error"]
```

- [ ] **Step 5: Run tests**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/eval/test_eval_runner.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add eval/test_bank.json eval/eval_runner.py eval/eval_report.py tests/eval/test_eval_runner.py
git commit -m "feat: evaluation harness with test bank, runner, and report comparison"
```

---

## Phases 2-5: Subsequent Plans

The remaining phases should be planned and executed after Phase 0 and Phase 1 are complete and validated. Each phase becomes its own implementation plan:

### Phase 2a Plan (to be written after Phase 1 completes)
- **Task 11**: Query Classifier — rule-based SIMPLE/COMPLEX/ANALYTICAL/CONVERSATIONAL router
- **Task 12**: Fast Path Pipeline — skip planner, reranker top-3, no RepairLoop
- **Task 13**: Streaming from UNDERSTAND phase — progressive status updates

### Phase 2b Plan (parallel with 2a)
- **Task 14**: Smart Extraction Routing — type+size classifier at extraction entry
- **Task 15**: Parallel Page Processing — ThreadPoolExecutor in splitter/merger
- **Task 16**: Priority Queue — heapq-based replacement for FIFO batch loop

### Phase 3a Plan
- **Task 17**: Parallelized Complex Path — concurrent Planner+Retrieval+KG+DocIntel
- **Task 18**: Redis Caching Layer — query embedding, search results, full response caches

### Phase 3b Plan
- **Task 19**: Parent-Child Chunking — overlap-aware chunking with parent expansion
- **Task 20**: Embedding Model Upgrade — bge-m3 with shadow collection validation

### Phase 4 Plan
- **Task 21**: Cross-Document Reasoning — KG-guided multi-doc retrieval
- **Task 22**: Citation Verification — separate fast-model verification pass
- **Task 23**: Pre-computed Document Intelligence — temporal/numerical/cross-doc analysis at processing time
- **Task 24**: Analytical Reasoner — domain-aware analysis prompt with structured output

### Phase 5 Plan
- **Task 25**: Reranker to GPU / ColBERT replacement
- **Task 26**: Embedding Service extraction from in-process to separate service
- **Task 27**: Table/Form Extraction Validation gate
