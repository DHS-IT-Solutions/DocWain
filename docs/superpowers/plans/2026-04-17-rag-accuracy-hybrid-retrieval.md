# RAG Accuracy — Hybrid Retrieval + Grounding Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-04-17-rag-accuracy-hybrid-retrieval-design.md`

**Goal:** Lift DocWain RAGAS metrics from faithfulness 0.439 / context recall 0.561 to **faithfulness ≥0.80 and context recall ≥0.75** on the 106-query `intensive_test.py` bank, by wiring already-written SPLADE sparse retrieval, RRF fusion, and Neo4j KG entity expansion into the live retriever, and by fixing the universally-false `_check_grounding` gate.

**Architecture:** This plan is a wiring job, not a rewrite. The live path is `core_agent.py` → `src/retrieval/retriever.py::UnifiedRetriever` (dense + keyword-fallback). We add a sparse branch using the existing `src/embedding/sparse.py::SparseEncoder`, fuse it with dense via the existing `src/retrieval/fusion.py::reciprocal_rank_fusion`, pull KG entity expansion from the existing `src/kg/retrieval.py::GraphAugmenter` (already initialised in `app_lifespan.py`), and fix the existing `Reasoner._check_grounding`. One new script handles the one-time SPLADE backfill of Qdrant's already-provisioned `keywords_vector` slot.

**Tech Stack:** Python 3.12, Qdrant (named vectors `content_vector` dense + `keywords_vector` sparse), Neo4j (via GraphAugmenter), SPLADE-v3 (naver/splade-v3), BGE-large-en-v1.5 for dense, FastAPI, pytest.

---

## File Structure

**Modified:**

- `src/generation/reasoner.py` — fix `_check_grounding` (which gate is over-firing; diagnose first).
- `src/retrieval/retriever.py` — `UnifiedRetriever` accepts `sparse_encoder` and `graph_augmenter`, adds sparse retrieval + RRF + KG expansion paths.
- `src/agent/core_agent.py` — pass `graph_augmenter` into `UnifiedRetriever` at construction; no per-call change needed because KG expansion uses the query string (GraphAugmenter does its own entity extraction).
- `src/api/rag_state.py` — add `sparse_encoder` field on `AppState`.
- `src/api/app_lifespan.py` — load `SparseEncoder` at startup alongside `embedding_model` and `reranker`.
- `src/embedding/pipeline/qdrant_ingestion.py` — populate `sparse_vector` at the existing `sparse_vector=None` site.

**Created:**

- `scripts/backfill_sparse_vectors.py` — one-shot backfill script (inventory / dry-run / full modes).
- `tests/generation/test_grounding_check.py` — lock reasoner grounding behaviour.
- `tests/retrieval/test_hybrid_retrieve.py` — lock dense+sparse+RRF retrieval.
- `tests/retrieval/test_kg_expansion.py` — lock KG expansion path.
- `tests/scripts/test_backfill_sparse.py` — dry-run / resumability / idempotence of the backfill script.
- `tests/embedding/test_sparse_encoder.py` — lock `SparseEncoder.encode` output shape and `sparse_to_qdrant` conversion (skip if file already exists and covers this).

**Deleted:**

- `src/retrieval/unified_retriever.py` — confirmed unimported by any live code.
- `tests/test_retrieval_integration.py` — only importer of the dead module; exercises no live path.

---

## Design note (refinement of spec §3)

The spec said `retrieve()` would take `query_entities: Optional[List[str]]`. Implementation inspection shows `src/kg/retrieval.py::GraphAugmenter.augment(query, subscription_id, profile_id)` already extracts entities internally using its own `EntityExtractor` and returns `GraphHints(evidence_chunk_ids=[...], doc_ids=[...])`. Rather than duplicate entity extraction in the retriever, the plan injects the pre-initialised `graph_augmenter` from `AppState` into the `UnifiedRetriever` constructor and calls `augment(query, ...)`. This is simpler, re-uses `GraphAugmenter`'s existing Neo4j caching, and removes the need to plumb `query_entities` through six call sites in `core_agent.py`. Net: `core_agent.py` changes reduce to one constructor argument at a single line. The spec's success criteria are unchanged.

---

## Task 0: Capture RAGAS baseline

**Why:** Every subsequent step diffs against this baseline. The Apr 11 metrics predate the vLLM model-symlink repoint to the HF-recovered weights and are not a valid baseline.

**Files:**
- Create: `tests/ragas_metrics.baseline.json` (snapshot of current RAGAS run, committed)

- [ ] **Step 1: Verify API and vLLM are running**

Run: `curl -s http://localhost:8000/api/health | head -c 200 && echo && curl -s http://localhost:8100/health`

Expected: Both return HTTP 200-compatible output (health JSON from the API, empty-body 200 from vLLM).

- [ ] **Step 2: Run the intensive test**

Run: `python scripts/intensive_test.py`

Expected: Script runs ~10–20 minutes, prints per-query status, writes `/tmp/intensive_test_results.json` and `tests/ragas_metrics.json`.

- [ ] **Step 3: Copy to baseline**

Run: `cp tests/ragas_metrics.json tests/ragas_metrics.baseline.json`

- [ ] **Step 4: Commit baseline**

Run:
```bash
git add -f tests/ragas_metrics.baseline.json tests/ragas_metrics.json
git commit -m "test(rag): capture pre-fix RAGAS baseline on current weights"
```

---

## Task 1: Diagnose the grounding check failure

**Why:** `tests/quality_audit_results.json` shows all 13 cases returning `grounded: false` even when content is correct. We must identify which of the two gates in `Reasoner._check_grounding` is over-firing (20% ungrounded-numbers vs 15% word-overlap) before editing. No speculative fixes.

**Files:**
- Create: `tests/generation/test_grounding_diagnose.py` (temporary diagnostic, deleted in Task 2 after fix)

- [ ] **Step 1: Write the diagnostic**

Create `tests/generation/test_grounding_diagnose.py`:

```python
"""One-shot diagnostic: which gate in Reasoner._check_grounding over-fires?

Temporary file. Deleted after the grounding fix lands in Task 2.
"""
import json
import logging
import re
from pathlib import Path

from src.generation.reasoner import Reasoner, _NUMBER_RE


def test_diagnose_which_gate_fires(caplog):
    caplog.set_level(logging.WARNING, logger="src.generation.reasoner")

    audit_path = Path("tests/quality_audit_results.json")
    audit = json.loads(audit_path.read_text())

    reasoner = Reasoner(llm_gateway=None)

    # Build synthetic evidence per case: we don't have the original evidence
    # so we use the response itself as "evidence" to isolate which gate
    # fires for bare answers. This answers: given real evidence that CONTAINS
    # the answer numbers/words, does _check_grounding still return False?
    for case in audit["results"]:
        answer = case["response_preview"]
        # evidence = answer text (so numbers and words ARE present)
        evidence = [{"text": answer}]
        grounded = reasoner._check_grounding(answer, evidence, doc_context=None)
        nums_in_answer = set(_NUMBER_RE.findall(answer))
        nums_in_evidence = set(_NUMBER_RE.findall(answer))  # same text
        words_in_answer = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', answer))
        words_in_evidence = words_in_answer  # same text
        print(f"\n{case['name']}: grounded={grounded}")
        print(f"  answer_nums={len(nums_in_answer)} evidence_nums={len(nums_in_evidence)}")
        print(f"  answer_words={len(words_in_answer)} evidence_words={len(words_in_evidence)}")
        print(f"  overlap_ratio={len(words_in_answer & words_in_evidence) / max(len(words_in_answer), 1):.2f}")

    # Report which log warnings fired
    print("\n--- WARNINGS EMITTED ---")
    for record in caplog.records:
        print(f"  {record.message}")
```

- [ ] **Step 2: Run the diagnostic and capture output**

Run: `pytest tests/generation/test_grounding_diagnose.py -v -s 2>&1 | tee /tmp/grounding_diag.log`

Expected: Test passes (no assertions). Output shows per-case grounding decision and warning messages. If many cases show `grounded=False` even when answer==evidence, the check has a structural bug. Read the output, identify which gate's warning fires most.

- [ ] **Step 3: Record the diagnosis**

Write a one-line note in the PR description or commit message for Task 2 stating which gate is the culprit (e.g., "20% ungrounded-numbers gate misfires because audit fixtures lack evidence text — fix is to make the check evidence-aware rather than answer-only"). Do NOT edit code yet.

- [ ] **Step 4: Commit the diagnostic**

Run:
```bash
git add tests/generation/test_grounding_diagnose.py
git commit -m "test(rag): add temporary diagnostic for _check_grounding gate misfire"
```

---

## Task 2: Fix the grounding check

**Why:** Make `grounded` a meaningful signal again. This fix doesn't move RAGAS numbers on its own but is mandatory — it lets subsequent steps verify correctness.

**Files:**
- Modify: `src/generation/reasoner.py` (method `_check_grounding`, lines ~232-352)
- Create: `tests/generation/test_grounding_check.py`
- Delete: `tests/generation/test_grounding_diagnose.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/generation/test_grounding_check.py`:

```python
"""Regression-lock the grounding gate fix.

Fixtures cover:
- grounded=True when answer numbers/words trace to evidence (normal case)
- grounded=True when answer is short (< 20 chars)
- grounded=False when answer numbers are fabricated (number gate)
- grounded=False when answer shares <15% words with evidence (word gate)
- grounded=False when evidence is empty
- grounded=True when evidence items contain text reached via doc_context.key_facts
"""
from src.generation.reasoner import Reasoner


def _reasoner():
    return Reasoner(llm_gateway=None)


def test_grounded_when_answer_traces_to_evidence():
    answer = "The invoice total is **$9,000.00** per document INV-42."
    evidence = [{"text": "Invoice INV-42 shows a total of $9,000.00 due on 2026-03-15."}]
    assert _reasoner()._check_grounding(answer, evidence) is True


def test_short_answer_grounded_when_evidence_exists():
    answer = "**$9,000.00**"
    evidence = [{"text": "Invoice total: $9,000.00"}]
    assert _reasoner()._check_grounding(answer, evidence) is True


def test_ungrounded_when_numbers_fabricated():
    answer = "The total is **$99,999.00** as of **2099-12-31**."
    evidence = [{"text": "The invoice shows various line items but no totals."}]
    assert _reasoner()._check_grounding(answer, evidence) is False


def test_ungrounded_when_word_overlap_below_threshold():
    answer = "The quick brown fox jumps over the lazy dog in the meadow."
    evidence = [{"text": "Bacteria cultures proliferate inside Petri dishes."}]
    assert _reasoner()._check_grounding(answer, evidence) is False


def test_ungrounded_when_evidence_empty():
    assert _reasoner()._check_grounding("anything", []) is False


def test_grounded_via_doc_context_key_facts():
    answer = "Candidate **Jessica Jones** has 8 years of experience."
    evidence = [{"text": "unrelated chunk text"}]
    doc_context = {"key_facts": ["Jessica Jones has 8 years of experience."]}
    assert _reasoner()._check_grounding(answer, evidence, doc_context=doc_context) is True
```

- [ ] **Step 2: Run tests and verify expected failures**

Run: `pytest tests/generation/test_grounding_check.py -v`

Expected: At least one or two tests FAIL, matching what the diagnostic in Task 1 identified. The exact failing tests depend on which gate is broken.

- [ ] **Step 3: Apply the fix**

Open `src/generation/reasoner.py`. Locate `_check_grounding` (around line 232). Based on the diagnostic in Task 1:

- If the **number gate** misfires (most likely): relax by only checking numbers when the answer contains more than 2 numbers AND evidence text is non-trivial. If the answer has ≤2 numbers, skip the number gate entirely. This matches real-world answers where a total or a single count is legitimately the only number.

- If the **word gate** misfires: raise the overlap threshold logic to divide by `max(len(answer_words), 20)` so short answers (e.g., "Not found in provided documents") don't fail by accident; and treat answers with < 10 meaningful words as trivially grounded when any evidence exists.

- If **both** misfire: apply both relaxations.

Replace the body of `_check_grounding` with the corrected logic. Keep the method signature, the `logger.warning` calls, and the False short-circuit on empty evidence. Example (adjust based on diagnosis):

```python
def _check_grounding(
    self, answer: str, evidence: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
) -> bool:
    if not evidence:
        return False

    evidence_parts = []
    for item in evidence:
        text = (
            item.get("text")
            or item.get("canonical_text")
            or item.get("embedding_text")
            or item.get("content")
            or ""
        )
        evidence_parts.append(text)

    if doc_context:
        for s in doc_context.get("summaries") or []:
            if s:
                evidence_parts.append(str(s))
        for f in doc_context.get("key_facts") or []:
            if f:
                evidence_parts.append(str(f))
        for kv in doc_context.get("key_values") or []:
            if isinstance(kv, dict):
                evidence_parts.append(" ".join(str(v) for v in kv.values()))
            elif kv:
                evidence_parts.append(str(kv))
        for e in doc_context.get("entities") or []:
            if isinstance(e, dict):
                evidence_parts.append(" ".join(str(v) for v in e.values()))
            elif e:
                evidence_parts.append(str(e))

    evidence_text = " ".join(evidence_parts)
    if not evidence_text.strip():
        logger.warning("[Reasoner] Grounding: evidence contains no text — UNGROUNDED")
        return False

    if len(answer.strip()) < 20:
        return True

    # --- Number gate (relaxed): only enforce when answer is number-heavy ---
    answer_numbers = set(_NUMBER_RE.findall(answer))
    if len(answer_numbers) >= 3:
        evidence_numbers = set(_NUMBER_RE.findall(evidence_text))
        ungrounded_nums = answer_numbers - evidence_numbers
        if len(ungrounded_nums) / len(answer_numbers) > 0.50:
            logger.warning(
                "[Reasoner] Grounding: %d/%d numbers not in evidence — UNGROUNDED",
                len(ungrounded_nums), len(answer_numbers),
            )
            return False

    # --- Word gate (relaxed): denominator floor + short-answer skip ---
    answer_words = set(
        w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', answer)
    )
    evidence_words = set(
        w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', evidence_text)
    )

    if len(answer_words) < 10:
        return True  # trivially grounded if evidence exists

    if answer_words and evidence_words:
        overlap = len(answer_words & evidence_words)
        denom = max(len(answer_words), 20)
        overlap_ratio = overlap / denom
        if overlap_ratio < 0.15 or overlap < 5:
            logger.warning(
                "[Reasoner] Grounding: only %d/%d words (%.0f%%) overlap — UNGROUNDED",
                overlap, len(answer_words), overlap_ratio * 100,
            )
            return False

    return True
```

**Note:** the exact thresholds above (`>= 3 numbers`, `> 0.50 ungrounded`, `< 10 words`, `max(..., 20)` denom) are calibrated to the diagnostic output. If Task 1's diagnostic showed a different pattern, tune to match — the important invariant is that the tests in Step 1 pass.

- [ ] **Step 4: Run unit tests to verify PASS**

Run: `pytest tests/generation/test_grounding_check.py -v`

Expected: All six tests pass.

- [ ] **Step 5: Run broader reasoner tests for non-regression**

Run: `pytest tests/generation/ -v`

Expected: All previously passing tests still pass. Any pre-existing failures unrelated to grounding should remain stable (not newly failing).

- [ ] **Step 6: Delete the diagnostic file**

Run: `rm tests/generation/test_grounding_diagnose.py`

- [ ] **Step 7: Commit**

Run:
```bash
git add src/generation/reasoner.py tests/generation/test_grounding_check.py
git rm tests/generation/test_grounding_diagnose.py
git commit -m "fix(rag): grounding check no longer returns False universally

Relaxed number gate to >=3 numbers with >50% unmatched threshold; word
gate uses denominator floor and skips for short answers. Lock behaviour
with regression fixtures. Grounded signal in logs/UI now reflects reality."
```

---

## Task 3: Add `SparseEncoder` to `AppState` and load it at startup

**Why:** The retriever needs a shared `SparseEncoder` instance. Loading at startup (parallel with the embedding model) avoids first-query latency spike.

**Files:**
- Modify: `src/api/rag_state.py`
- Modify: `src/api/app_lifespan.py`

- [ ] **Step 1: Write a failing startup test**

Create `tests/api/test_app_lifespan_sparse.py`:

```python
"""Verify app_lifespan populates state.sparse_encoder."""
from unittest.mock import patch, MagicMock

from src.api.rag_state import AppState


def test_appstate_has_sparse_encoder_field():
    # Constructor must accept sparse_encoder
    state = AppState(
        embedding_model=None,
        reranker=None,
        qdrant_client=None,
        redis_client=None,
        ollama_client=None,
        rag_system=None,
        sparse_encoder="stub-sentinel",
    )
    assert state.sparse_encoder == "stub-sentinel"


def test_appstate_sparse_encoder_defaults_to_none():
    state = AppState(
        embedding_model=None,
        reranker=None,
        qdrant_client=None,
        redis_client=None,
        ollama_client=None,
        rag_system=None,
    )
    assert state.sparse_encoder is None
```

- [ ] **Step 2: Run tests to verify FAIL**

Run: `pytest tests/api/test_app_lifespan_sparse.py -v`

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'sparse_encoder'`.

- [ ] **Step 3: Add `sparse_encoder` field to `AppState`**

Modify `src/api/rag_state.py`. Locate the `AppState` dataclass (lines ~13-27). Add one field after `vllm_manager`:

```python
@dataclass
class AppState:
    embedding_model: Any
    reranker: Any
    qdrant_client: Any
    redis_client: Any
    ollama_client: Any
    rag_system: Any
    llm_gateway: Any = None
    multi_agent_gateway: Any = None
    graph_augmenter: Any = None
    vllm_manager: Any = None
    sparse_encoder: Any = None  # NEW — src.embedding.sparse.SparseEncoder
    instance_ids: Dict[str, str] = field(default_factory=dict)
    qdrant_index_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    profile_expertise_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
```

Also add one line in `register_instance_ids` (around line 72):

```python
    _assign_instance_id("sparse_encoder", state.sparse_encoder, state.instance_ids)
```

- [ ] **Step 4: Re-run test, verify PASS**

Run: `pytest tests/api/test_app_lifespan_sparse.py -v`

Expected: Both tests PASS.

- [ ] **Step 5: Wire SparseEncoder load in `app_lifespan.py`**

Open `src/api/app_lifespan.py`. Locate the parallel model-loading block (lines ~46-85). Add a third loader:

```python
    # Add below the existing `sparse_encoder = None` variable declaration
    sparse_encoder = None

    def _load_sparse_encoder():
        nonlocal sparse_encoder
        try:
            from src.embedding.sparse import SparseEncoder
            sparse_encoder = SparseEncoder()  # lazy — model loads on first encode()
        except Exception as exc:  # noqa: BLE001
            logger.error("SparseEncoder init failed: %s", exc)

    t_sparse = threading.Thread(target=_load_sparse_encoder, daemon=True)
    t_sparse.start()
    # ... existing joins ...
    t_sparse.join()
```

Place `sparse_encoder = None` alongside the other top-level `= None` declarations (around line 47). Place the `_load_sparse_encoder` function and `t_sparse.start()` alongside the existing `t_embed.start()` and `t_rerank.start()` calls (around lines 68-71). Add `t_sparse.join()` after `t_rerank.join()` (around line 85).

Then, in the `AppState(...)` constructor call near line 224-236, add `sparse_encoder=sparse_encoder` to the kwargs.

- [ ] **Step 6: Run-time verification (no brittle integration test)**

`initialize_app_state` has too many side-effects to unit-test cleanly (Neo4j init, Redis cleanup, intelligence router, Mongo migrations, etc.). Rather than mocking all of them, verify the wire-up at runtime with a one-line check via the running API:

Run (after starting the API in a separate terminal or trusting the existing systemd service): `curl -s http://localhost:8000/api/health/instances 2>/dev/null | python -c "import json,sys; d=json.loads(sys.stdin.read()); print('sparse_encoder' in d.get('instance_ids', {}))"`

Expected: `True`. If no `/api/health/instances` endpoint exists, fall back to checking logs from a recent API startup for the line `Singleton instance IDs: {...sparse_encoder...}`.

If the API is not currently running this session, skip this step — the dataclass test in Step 4 is sufficient to lock the `AppState.sparse_encoder` field, and the first API restart will exercise the load path.

- [ ] **Step 7: Commit**

Run:
```bash
git add src/api/rag_state.py src/api/app_lifespan.py tests/api/test_app_lifespan_sparse.py
git commit -m "feat(rag): SparseEncoder on AppState, loaded in parallel at startup"
```

---

## Task 4: Populate sparse vector in ingestion going forward

**Why:** Without this, every newly ingested chunk lands in Qdrant with an empty sparse slot, and hybrid retrieval silently falls back to dense for new uploads.

**Files:**
- Modify: `src/embedding/pipeline/qdrant_ingestion.py` (line ~295 `sparse_vector=None`)
- Modify: `tests/embedding/` (if a test exercises ingestion; add a new test if none exists)

- [ ] **Step 1: Locate ingestion mock tests**

Run: `grep -rn "ingest_payloads\|sparse_vector=None" tests/ 2>/dev/null | head -10`

Read any matching tests to understand the mock pattern. If there's a test that exercises `ingest_payloads` directly, use its pattern. If not, Step 2 creates a fresh test.

- [ ] **Step 2: Write a failing test for sparse population**

Create (or extend) `tests/embedding/test_ingestion_sparse.py`:

```python
"""Verify ingest_payloads encodes sparse vectors into ChunkRecord.sparse_vector."""
from unittest.mock import MagicMock, patch

from src.embedding.pipeline import qdrant_ingestion


def test_ingest_payloads_populates_sparse_vector():
    raw_payloads = [
        {
            "subscription_id": "sub-1",
            "profile_id": "p-1",
            "document_id": "doc-1",
            "canonical_text": "Invoice INV-42 total $9,000.00",
            "embedding_text": "Invoice INV-42 total $9,000.00",
            "metadata": {"source_file": "inv42.pdf"},
        }
    ]

    fake_sparse_encoder = MagicMock()
    fake_sparse_encoder.encode.return_value = {"indices": [1, 5, 7], "values": [0.3, 0.9, 0.2]}

    fake_vector_store = MagicMock()
    fake_vector_store.upsert_records.return_value = 1

    captured_records = []

    def capture(collection_name, records, batch_size):
        captured_records.extend(records)
        return 1

    fake_vector_store.upsert_records.side_effect = capture

    with patch.object(qdrant_ingestion, "QdrantVectorStore", return_value=fake_vector_store), \
         patch.object(qdrant_ingestion, "_ollama_embed", return_value=[[0.0] * 1024]), \
         patch("src.api.rag_state.get_app_state") as mock_get_state:
        mock_get_state.return_value = MagicMock(sparse_encoder=fake_sparse_encoder)
        qdrant_ingestion.ingest_payloads(raw_payloads)

    assert len(captured_records) == 1
    rec = captured_records[0]
    # sparse_vector must not be None and must be a SparseVector-like object
    assert rec.sparse_vector is not None
    # It should carry the indices/values from the encoder output
    assert hasattr(rec.sparse_vector, "indices")
    assert list(rec.sparse_vector.indices) == [1, 5, 7]
    assert list(rec.sparse_vector.values) == [0.3, 0.9, 0.2]


def test_ingest_payloads_graceful_when_sparse_encoder_missing():
    raw_payloads = [
        {
            "subscription_id": "sub-1",
            "profile_id": "p-1",
            "document_id": "doc-1",
            "canonical_text": "text",
            "metadata": {"source_file": "f.pdf"},
        }
    ]

    fake_vector_store = MagicMock()
    fake_vector_store.upsert_records.return_value = 1
    captured_records = []
    fake_vector_store.upsert_records.side_effect = lambda c, r, batch_size: captured_records.extend(r) or 1

    with patch.object(qdrant_ingestion, "QdrantVectorStore", return_value=fake_vector_store), \
         patch.object(qdrant_ingestion, "_ollama_embed", return_value=[[0.0] * 1024]), \
         patch("src.api.rag_state.get_app_state") as mock_get_state:
        mock_get_state.return_value = MagicMock(sparse_encoder=None)
        qdrant_ingestion.ingest_payloads(raw_payloads)

    # sparse_vector must be None when encoder is unavailable
    assert captured_records[0].sparse_vector is None
```

- [ ] **Step 3: Run the tests to verify FAIL**

Run: `pytest tests/embedding/test_ingestion_sparse.py -v`

Expected: First test FAILS because `sparse_vector` is still None.

- [ ] **Step 4: Modify `qdrant_ingestion.py`**

Open `src/embedding/pipeline/qdrant_ingestion.py`. At the top, add the import:

```python
from src.embedding.sparse import sparse_to_qdrant
```

Locate the loop at lines ~272-298 that builds `ChunkRecord`s. Before the loop, fetch the sparse encoder once:

```python
        # Fetch sparse encoder from AppState (None if API not running, e.g. tests)
        try:
            from src.api.rag_state import get_app_state
            _app_state = get_app_state()
            _sparse_encoder = _app_state.sparse_encoder if _app_state else None
        except Exception:
            _sparse_encoder = None
```

Place this after the `contents = [...]` / `vectors = _ollama_embed(contents)` block and before the `records: List[ChunkRecord] = []` line (around line 270).

Then, inside the loop, replace `sparse_vector=None` with:

```python
            sparse_vector_obj = None
            if _sparse_encoder is not None:
                try:
                    _text_for_sparse = payload.get("canonical_text") or payload.get("content") or ""
                    if _text_for_sparse:
                        sparse_dict = _sparse_encoder.encode(_text_for_sparse)
                        sparse_vector_obj = sparse_to_qdrant(sparse_dict)
                except Exception as exc:
                    logger.warning("Sparse encode failed for chunk_id=%s: %s", chunk_id, exc)

            records.append(
                ChunkRecord(
                    chunk_id=str(chunk_id),
                    dense_vector=vector,
                    sparse_vector=sparse_vector_obj,
                    payload=payload,
                )
            )
```

- [ ] **Step 5: Verify `ChunkRecord.sparse_vector` flows through to Qdrant**

Run: `grep -n "sparse_vector" /home/ubuntu/PycharmProjects/DocWain/src/api/vector_store.py | head -20`

Expected: `upsert_records` or equivalent should pass the sparse vector through. If it doesn't, fix it — the `keywords_vector` sparse slot must be populated. Open `src/api/vector_store.py` and locate the `upsert_records` method; ensure it reads `record.sparse_vector` and includes it in the `PointStruct`'s `vector` dict as `{"keywords_vector": record.sparse_vector}` when non-None.

- [ ] **Step 6: Re-run tests**

Run: `pytest tests/embedding/test_ingestion_sparse.py -v`

Expected: Both tests PASS.

- [ ] **Step 7: Run adjacent tests for non-regression**

Run: `pytest tests/embedding/ -v`

Expected: No newly failing tests.

- [ ] **Step 8: Commit**

Run:
```bash
git add src/embedding/pipeline/qdrant_ingestion.py src/api/vector_store.py tests/embedding/test_ingestion_sparse.py
git commit -m "feat(rag): ingestion populates Qdrant sparse slot via SparseEncoder

Forward-looking only — every newly ingested chunk now carries a SPLADE
sparse vector alongside its dense vector. Backfill for existing chunks
handled by scripts/backfill_sparse_vectors.py in Task 5-7."
```

---

## Task 5: Backfill script — skeleton, argparse, inventory mode

**Why:** Start with the read-only path to verify the job can see all collections and estimate runtime before writing anything.

**Files:**
- Create: `scripts/backfill_sparse_vectors.py`
- Create: `tests/scripts/test_backfill_sparse.py`

- [ ] **Step 1: Write failing tests for inventory mode**

Create `tests/scripts/test_backfill_sparse.py`:

```python
"""Lock the behaviour of scripts/backfill_sparse_vectors.py."""
import json
from unittest.mock import MagicMock

import pytest

from scripts import backfill_sparse_vectors as bf


def test_inventory_mode_lists_collections_and_counts():
    fake_client = MagicMock()
    fake_client.get_collections.return_value.collections = [
        MagicMock(name="c1"),
        MagicMock(name="c2"),
    ]
    fake_client.get_collections.return_value.collections[0].name = "sub_aaa"
    fake_client.get_collections.return_value.collections[1].name = "sub_bbb"
    fake_client.count.side_effect = [
        MagicMock(count=1000),
        MagicMock(count=42),
    ]

    inventory = bf.inventory_collections(fake_client)

    assert inventory == [
        {"collection": "sub_aaa", "point_count": 1000},
        {"collection": "sub_bbb", "point_count": 42},
    ]


def test_inventory_mode_filters_by_subscription():
    fake_client = MagicMock()
    fake_client.get_collections.return_value.collections = [
        MagicMock(name="c1"),
        MagicMock(name="c2"),
    ]
    fake_client.get_collections.return_value.collections[0].name = "sub_aaa"
    fake_client.get_collections.return_value.collections[1].name = "sub_bbb"
    fake_client.count.return_value = MagicMock(count=1000)

    inventory = bf.inventory_collections(fake_client, subscription_id="aaa")

    assert len(inventory) == 1
    assert inventory[0]["collection"] == "sub_aaa"
```

- [ ] **Step 2: Verify tests FAIL**

Run: `pytest tests/scripts/test_backfill_sparse.py -v`

Expected: `ModuleNotFoundError: No module named 'scripts.backfill_sparse_vectors'` — the script doesn't exist yet.

- [ ] **Step 3: Create skeleton + inventory mode**

Create `scripts/backfill_sparse_vectors.py`:

```python
#!/usr/bin/env python3
"""One-shot backfill of SPLADE sparse vectors into existing Qdrant chunks.

Qdrant collections already have a sparse slot named `keywords_vector`
provisioned (see src/api/vector_store.py). Existing chunks were indexed
with sparse_vector=None. This script iterates every chunk, encodes its
canonical_text with src.embedding.sparse.SparseEncoder, and upserts the
sparse vector while leaving the dense vector and payload untouched.

Usage:
    # Size check, no writes
    python scripts/backfill_sparse_vectors.py --inventory-only
    python scripts/backfill_sparse_vectors.py --inventory-only --subscription-id X

    # Dry run (encode but don't upsert) — verifies GPU + Qdrant paths
    python scripts/backfill_sparse_vectors.py --dry-run --subscription-id X

    # Full run — encode + upsert, resumable
    python scripts/backfill_sparse_vectors.py --subscription-id X
    python scripts/backfill_sparse_vectors.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from qdrant_client import QdrantClient

from src.api.config import Config
from src.embedding.sparse import SparseEncoder, sparse_to_qdrant

logger = logging.getLogger("backfill_sparse")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_client() -> QdrantClient:
    return QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)


def inventory_collections(
    client: QdrantClient, subscription_id: Optional[str] = None
) -> list[dict]:
    """Return a list of {collection, point_count} for every (or one) collection."""
    out = []
    for col in client.get_collections().collections or []:
        name = getattr(col, "name", str(col))
        if subscription_id and subscription_id not in name:
            continue
        try:
            count_resp = client.count(collection_name=name, exact=True)
            out.append({"collection": name, "point_count": count_resp.count})
        except Exception as exc:
            logger.warning("Count failed for %s: %s", name, exc)
            out.append({"collection": name, "point_count": -1, "error": str(exc)})
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subscription-id", help="Single subscription to process (collection name substring)")
    ap.add_argument("--all", action="store_true", help="Process every collection the client can see")
    ap.add_argument("--batch-size", type=int, default=64, help="SPLADE batch size")
    ap.add_argument("--inventory-only", action="store_true", help="List collections and point counts, then exit")
    ap.add_argument("--dry-run", action="store_true", help="Encode but do not upsert")
    ap.add_argument("--concurrency", type=int, default=1, help="Not yet used; fixed to 1 while vLLM is resident")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = parse_args(argv)

    if not (args.subscription_id or args.all or args.inventory_only):
        logger.error("Must specify --subscription-id, --all, or --inventory-only")
        return 2

    client = make_client()

    if args.inventory_only:
        inventory = inventory_collections(client, subscription_id=args.subscription_id)
        print(f"{'COLLECTION':<40} {'POINTS':>12}")
        print("-" * 54)
        for row in inventory:
            print(f"{row['collection']:<40} {row['point_count']:>12}")
        total = sum(r["point_count"] for r in inventory if r["point_count"] > 0)
        print("-" * 54)
        print(f"{'TOTAL':<40} {total:>12}")
        return 0

    # Full processing path (implemented in Task 7)
    logger.error("Full processing not yet implemented. Use --inventory-only for now.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
```

Also create the test directory if needed: `mkdir -p tests/scripts && touch tests/scripts/__init__.py`.

- [ ] **Step 4: Verify tests PASS**

Run: `pytest tests/scripts/test_backfill_sparse.py -v`

Expected: Both tests PASS.

- [ ] **Step 5: Run the inventory against the live Qdrant**

Run: `python scripts/backfill_sparse_vectors.py --inventory-only | tee /tmp/backfill_inventory.log`

Expected: A table of all collections with point counts. Compare the TOTAL to what you expect from production traffic. If a collection has a count that seems wrong (0, or a huge order-of-magnitude surprise), investigate before proceeding to the full backfill.

- [ ] **Step 6: Commit**

Run:
```bash
git add scripts/backfill_sparse_vectors.py tests/scripts/test_backfill_sparse.py tests/scripts/__init__.py
git commit -m "feat(rag): backfill script skeleton with inventory mode

Lists every Qdrant collection and its point count so we can size the
SPLADE backfill outage window before running it."
```

---

## Task 6: Backfill script — dry-run with SPLADE encoding

**Why:** Verify SPLADE + Qdrant scroll paths work end-to-end without mutating anything. Catches GPU contention, model-load failures, and Qdrant API issues before we touch real data.

**Files:**
- Modify: `scripts/backfill_sparse_vectors.py`
- Modify: `tests/scripts/test_backfill_sparse.py`

- [ ] **Step 1: Write failing tests for dry-run**

Append to `tests/scripts/test_backfill_sparse.py`:

```python
def test_process_collection_dry_run_encodes_but_doesnt_upsert():
    fake_client = MagicMock()
    # 2 points in the collection
    fake_client.scroll.return_value = (
        [
            MagicMock(id="p1", payload={"canonical_text": "invoice total $1000"}),
            MagicMock(id="p2", payload={"canonical_text": "contract agreement"}),
        ],
        None,  # next_offset == None means end
    )

    fake_encoder = MagicMock()
    fake_encoder.encode.side_effect = [
        {"indices": [1, 2], "values": [0.5, 0.7]},
        {"indices": [3, 4], "values": [0.8, 0.2]},
    ]

    stats = bf.process_collection(
        fake_client, "sub_aaa", fake_encoder, batch_size=64, dry_run=True
    )

    assert stats["encoded"] == 2
    assert stats["upserted"] == 0
    # No writes
    fake_client.upsert.assert_not_called()
    fake_client.set_payload.assert_not_called()
```

- [ ] **Step 2: Verify test FAILS**

Run: `pytest tests/scripts/test_backfill_sparse.py::test_process_collection_dry_run_encodes_but_doesnt_upsert -v`

Expected: `AttributeError: module has no attribute 'process_collection'`.

- [ ] **Step 3: Implement `process_collection`**

Modify `scripts/backfill_sparse_vectors.py`. Add a new function above `main`:

```python
def _iter_batches(
    client: QdrantClient,
    collection_name: str,
    batch_size: int,
) -> Iterator[list]:
    """Yield successive batches of Qdrant points (resumable via payload marker)."""
    from qdrant_client.models import Filter, FieldCondition, IsNullCondition, PayloadField

    next_offset = None
    while True:
        # Filter: only points WITHOUT sparse_backfilled_at (resumable)
        # IsNullCondition matches points where the field is missing or null.
        scroll_filter = Filter(
            must=[
                IsNullCondition(is_null=PayloadField(key="sparse_backfilled_at"))
            ]
        )
        try:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
        except Exception as exc:
            logger.error("Scroll failed for %s: %s", collection_name, exc)
            return

        if not points:
            return

        yield points

        if next_offset is None:
            return


def process_collection(
    client: QdrantClient,
    collection_name: str,
    encoder: SparseEncoder,
    batch_size: int,
    *,
    dry_run: bool = False,
) -> dict:
    """Encode and (optionally) upsert sparse vectors for all un-backfilled chunks."""
    stats = {"encoded": 0, "upserted": 0, "skipped": 0, "errors": 0}

    for batch in _iter_batches(client, collection_name, batch_size):
        texts: list[str] = []
        point_ids: list = []

        for pt in batch:
            payload = pt.payload or {}
            text = payload.get("canonical_text") or payload.get("embedding_text") or payload.get("text") or ""
            if not text:
                stats["skipped"] += 1
                continue
            texts.append(text)
            point_ids.append(pt.id)

        if not texts:
            continue

        try:
            sparse_dicts = encoder.encode_batch(texts)
            stats["encoded"] += len(sparse_dicts)
        except Exception as exc:
            logger.error("Encode batch failed for %s: %s", collection_name, exc)
            stats["errors"] += len(texts)
            continue

        if dry_run:
            continue

        # Upsert per-point (named vector update + payload merge)
        from qdrant_client.models import PointVectors

        upsert_points = [
            PointVectors(id=pid, vector={"keywords_vector": sparse_to_qdrant(sd)})
            for pid, sd in zip(point_ids, sparse_dicts)
        ]
        try:
            client.update_vectors(collection_name=collection_name, points=upsert_points)
            # Mark as backfilled via payload
            client.set_payload(
                collection_name=collection_name,
                payload={"sparse_backfilled_at": _iso_now()},
                points=point_ids,
            )
            stats["upserted"] += len(upsert_points)
        except Exception as exc:
            logger.error("Upsert failed for %s: %s", collection_name, exc)
            stats["errors"] += len(upsert_points)

        logger.info(
            "collection=%s processed=%d upserted=%d errors=%d",
            collection_name, stats["encoded"], stats["upserted"], stats["errors"],
        )

    return stats
```

Then wire `process_collection` into `main`:

```python
def main(argv: Optional[List[str]] = None) -> int:
    # ... existing setup ...

    if args.inventory_only:
        # ... existing inventory path ...
        return 0

    # Resolve collections to process
    inventory = inventory_collections(client, subscription_id=args.subscription_id)
    if not inventory:
        logger.error("No collections matched.")
        return 1

    encoder = SparseEncoder()
    grand_total = {"encoded": 0, "upserted": 0, "skipped": 0, "errors": 0}

    for row in inventory:
        name = row["collection"]
        logger.info("=== Processing %s (%d points) ===", name, row["point_count"])
        t0 = time.time()
        stats = process_collection(
            client, name, encoder, batch_size=args.batch_size, dry_run=args.dry_run,
        )
        elapsed = time.time() - t0
        logger.info("%s done in %.1fs: %s", name, elapsed, stats)
        for k in grand_total:
            grand_total[k] += stats.get(k, 0)

    logger.info("=== GRAND TOTAL === %s", grand_total)
    return 0 if grand_total["errors"] == 0 else 1
```

- [ ] **Step 4: Verify test PASS**

Run: `pytest tests/scripts/test_backfill_sparse.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Dry-run against a single collection**

Pick the smallest real collection from the earlier inventory (say `sub_aaa`):

Run: `python scripts/backfill_sparse_vectors.py --dry-run --subscription-id aaa --batch-size 16`

Expected: Logs show N chunks encoded, 0 upserted. No errors. Typical rate 30-60 chunks/sec on A100 with vLLM resident.

- [ ] **Step 6: Watch GPU pressure**

In another terminal during the dry-run:
Run: `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 2`

Expected: memory.used climbs by ~1-2GB (SPLADE model + batch); utilization bumps. Memory should NOT push past ~79GB on the 80GB A100. If it does, drop `--batch-size 8` and repeat.

- [ ] **Step 7: Commit**

Run:
```bash
git add scripts/backfill_sparse_vectors.py tests/scripts/test_backfill_sparse.py
git commit -m "feat(rag): backfill script dry-run + full processing logic

Dry-run verifies SPLADE encoding path end-to-end without mutating Qdrant.
Full path (--subscription-id X / --all) upserts the sparse vector and
marks each processed point with sparse_backfilled_at for resumability."
```

---

## Task 7: Run the full backfill

**Why:** Populates the sparse slot in all existing chunks. This is the single biggest lever for context recall.

**Files:** No code changes; this is the operational step.

- [ ] **Step 1: Verify vLLM is healthy before starting**

Run: `curl -s http://localhost:8100/health && echo && nvidia-smi --query-gpu=memory.used --format=csv,noheader`

Expected: HTTP 200 from vLLM; memory ~74 GiB used (model resident).

- [ ] **Step 2: Run the backfill for a single collection first (smoke test)**

Pick the smallest real collection:

Run: `python scripts/backfill_sparse_vectors.py --subscription-id <smallest> --batch-size 16 2>&1 | tee /tmp/backfill_first.log`

Expected: Logs show `encoded == upserted == point_count` for that collection. `errors == 0`. Completes in minutes.

- [ ] **Step 3: Spot-check that vLLM is still serving**

Run: `curl -s -X POST http://localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"docwain-fast","messages":[{"role":"user","content":"hello"}],"max_tokens":10}' | head -c 500`

Expected: A completion response. If vLLM is not responding, stop the backfill and diagnose.

- [ ] **Step 4: Run the backfill for all remaining collections**

Run: `python scripts/backfill_sparse_vectors.py --all --batch-size 16 2>&1 | tee /tmp/backfill_all.log`

Expected: Grand total matches total points across all collections. Runtime proportional to point count (~30-60 chunks/sec).

- [ ] **Step 5: Verify backfill by sampling**

Run:
```python
python -c "
from qdrant_client import QdrantClient
from src.api.config import Config
c = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
for col in c.get_collections().collections:
    name = col.name
    sample = c.scroll(collection_name=name, limit=3, with_payload=True, with_vectors=True)[0]
    for pt in sample:
        has_sparse = pt.vector and 'keywords_vector' in (pt.vector if isinstance(pt.vector, dict) else {})
        marker = (pt.payload or {}).get('sparse_backfilled_at')
        print(f'{name}: id={pt.id} has_sparse={has_sparse} marker={marker}')
"
```

Expected: Every sample shows `has_sparse=True` and `marker` is a recent ISO timestamp.

- [ ] **Step 6: Commit the operational log**

Run:
```bash
git add -f /tmp/backfill_all.log  # only if you want an audit trail; otherwise skip
```

Or simply record the final stats line from the log in the commit message for the next task.

---

## Task 8: Hybrid dense+sparse retrieval in `UnifiedRetriever`

**Why:** With sparse populated, wire hybrid retrieval into the live path. Dense + sparse merged via RRF.

**Files:**
- Modify: `src/retrieval/retriever.py`
- Modify: `src/agent/core_agent.py` (constructor of retriever — pass `sparse_encoder`)
- Create: `tests/retrieval/test_hybrid_retrieve.py`

- [ ] **Step 1: Write failing tests for hybrid retrieval**

Create `tests/retrieval/test_hybrid_retrieve.py`:

```python
"""Lock dense+sparse+RRF hybrid retrieval behaviour."""
from unittest.mock import MagicMock

from src.retrieval.retriever import UnifiedRetriever, EvidenceChunk


def _make_point(chunk_id: str, text: str, score: float):
    pt = MagicMock()
    pt.payload = {
        "canonical_text": text,
        "chunk": {"id": chunk_id, "type": "text"},
        "document_id": f"doc-{chunk_id[:2]}",
        "profile_id": "p-1",
        "source_name": f"{chunk_id}.pdf",
        "subscription_id": "sub-1",
    }
    pt.score = score
    return pt


def test_hybrid_retrieve_uses_sparse_when_encoder_present():
    """When sparse_encoder is provided, _sparse_search is called per profile."""
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value.points = [
        _make_point("cd1", "dense result 1", 0.9),
        _make_point("cd2", "dense result 2", 0.8),
    ]
    fake_qdrant.scroll.return_value = ([], None)  # no fill-missing

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    fake_sparse = MagicMock()
    fake_sparse.encode.return_value = {"indices": [1, 2], "values": [0.5, 0.7]}

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=fake_sparse,
    )
    result = retriever.retrieve("test query", "sub-1", ["p-1"], top_k=10)

    # Sparse encoder was called at least once
    assert fake_sparse.encode.called
    # query_points was called twice per profile: once dense, once sparse
    calls = fake_qdrant.query_points.call_args_list
    # Must include at least one call with using="content_vector" and one with using="keywords_vector"
    usings = [kw.get("using") for _, kw in calls]
    assert "content_vector" in usings
    assert "keywords_vector" in usings


def test_hybrid_retrieve_degrades_to_dense_when_sparse_encoder_none():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value.points = [
        _make_point("cd1", "dense", 0.9),
    ]
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=None,
    )
    result = retriever.retrieve("test", "sub-1", ["p-1"], top_k=10)

    # Only dense search was called
    calls = fake_qdrant.query_points.call_args_list
    usings = [kw.get("using") for _, kw in calls]
    assert "content_vector" in usings
    assert "keywords_vector" not in usings
    assert len(result.chunks) >= 0  # smoke


def test_hybrid_retrieve_degrades_when_sparse_search_raises():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    call_count = {"n": 0}

    def query_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if kwargs.get("using") == "keywords_vector":
            raise RuntimeError("sparse server error")
        res = MagicMock()
        res.points = [_make_point("cd1", "dense", 0.9)]
        return res

    fake_qdrant.query_points.side_effect = query_side_effect
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]
    fake_sparse = MagicMock()
    fake_sparse.encode.return_value = {"indices": [1], "values": [0.5]}

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=fake_sparse,
    )
    # Must not raise — sparse failure is a warning, not an error
    result = retriever.retrieve("test", "sub-1", ["p-1"], top_k=10)
    # Dense result should still land
    assert len(result.chunks) >= 1
```

- [ ] **Step 2: Verify tests FAIL**

Run: `pytest tests/retrieval/test_hybrid_retrieve.py -v`

Expected: `TypeError: __init__() got an unexpected keyword argument 'sparse_encoder'`.

- [ ] **Step 3: Add `sparse_encoder` to `UnifiedRetriever.__init__`**

Modify `src/retrieval/retriever.py`:

```python
    def __init__(self, qdrant_client, embedder, sparse_encoder=None, graph_augmenter=None):
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.graph_augmenter = graph_augmenter  # used in Task 9
        self._collection_exists_cache: dict[str, tuple[bool, float]] = {}
```

- [ ] **Step 4: Add `_sparse_search` method and wire it into `_search_profile`**

Add this method below `_search_profile`:

```python
    def _sparse_search(
        self,
        collection_name: str,
        query: str,
        subscription_id: str,
        profile_id: str,
        *,
        document_ids: Optional[List[str]] = None,
        top_k: int = 30,
    ) -> List[EvidenceChunk]:
        """SPLADE sparse search against Qdrant's keywords_vector named sparse slot."""
        if self.sparse_encoder is None:
            return []
        try:
            from qdrant_client.models import SparseVector
            sparse_dict = self.sparse_encoder.encode(query)
            sparse_query = SparseVector(
                indices=sparse_dict["indices"],
                values=sparse_dict["values"],
            )
        except Exception as exc:
            logger.warning("Sparse encode failed for query: %s", exc)
            return []

        qfilter = self._build_filter(subscription_id, profile_id, document_ids)

        try:
            result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=sparse_query,
                using="keywords_vector",
                query_filter=qfilter,
                limit=top_k,
                with_payload=True,
            )
            points = result.points if hasattr(result, "points") else []
        except Exception as exc:
            logger.warning("Sparse search failed collection=%s profile=%s: %s",
                           collection_name, profile_id, exc)
            return []

        points = [
            pt for pt in points
            if (pt.payload or {}).get("resolution", "chunk") not in ("doc_index", "doc_intelligence")
        ]
        return [self._point_to_chunk(pt, profile_id) for pt in points]
```

Now modify `_search_profile` to call sparse in parallel with dense and RRF-merge:

```python
    def _search_profile(
        self,
        collection_name: str,
        query: str,
        query_vector: list,
        subscription_id: str,
        profile_id: str,
        *,
        document_ids: Optional[List[str]] = None,
        top_k: int = 30,
        correlation_id: Optional[str] = None,
    ) -> List[EvidenceChunk]:
        """Dense + sparse hybrid search for a single profile, with RRF fusion."""
        qfilter = self._build_filter(subscription_id, profile_id, document_ids)

        # Dense (existing)
        try:
            result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="content_vector",
                query_filter=qfilter,
                limit=top_k,
                with_payload=True,
            )
            dense_points = result.points if hasattr(result, "points") else []
        except Exception:
            logger.exception(
                "Dense search failed collection=%s profile=%s cid=%s",
                collection_name, profile_id, correlation_id,
            )
            dense_points = []

        dense_points = [
            pt for pt in dense_points
            if (pt.payload or {}).get("resolution", "chunk") not in ("doc_index", "doc_intelligence")
        ]
        dense_chunks = [self._point_to_chunk(pt, profile_id) for pt in dense_points]

        # Sparse (new — empty list when encoder is None or fails)
        sparse_chunks = self._sparse_search(
            collection_name, query, subscription_id, profile_id,
            document_ids=document_ids, top_k=top_k,
        )

        # RRF fusion
        chunks = self._rrf_merge(dense_chunks, sparse_chunks, top_k=top_k)

        # Keyword fallback: only if hybrid candidates are still sparse
        high_quality = [c for c in chunks if c.score >= self._HIGH_QUALITY_THRESHOLD]
        if len(high_quality) < self._DENSE_MIN:
            fallback = self._keyword_fallback(
                collection_name, query, qfilter, top_k, existing_ids={c.chunk_id for c in chunks},
            )
            chunks.extend(fallback)

        return chunks
```

Add the RRF merge helper below `_sparse_search`:

```python
    @staticmethod
    def _rrf_merge(
        dense: List[EvidenceChunk],
        sparse: List[EvidenceChunk],
        top_k: int = 30,
        k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[EvidenceChunk]:
        """Reciprocal Rank Fusion of dense + sparse chunk lists."""
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, EvidenceChunk] = {}

        for rank, c in enumerate(dense):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + dense_weight / (k + rank + 1)
            chunks_by_id[c.chunk_id] = c

        for rank, c in enumerate(sparse):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + sparse_weight / (k + rank + 1)
            # Prefer dense chunk when both present (has richer score/metadata)
            chunks_by_id.setdefault(c.chunk_id, c)

        fused_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        merged: List[EvidenceChunk] = []
        for cid in fused_ids[:top_k]:
            c = chunks_by_id[cid]
            # Overwrite score with the fused RRF score so downstream rerank has
            # a consistent signal ordering
            c.score = scores[cid]
            merged.append(c)
        return merged
```

- [ ] **Step 5: Verify tests PASS**

Run: `pytest tests/retrieval/test_hybrid_retrieve.py -v`

Expected: All three tests PASS.

- [ ] **Step 6: Wire `sparse_encoder` through `core_agent.py`**

Modify `src/agent/core_agent.py`. Locate line ~148 where `UnifiedRetriever` is constructed:

```python
        # Before
        self._retriever = UnifiedRetriever(qdrant_client=qdrant_client, embedder=embedder)

        # After
        from src.api.rag_state import get_app_state
        _state = get_app_state()
        self._retriever = UnifiedRetriever(
            qdrant_client=qdrant_client,
            embedder=embedder,
            sparse_encoder=_state.sparse_encoder if _state else None,
            graph_augmenter=_state.graph_augmenter if _state else None,
        )
```

- [ ] **Step 7: Run the agent test suite for non-regression**

Run: `pytest tests/agent/ tests/retrieval/ -v`

Expected: Previously passing tests still pass. Any failures should be in the new hybrid test file only (which should pass) or in tests that were already flaky — verify the diff, not the pass count.

- [ ] **Step 8: Run a live smoke query**

Run:
```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is the invoice total?","profile_id":"67fde0754e36c00b14cea7f5_test","subscription_id":"67fde0754e36c00b14cea7f5","user_id":"plan@docwain.ai"}' \
  | head -c 500
```

Expected: A normal response. Watch the logs — you should now see sparse queries in Qdrant's access pattern.

- [ ] **Step 9: Re-run RAGAS**

Run: `python scripts/intensive_test.py`

Then: `cp tests/ragas_metrics.json tests/ragas_metrics.post_hybrid.json`

Compare the aggregate block against `tests/ragas_metrics.baseline.json`. Expected: `context_recall` moves up noticeably (towards 0.75). `answer_faithfulness` may also nudge up. If recall does NOT move, STOP and diagnose before Task 9 — the sparse search may not be hitting populated slots, or the backfill missed a collection.

- [ ] **Step 10: Commit**

Run:
```bash
git add src/retrieval/retriever.py src/agent/core_agent.py tests/retrieval/test_hybrid_retrieve.py tests/ragas_metrics.json tests/ragas_metrics.post_hybrid.json
git commit -m "feat(rag): hybrid dense+sparse retrieval with RRF fusion

UnifiedRetriever now queries both content_vector (BGE) and
keywords_vector (SPLADE) slots in Qdrant and merges via reciprocal rank
fusion (dense 0.6 / sparse 0.4). Degrades to dense-only when
SparseEncoder is absent or sparse search raises. core_agent.py injects
the encoder from AppState at construction."
```

---

## Task 9: KG entity expansion via `GraphAugmenter`

**Why:** Catches cross-document cases where the answer lives in a different doc than the one the query text matches on.

**Files:**
- Modify: `src/retrieval/retriever.py` (add KG expansion step)
- Create: `tests/retrieval/test_kg_expansion.py`

- [ ] **Step 1: Write failing tests for KG expansion**

Create `tests/retrieval/test_kg_expansion.py`:

```python
"""Lock KG entity expansion behaviour."""
from unittest.mock import MagicMock

from src.retrieval.retriever import UnifiedRetriever, EvidenceChunk


def test_kg_expansion_adds_chunks_when_augmenter_returns_chunk_ids():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    # Dense returns one chunk
    dense_pt = MagicMock()
    dense_pt.payload = {
        "canonical_text": "dense text",
        "chunk": {"id": "cd1", "type": "text"},
        "document_id": "doc-1",
        "profile_id": "p-1",
        "source_name": "a.pdf",
    }
    dense_pt.score = 0.9
    fake_qdrant.query_points.return_value.points = [dense_pt]
    fake_qdrant.scroll.return_value = ([], None)

    # GraphAugmenter returns evidence_chunk_ids from a different doc
    from src.kg.retrieval import GraphHints, GraphSnippet
    hints = GraphHints(
        evidence_chunk_ids=["kg1", "kg2"],
        doc_ids=["doc-2"],
        graph_snippets=[
            GraphSnippet(text="kg chunk text 1", doc_id="doc-2", doc_name="b.pdf", chunk_id="kg1", relation="MENTIONS"),
            GraphSnippet(text="kg chunk text 2", doc_id="doc-2", doc_name="b.pdf", chunk_id="kg2", relation="MENTIONS"),
        ],
    )
    fake_augmenter = MagicMock()
    fake_augmenter.augment.return_value = hints

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=None,
        graph_augmenter=fake_augmenter,
    )
    result = retriever.retrieve("find related entity", "sub-1", ["p-1"], top_k=10)

    # kg1 and kg2 chunk IDs must appear in the result
    ids = {c.chunk_id for c in result.chunks}
    assert "kg1" in ids
    assert "kg2" in ids


def test_kg_expansion_silent_when_augmenter_none():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value.points = []
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=None,
        graph_augmenter=None,
    )
    # Must not raise
    result = retriever.retrieve("query", "sub-1", ["p-1"], top_k=10)
    assert result is not None


def test_kg_expansion_silent_when_augmenter_raises():
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value.points = []
    fake_qdrant.scroll.return_value = ([], None)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = [[0.1] * 1024]
    fake_augmenter = MagicMock()
    fake_augmenter.augment.side_effect = RuntimeError("neo4j down")

    retriever = UnifiedRetriever(
        qdrant_client=fake_qdrant,
        embedder=fake_embedder,
        sparse_encoder=None,
        graph_augmenter=fake_augmenter,
    )
    result = retriever.retrieve("query", "sub-1", ["p-1"], top_k=10)
    # Must degrade gracefully
    assert result is not None
```

- [ ] **Step 2: Verify tests FAIL**

Run: `pytest tests/retrieval/test_kg_expansion.py -v`

Expected: First test FAILS because `kg1`/`kg2` aren't in the result (KG expansion not yet implemented).

- [ ] **Step 3: Implement `_kg_expand` and wire into `retrieve`**

Modify `src/retrieval/retriever.py`. Add this method above `_build_filter`:

```python
    def _kg_expand(
        self,
        query: str,
        subscription_id: str,
        profile_ids: List[str],
    ) -> List[EvidenceChunk]:
        """Use GraphAugmenter to pull 1-hop KG chunks for entities in the query."""
        if self.graph_augmenter is None:
            return []

        all_chunks: List[EvidenceChunk] = []
        for pid in profile_ids:
            try:
                hints = self.graph_augmenter.augment(query, subscription_id, pid)
            except Exception as exc:
                logger.warning("KG augment failed for profile=%s: %s", pid, exc)
                continue

            # Materialise graph_snippets as EvidenceChunks (one per snippet)
            for snip in hints.graph_snippets:
                all_chunks.append(EvidenceChunk(
                    text=snip.text or "",
                    source_name=snip.doc_name or snip.doc_id or "",
                    document_id=snip.doc_id or "",
                    profile_id=pid,
                    section=snip.relation or "",
                    page_start=0,
                    page_end=0,
                    score=0.4,  # KG-sourced — below dense high-quality threshold but non-trivial
                    chunk_id=snip.chunk_id,
                    chunk_type="kg",
                ))

        return all_chunks
```

Then modify `retrieve` to add KG chunks after the per-profile loop (around line 125) and before `_fill_missing_documents`:

```python
        # After: all_chunks collected from _search_profile calls
        kg_chunks = self._kg_expand(query, subscription_id, profile_ids)
        # Merge, preferring existing (dense/sparse) chunks over KG for duplicates
        existing_ids = {c.chunk_id for c in all_chunks}
        for kc in kg_chunks:
            if kc.chunk_id not in existing_ids:
                all_chunks.append(kc)
                existing_ids.add(kc.chunk_id)
```

- [ ] **Step 4: Verify tests PASS**

Run: `pytest tests/retrieval/test_kg_expansion.py -v`

Expected: All three tests PASS.

- [ ] **Step 5: Run broader retrieval tests**

Run: `pytest tests/retrieval/ -v`

Expected: No newly failing tests.

- [ ] **Step 6: Re-run RAGAS**

Run: `python scripts/intensive_test.py`

Then: `cp tests/ragas_metrics.json tests/ragas_metrics.post_kg.json`

Compare aggregate against `tests/ragas_metrics.post_hybrid.json`. Expected: `context_recall` may nudge up further; cross-document queries (HR-level comparisons, etc.) should improve.

- [ ] **Step 7: Commit**

Run:
```bash
git add src/retrieval/retriever.py tests/retrieval/test_kg_expansion.py tests/ragas_metrics.json tests/ragas_metrics.post_kg.json
git commit -m "feat(rag): KG entity expansion via GraphAugmenter

UnifiedRetriever calls GraphAugmenter.augment() per profile; the
returned graph_snippets become low-score EvidenceChunks that join the
candidate pool. Neo4j failure or absent augmenter is a warning, not an
error."
```

---

## Task 10: Delete dead code

**Why:** The unused 3-layer `unified_retriever.py` and its test file mislead future readers.

**Files:**
- Delete: `src/retrieval/unified_retriever.py`
- Delete: `tests/test_retrieval_integration.py`

- [ ] **Step 1: Confirm no imports outside the file itself and its test**

Run: `grep -rn "from src.retrieval.unified_retriever\|import unified_retriever" /home/ubuntu/PycharmProjects/DocWain/src/ /home/ubuntu/PycharmProjects/DocWain/tests/ /home/ubuntu/PycharmProjects/DocWain/scripts/ 2>/dev/null | grep -v __pycache__`

Expected: Only matches in `tests/test_retrieval_integration.py`. If anything else matches, STOP and investigate — delete is unsafe until those references are removed.

- [ ] **Step 2: Delete both files**

Run:
```bash
git rm src/retrieval/unified_retriever.py
git rm tests/test_retrieval_integration.py
```

- [ ] **Step 3: Run the full test suite for non-regression**

Run: `pytest tests/ -x --ignore=tests/teams_app --ignore=tests/standalone -q 2>&1 | tail -50`

Expected: No test collection errors. Any failing tests should be pre-existing flakes (compare against the baseline run's pass count before any code change in Task 0).

- [ ] **Step 4: Commit**

Run:
```bash
git commit -m "chore(rag): delete unused 3-layer retriever and its test file

src/retrieval/unified_retriever.py was never imported by live code.
tests/test_retrieval_integration.py was its only importer and exercised
no live path. Both deleted to prevent future reader confusion."
```

---

## Task 11: Final gate check and retrospective

**Why:** Prove the workstream succeeded or capture concrete diagnostics for the next workstream.

**Files:**
- Create: `docs/superpowers/retrospectives/2026-04-17-rag-accuracy.md`

- [ ] **Step 1: Run the final RAGAS**

Run: `python scripts/intensive_test.py`

- [ ] **Step 2: Compare baseline vs final**

Run:
```bash
python -c "
import json
base = json.load(open('tests/ragas_metrics.baseline.json'))['aggregate']
final = json.load(open('tests/ragas_metrics.json'))['aggregate']
for k in sorted(set(base) | set(final)):
    print(f'{k:<30} baseline={base.get(k, 0):.3f}  final={final.get(k, 0):.3f}  delta={final.get(k, 0)-base.get(k, 0):+.3f}')
"
```

Expected: `answer_faithfulness` rises from ~0.44 to ≥0.80. `context_recall` rises from ~0.56 to ≥0.75. `hallucination_rate` ≤0.05. `grounding_bypass_rate` ≤0.02.

- [ ] **Step 3: Compare latency**

Run:
```bash
python -c "
import json
res = json.load(open('/tmp/intensive_test_results.json'))
latencies = sorted(r['latency_ms'] for r in res.get('results', []) if r.get('latency_ms'))
p50 = latencies[len(latencies)//2] if latencies else 0
p95 = latencies[int(len(latencies)*0.95)] if latencies else 0
print(f'p50={p50:.0f}ms p95={p95:.0f}ms n={len(latencies)}')
"
```

Expected: p50 within +300ms of baseline.

- [ ] **Step 4: Write the retrospective**

Create `docs/superpowers/retrospectives/2026-04-17-rag-accuracy.md`:

```markdown
# RAG Accuracy Workstream — Retrospective (2026-04-17)

## Outcome
- Faithfulness: X.XXX → Y.YYY (target 0.80: PASS/FAIL)
- Context recall: X.XXX → Y.YYY (target 0.75: PASS/FAIL)
- Hallucination rate: X.XXX → Y.YYY (target ≤0.05: PASS/FAIL)
- Latency p50 delta: +X ms (budget +300: PASS/FAIL)

## What moved what
- Task 2 (grounding fix): <impact or "no RAGAS impact, correctness only">
- Task 8 (hybrid dense+sparse): <delta in context_recall from baseline to post_hybrid>
- Task 9 (KG expansion): <delta in context_recall from post_hybrid to post_kg>

## Surprises
- <anything unexpected — e.g., "KG expansion added no measurable gain on this bank; likely bank under-represents cross-doc queries">

## Recommended next workstream
- <one of: extraction stubs (workstream 2), unified-model training (workstream 3), cleanup/consolidation, or "retrieval gains plateaued — extraction is the next bottleneck">
```

- [ ] **Step 5: Commit the retrospective**

Run:
```bash
git add -f docs/superpowers/retrospectives/2026-04-17-rag-accuracy.md tests/ragas_metrics.json
git commit -m "docs(rag): retrospective for hybrid retrieval workstream"
```

- [ ] **Step 6: Gate decision**

If all success criteria met: announce completion and recommend the next workstream based on the retrospective's "Recommended next workstream" line.

If criteria NOT met: the workstream remains open. The retrospective's "What moved what" section becomes the diagnostic for either (a) extending this spec with additional retrieval fixes or (b) concluding retrieval is not the bottleneck and escalating to the extraction workstream.

---

## Self-review checklist (author's own, already applied)

**Spec coverage:**
- §1 Grounding fix → Tasks 1-2
- §2 Hybrid retrieval (dense+sparse+RRF) → Task 8
- §3 SparseEncoder at AppState → Task 3
- §4 Ingestion populates sparse → Task 4
- §5 Backfill script → Tasks 5-7
- §6 KG expansion → Task 9 (simplified via GraphAugmenter — documented in "Design note" section above)
- §7 Dead-code deletion → Task 10
- Success gate → Task 11

**Type consistency:** `EvidenceChunk`, `RetrievalResult`, `GraphHints`, `GraphSnippet`, `SparseEncoder`, `ChunkRecord` names stable across tasks.

**Placeholder scan:** No TBDs. All code blocks are concrete. Backfill script, tests, and the grounding fix body are fully written out.

**One intentional flex:** Task 2's grounding-fix body has commented calibration ranges ("tune to match [diagnosis]"). This is intentional — the exact thresholds depend on Task 1's diagnostic output and locking them before diagnosis would be guessing. The invariant "Task 2 tests must pass" forces convergence.
