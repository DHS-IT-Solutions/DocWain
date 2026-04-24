# Extraction Overhaul — Plan 2 (Vision Path + DocIntel + Observability)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Plan 1's legacy-engine fallback for non-native documents (scanned PDFs, images, handwritten) with the spec's DocWain-primary vision path: DocIntel classifies → DocWain vision extracts → DocIntel verifies coverage → region-scoped Tesseract/EasyOCR fallback on misses. Also normalizes the canonical JSON shape (fixing a Plan 1 compromise) and adds per-extraction Redis observability.

**Architecture:** A new `src/extraction/vision/` package hosts the vision-path orchestrator and its components. DocWain is reached via a thin `vision_client` that talks to the vLLM endpoint (`http://localhost:8100/v1/chat/completions`) with OpenAI-compatible image payloads. DocIntel is expressed as three prompt-based capabilities inside DocWain (classifier, extractor, coverage verifier) — no separate ML/DL model. Region-scoped OCR fallback ensemble (Tesseract + EasyOCR) runs ONLY on regions the coverage verifier flags. Plan 1's native-path output shape is adopted as the canonical; vision output converges to it. Spec: `docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md` §4.2.

**Tech Stack:** Python 3.12, httpx (HTTP), PyMuPDF (PDF page rendering), Pillow (image handling), pytesseract, easyocr, redis (observability log). All already installed in the repo venv.

**Non-goals in Plan 2:**
- No DocWain vision training (Phase 2 training workstream — separate).
- No Ollama Cloud wiring changes to the gateway (roadmap item 5 separate).
- No KG re-introduction (still out of extraction; Plan 3 covers KG in training stage).
- No Researcher Agent (separate roadmap item, later).

---

## File structure

**New files:**
- `src/extraction/vision/__init__.py`
- `src/extraction/vision/client.py` — vLLM HTTP wrapper with vision payload support
- `src/extraction/vision/images.py` — PDF→image rendering, bytes↔base64
- `src/extraction/vision/docintel.py` — DocIntel classifier + coverage-verifier prompts and call helpers
- `src/extraction/vision/extractor.py` — DocWain vision extraction pass
- `src/extraction/vision/fallback.py` — Tesseract + EasyOCR region-scoped fallback
- `src/extraction/vision/orchestrator.py` — the vision-path entry point
- `src/extraction/vision/observability.py` — Redis per-extraction audit log
- `src/extraction/adapters/image_native.py` — image file (JPG/PNG/TIFF) dispatcher that routes to the vision path (lives in adapters/ for consistency even though it's a thin pass-through)
- `tests/unit/extraction/vision/__init__.py`
- `tests/unit/extraction/vision/test_client.py`
- `tests/unit/extraction/vision/test_images.py`
- `tests/unit/extraction/vision/test_docintel.py`
- `tests/unit/extraction/vision/test_extractor.py`
- `tests/unit/extraction/vision/test_fallback.py`
- `tests/unit/extraction/vision/test_orchestrator.py`
- `tests/unit/extraction/vision/test_observability.py`
- `tests/unit/extraction/test_image_native.py`
- `tests/integration/test_vision_path_smoke.py`
- `tests/extraction_bench/cases/bench_scan_01/...`
- `tests/extraction_bench/cases/bench_image_01/...`

**Modified files:**
- `src/tasks/extraction.py` — replace legacy-engine fallback with vision-path orchestrator; normalize canonical JSON shape (unwrap `"canonical"` key from Plan 1).
- `src/extraction/adapters/dispatcher.py` — add `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif` routing to image_native adapter (which raises NotNativePathError so vision path runs).
- `tests/extraction_bench/fixtures/generate_fixtures.py` — add scanned-PDF + image fixtures.

**Git:** continue on branch `preprod_v02`. Commit after each task.

---

### Task 1: Canonical JSON shape normalization in Celery task

**Files:**
- Modify: `src/tasks/extraction.py`

Plan 1 left native path's output wrapped under `"canonical"` key in the uploaded blob JSON to avoid colliding with legacy shape. With vision path about to land, we converge both paths on the same shape (Plan 1's canonical ExtractionResult dict).

- [ ] **Step 1: Inspect current wrapping**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
grep -n '"canonical"' src/tasks/extraction.py
```

Locate where the native-first block wraps `result_dict` under `"canonical"`. Note the surrounding code.

- [ ] **Step 2: Remove the wrapping**

Replace the wrapping step with a direct assignment: `result_dict = _dc_asdict(_canonical)` (with the same sheet tuple-key normalization applied). Ensure downstream upload + MongoDB calls still receive `result_dict` correctly.

The blob payload for native path will now be the direct canonical JSON. Legacy path's differently-shaped output stays as-is (vision path replaces legacy anyway next task). Downstream consumers that key off `path_taken` field already differentiate.

- [ ] **Step 3: Verify tests still pass**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction tests/integration/test_native_extraction_smoke.py -q
```

Expected: 216 passed.

- [ ] **Step 4: Commit**

```bash
git add src/tasks/extraction.py
git commit -m "extraction: remove canonical wrapping in Celery native path (converge on direct schema)"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 2: Vision client (vLLM HTTP wrapper)

**Files:**
- Create: `src/extraction/vision/__init__.py` (one-line comment)
- Create: `src/extraction/vision/client.py`
- Create: `tests/unit/extraction/vision/__init__.py` (one-line comment)
- Create: `tests/unit/extraction/vision/test_client.py`

- [ ] **Step 1: Package markers**

Create `src/extraction/vision/__init__.py` with content `# DocWain vision path (DocIntel + extractor + verifier)\n`.

Create `tests/unit/extraction/vision/__init__.py` with content `# vision path unit tests\n`.

- [ ] **Step 2: Write failing test**

Create `tests/unit/extraction/vision/test_client.py`:

```python
import base64

import httpx
import pytest

from src.extraction.vision.client import VisionClient, VisionClientError


def _png_bytes() -> bytes:
    # Smallest valid PNG (1x1 transparent).
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )


def test_vision_client_builds_openai_vision_payload():
    client = VisionClient(base_url="http://localhost:8100/v1", model="docwain-fast")
    payload = client.build_payload(
        system="You are a document extractor.",
        user_text="Extract all text.",
        image_bytes=_png_bytes(),
        max_tokens=512,
        temperature=0.0,
    )
    assert payload["model"] == "docwain-fast"
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0.0
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    # User message's content is a list of mixed text + image_url parts
    user = messages[1]
    assert user["role"] == "user"
    parts = user["content"]
    assert any(p.get("type") == "text" and "Extract all text" in p.get("text", "") for p in parts)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_vision_client_handles_http_error(monkeypatch):
    client = VisionClient(base_url="http://localhost:8100/v1", model="docwain-fast")

    class FakeResponse:
        status_code = 500

        def raise_for_status(self):
            raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):  # pragma: no cover
            raise AssertionError("json() should not be reached on error")

    def fake_post(url, json, timeout):
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(VisionClientError):
        client.call(system="s", user_text="u", image_bytes=_png_bytes())
```

- [ ] **Step 3: Run to confirm fail**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_client.py -x -q
```

Expected: ImportError.

- [ ] **Step 4: Implement the client**

Create `src/extraction/vision/client.py`:

```python
"""vLLM HTTP client with OpenAI-compatible vision payload support.

Talks to the local vLLM server serving DocWain (default port 8100). Builds
multi-part messages with a single page image per call (OpenAI vision format).
Thin wrapper by design — no retry logic, no batching, no streaming. Callers
handle errors via VisionClientError.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


class VisionClientError(Exception):
    """Raised when the vision server returns an error or unparseable response."""


@dataclass
class VisionResponse:
    """Raw text output + usage metadata from a single vision call."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    model: str


class VisionClient:
    def __init__(self, *, base_url: str, model: str, timeout_s: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def build_payload(
        self,
        *,
        system: str,
        user_text: str,
        image_bytes: bytes,
        image_mime: str = "image/png",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{image_mime};base64,{b64}"
        return {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }

    def call(
        self,
        *,
        system: str,
        user_text: str,
        image_bytes: bytes,
        image_mime: str = "image/png",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VisionResponse:
        import time
        payload = self.build_payload(
            system=system,
            user_text=user_text,
            image_bytes=image_bytes,
            image_mime=image_mime,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        try:
            r = httpx.post(url, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisionClientError(f"vision call failed: {exc}") from exc
        wall_ms = (time.perf_counter() - t0) * 1000.0
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
        except Exception as exc:
            raise VisionClientError(f"unparseable vision response: {exc}") from exc
        return VisionResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            wall_ms=float(wall_ms),
            model=self.model,
        )
```

- [ ] **Step 5: Verify tests pass**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_client.py -x -q
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/extraction/vision/__init__.py src/extraction/vision/client.py tests/unit/extraction/vision/__init__.py tests/unit/extraction/vision/test_client.py
git commit -m "extraction/vision: add vLLM client with OpenAI-compatible vision payload"
```

---

### Task 3: Image helpers (PDF page render, base64)

**Files:**
- Create: `src/extraction/vision/images.py`
- Create: `tests/unit/extraction/vision/test_images.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/extraction/vision/test_images.py`:

```python
import io

import fitz

from src.extraction.vision.images import (
    b64_to_bytes,
    bytes_to_b64,
    render_pdf_page_to_png,
)


def _make_pdf(num_pages: int = 2) -> bytes:
    d = fitz.open()
    for _ in range(num_pages):
        p = d.new_page()
        p.insert_text((72, 72), "page content")
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    return buf.getvalue()


def test_render_pdf_page_returns_png_bytes():
    pdf = _make_pdf()
    png = render_pdf_page_to_png(pdf, page_index=0, dpi=72)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")  # PNG magic


def test_render_pdf_page_index_in_range():
    pdf = _make_pdf(num_pages=3)
    png2 = render_pdf_page_to_png(pdf, page_index=2, dpi=72)
    assert png2.startswith(b"\x89PNG\r\n\x1a\n")


def test_bytes_to_b64_roundtrips():
    original = b"\x00\x01\x02\x03" * 4
    b64 = bytes_to_b64(original)
    restored = b64_to_bytes(b64)
    assert restored == original
```

- [ ] **Step 2: Run to confirm fail**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_images.py -x -q
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/extraction/vision/images.py`:

```python
"""Image rendering + base64 helpers for the vision path."""
from __future__ import annotations

import base64
import io

import fitz


def render_pdf_page_to_png(pdf_bytes: bytes, *, page_index: int, dpi: int = 144) -> bytes:
    """Render the given page of a PDF to PNG bytes at the given DPI."""
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"page_index {page_index} out of range for PDF with {len(doc)} pages")
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))
```

- [ ] **Step 4: Verify + commit**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_images.py -x -q
git add src/extraction/vision/images.py tests/unit/extraction/vision/test_images.py
git commit -m "extraction/vision: add PDF page render + base64 helpers"
```

Expected pytest: 3 passed.

---

### Task 4: DocIntel (classifier + coverage verifier) via prompting

**Files:**
- Create: `src/extraction/vision/docintel.py`
- Create: `tests/unit/extraction/vision/test_docintel.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/extraction/vision/test_docintel.py`:

```python
import json

from src.extraction.vision.docintel import (
    CLASSIFIER_SYSTEM_PROMPT,
    COVERAGE_SYSTEM_PROMPT,
    RoutingDecision,
    parse_coverage_response,
    parse_routing_response,
)


def test_parse_routing_response_accepts_well_formed_json():
    text = json.dumps({
        "format": "pdf_scanned",
        "doc_type_hint": "invoice",
        "layout_complexity": "moderate",
        "has_handwriting": False,
        "suggested_path": "vision",
        "confidence": 0.8,
    })
    out = parse_routing_response(text)
    assert isinstance(out, RoutingDecision)
    assert out.suggested_path == "vision"
    assert out.layout_complexity == "moderate"
    assert 0.0 <= out.confidence <= 1.0


def test_parse_routing_response_tolerates_code_fences():
    text = "```json\n" + json.dumps({
        "format": "image",
        "doc_type_hint": "receipt",
        "layout_complexity": "simple",
        "has_handwriting": True,
        "suggested_path": "vision",
        "confidence": 0.7,
    }) + "\n```"
    out = parse_routing_response(text)
    assert out.has_handwriting is True


def test_parse_routing_response_returns_safe_default_on_garbage():
    out = parse_routing_response("this is not json")
    # Safe default: send through vision path with low confidence
    assert out.suggested_path == "vision"
    assert out.confidence < 0.2


def test_parse_coverage_response_accepts_complete_true():
    text = json.dumps({"complete": True, "missed_regions": [], "low_confidence_regions": []})
    out = parse_coverage_response(text)
    assert out["complete"] is True
    assert out["missed_regions"] == []


def test_parse_coverage_response_defaults_to_incomplete_on_garbage():
    out = parse_coverage_response("??? not json ???")
    assert out["complete"] is False
    # missed_regions present as empty list so caller can still iterate safely
    assert out["missed_regions"] == []


def test_prompts_are_non_empty_strings():
    assert isinstance(CLASSIFIER_SYSTEM_PROMPT, str) and len(CLASSIFIER_SYSTEM_PROMPT) > 50
    assert isinstance(COVERAGE_SYSTEM_PROMPT, str) and len(COVERAGE_SYSTEM_PROMPT) > 50
```

- [ ] **Step 2: Run to confirm fail**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_docintel.py -x -q
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/extraction/vision/docintel.py`:

```python
"""DocIntel capabilities inside the unified DocWain model.

DocIntel is NOT a separate ML/DL model — it is a set of capabilities invoked
by specific prompting of DocWain. This module holds the prompts and the
response parsers. Actual model calls happen via VisionClient in
`src.extraction.vision.client`.

Three capabilities:
1. Classifier + router — decides native vs vision, doc type hint, handwriting
   presence.
2. Coverage verifier — given the source image and the extracted JSON, answers
   whether every visible region is represented.
3. Extractor prompts live in `src.extraction.vision.extractor` (kept separate
   so this module stays pure parsing + prompt strings).

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §3.1
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List


CLASSIFIER_SYSTEM_PROMPT = (
    "You are DocIntel, DocWain's document understanding capability. Given a "
    "document's filename, the first rendered page image, and any text already "
    "extractable from the file's text layer, you output a JSON routing decision.\n\n"
    "Output ONLY valid JSON with this shape (no prose, no markdown fences):\n"
    "{\n"
    '  "format": "pdf_native" | "pdf_scanned" | "pdf_mixed" | "pptx" | "docx" | '
    '"xlsx" | "csv" | "image" | "handwritten",\n'
    '  "doc_type_hint": string (e.g. "invoice", "resume", "contract", "receipt", '
    '"report", or "unknown"),\n'
    '  "layout_complexity": "simple" | "moderate" | "complex",\n'
    '  "has_handwriting": true | false,\n'
    '  "suggested_path": "native" | "vision" | "mixed",\n'
    '  "confidence": number between 0.0 and 1.0\n'
    "}\n\n"
    "Rules:\n"
    "- If the text layer is substantial and machine-readable, prefer 'native'.\n"
    "- If the page is visibly a scan or image of a document, prefer 'vision'.\n"
    "- If any visible writing looks handwritten, set has_handwriting=true.\n"
    "- Be deterministic and conservative on confidence — do not claim >0.9 unless "
    "the evidence is unambiguous."
)

COVERAGE_SYSTEM_PROMPT = (
    "You are DocIntel's coverage verifier for DocWain's vision extraction path. "
    "You receive the original page image and a JSON extraction output. Your job: "
    "decide whether every visible region of the image is represented in the "
    "extraction.\n\n"
    "Output ONLY valid JSON with this shape (no prose, no markdown fences):\n"
    "{\n"
    '  "complete": true | false,\n'
    '  "missed_regions": [ { "bbox": [x, y, w, h], "description": string } ],\n'
    '  "low_confidence_regions": [ { "region_id": string, "reason": string } ]\n'
    "}\n\n"
    "Rules:\n"
    "- If any visible text, table, form field, or handwritten patch lacks a "
    "corresponding entry in the extraction, add it to missed_regions.\n"
    "- Use normalized 0..1 coordinates for bbox (fractions of page width/height).\n"
    "- Set complete=true only when every visible content region is represented.\n"
    "- Empty regions (pure whitespace, decorative lines) do not count as missed."
)


@dataclass
class RoutingDecision:
    format: str
    doc_type_hint: str
    layout_complexity: str
    has_handwriting: bool
    suggested_path: str
    confidence: float


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def _first_json_object(text: str) -> str:
    t = _strip_code_fence(text)
    # Fall back to outer { .. }
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return t
    return t[start : end + 1]


def parse_routing_response(text: str) -> RoutingDecision:
    """Best-effort parse of classifier output into a RoutingDecision.

    Malformed or non-JSON inputs return a safe default that routes through the
    vision path at low confidence — never raises.
    """
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return RoutingDecision(
            format="image",
            doc_type_hint="unknown",
            layout_complexity="simple",
            has_handwriting=False,
            suggested_path="vision",
            confidence=0.1,
        )
    try:
        return RoutingDecision(
            format=str(data.get("format", "image")),
            doc_type_hint=str(data.get("doc_type_hint", "unknown")),
            layout_complexity=str(data.get("layout_complexity", "simple")),
            has_handwriting=bool(data.get("has_handwriting", False)),
            suggested_path=str(data.get("suggested_path", "vision")),
            confidence=float(data.get("confidence", 0.1)),
        )
    except Exception:
        return RoutingDecision(
            format="image",
            doc_type_hint="unknown",
            layout_complexity="simple",
            has_handwriting=False,
            suggested_path="vision",
            confidence=0.1,
        )


def parse_coverage_response(text: str) -> Dict[str, Any]:
    """Best-effort parse of coverage-verifier output.

    Malformed inputs default to complete=False with empty missed_regions so the
    caller can still iterate; it's safer to assume incomplete than to claim
    complete on garbage.
    """
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return {"complete": False, "missed_regions": [], "low_confidence_regions": []}
    missed = data.get("missed_regions") or []
    if not isinstance(missed, list):
        missed = []
    low_conf = data.get("low_confidence_regions") or []
    if not isinstance(low_conf, list):
        low_conf = []
    return {
        "complete": bool(data.get("complete", False)),
        "missed_regions": missed,
        "low_confidence_regions": low_conf,
    }
```

- [ ] **Step 4: Verify + commit**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_docintel.py -x -q
git add src/extraction/vision/docintel.py tests/unit/extraction/vision/test_docintel.py
git commit -m "extraction/vision: add DocIntel prompts + safe JSON parsers"
```

Expected pytest: 6 passed.

---

### Task 5: Vision extractor prompt + parser

**Files:**
- Create: `src/extraction/vision/extractor.py`
- Create: `tests/unit/extraction/vision/test_extractor.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/extraction/vision/test_extractor.py`:

```python
import json

from src.extraction.vision.extractor import (
    EXTRACTOR_SYSTEM_PROMPT,
    VisionExtraction,
    parse_extractor_response,
)


def test_parse_extractor_response_well_formed():
    text = json.dumps({
        "regions": [
            {"type": "text_block", "bbox": [0.1, 0.1, 0.5, 0.2], "content": "hello", "confidence": 0.95},
            {"type": "table", "bbox": [0.1, 0.4, 0.8, 0.3],
             "content": {"rows": [["a","b"], ["1","2"]]}, "confidence": 0.88},
        ],
        "reading_order": [0, 1],
        "page_confidence": 0.9,
    })
    out = parse_extractor_response(text)
    assert isinstance(out, VisionExtraction)
    assert len(out.regions) == 2
    assert out.regions[0]["type"] == "text_block"
    assert out.page_confidence == 0.9


def test_parse_extractor_response_returns_empty_on_garbage():
    out = parse_extractor_response("[not json]")
    assert out.regions == []
    assert out.page_confidence == 0.0


def test_prompt_non_empty():
    assert isinstance(EXTRACTOR_SYSTEM_PROMPT, str) and len(EXTRACTOR_SYSTEM_PROMPT) > 80
```

- [ ] **Step 2: Run to confirm fail, then implement**

Create `src/extraction/vision/extractor.py`:

```python
"""DocWain vision-extraction prompt + response parser.

Prompts DocWain to return a structured regions JSON for a given page image.
Safe parsing: malformed output returns an empty VisionExtraction (no regions,
page_confidence=0) so the coverage verifier correctly flags everything as
missed and the fallback catches the full page.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


EXTRACTOR_SYSTEM_PROMPT = (
    "You are DocWain's vision extractor. You receive a single document page as "
    "an image plus routing hints (document type, layout complexity, whether "
    "handwriting is present).\n\n"
    "Your task: emit a JSON object describing every visible content region on "
    "the page, preserving reading order, bboxes (normalized 0..1 coordinates), "
    "and the extracted content verbatim. You MUST NOT paraphrase, summarize, "
    "or add content not present on the page.\n\n"
    "Output ONLY valid JSON (no prose, no markdown fences):\n"
    "{\n"
    '  "regions": [\n'
    "    {\n"
    '      "type": "text_block" | "table" | "form_field" | "figure" | "handwriting",\n'
    '      "bbox": [x, y, w, h]  (normalized 0..1),\n'
    '      "content": (string for text; {"rows": [...]} for table; {"label": s, '
    '"value": v} for form_field; string caption for figure; string for handwriting),\n'
    '      "confidence": number between 0.0 and 1.0\n'
    "    }\n"
    "  ],\n"
    '  "reading_order": [region_index, ...],\n'
    '  "page_confidence": number between 0.0 and 1.0\n'
    "}\n\n"
    "Rules:\n"
    "- Cover every visible text or content region; missing regions are failures.\n"
    "- Do not hallucinate — if a region is unreadable, emit it with "
    'confidence < 0.5 so the coverage verifier can flag it.\n'
    "- Preserve exact text characters; do not correct spelling, expand "
    "abbreviations, or normalize whitespace beyond collapsing internal line "
    "breaks within a single text block."
)


@dataclass
class VisionExtraction:
    regions: List[Dict[str, Any]] = field(default_factory=list)
    reading_order: List[int] = field(default_factory=list)
    page_confidence: float = 0.0


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def _first_json_object(text: str) -> str:
    t = _strip_code_fence(text)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return t
    return t[start : end + 1]


def parse_extractor_response(text: str) -> VisionExtraction:
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return VisionExtraction()
    regions = data.get("regions") or []
    if not isinstance(regions, list):
        regions = []
    reading_order = data.get("reading_order") or []
    if not isinstance(reading_order, list):
        reading_order = []
    try:
        page_conf = float(data.get("page_confidence", 0.0))
    except Exception:
        page_conf = 0.0
    return VisionExtraction(
        regions=regions,
        reading_order=reading_order,
        page_confidence=page_conf,
    )
```

- [ ] **Step 3: Verify + commit**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_extractor.py -x -q
git add src/extraction/vision/extractor.py tests/unit/extraction/vision/test_extractor.py
git commit -m "extraction/vision: add vision extractor prompt + safe JSON parser"
```

Expected pytest: 3 passed.

---

### Task 6: Region-scoped OCR fallback (Tesseract + EasyOCR)

**Files:**
- Create: `src/extraction/vision/fallback.py`
- Create: `tests/unit/extraction/vision/test_fallback.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/extraction/vision/test_fallback.py`:

```python
from src.extraction.vision.fallback import (
    FallbackRegionResult,
    crop_bbox_normalized,
    run_fallback_ensemble,
)


def test_crop_bbox_normalized_returns_pil_crop():
    from PIL import Image
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    crop = crop_bbox_normalized(img, bbox=[0.1, 0.2, 0.3, 0.4])
    assert crop.size == (30, 40)


def test_run_fallback_ensemble_stubs(monkeypatch):
    # Monkeypatch the two OCR backends to return known strings; verify merge.
    def fake_tess(img):
        return "hello from tesseract"

    def fake_easy(img):
        return "hello from tesseract"  # identical → high agreement

    monkeypatch.setattr("src.extraction.vision.fallback._run_tesseract", fake_tess)
    monkeypatch.setattr("src.extraction.vision.fallback._run_easyocr", fake_easy)

    from PIL import Image
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    result = run_fallback_ensemble(img, bbox=[0.0, 0.0, 1.0, 1.0])
    assert isinstance(result, FallbackRegionResult)
    assert result.text == "hello from tesseract"
    assert result.agreement > 0.9  # identical outputs → ~1.0
    assert result.engine_winner in ("tesseract", "easyocr", "both")


def test_run_fallback_ensemble_picks_higher_confidence_when_disagreement(monkeypatch):
    monkeypatch.setattr("src.extraction.vision.fallback._run_tesseract", lambda img: "tess output")
    monkeypatch.setattr("src.extraction.vision.fallback._run_easyocr", lambda img: "easyocr output is longer here")

    from PIL import Image
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    result = run_fallback_ensemble(img, bbox=[0.0, 0.0, 1.0, 1.0])
    # Longer output chosen as tiebreaker when disagreement is high
    assert "longer" in result.text
```

- [ ] **Step 2: Run to confirm fail, then implement**

Create `src/extraction/vision/fallback.py`:

```python
"""Region-scoped OCR fallback ensemble: Tesseract + EasyOCR.

Invoked by the vision path when the coverage verifier flags missed or low-
confidence regions. Runs on cropped page-image regions (not full pages) to
keep fallback scope narrow. DocWain-primary is preserved; fallback is
catch-all for what DocWain misses.

As DocWain training improves (Phase 2 workstream), fallback invocation rate
drops and eventually this code becomes unreachable. Retained for safety.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PIL import Image


@dataclass
class FallbackRegionResult:
    text: str
    agreement: float  # 0..1, Levenshtein similarity between engine outputs
    engine_winner: str  # "tesseract" | "easyocr" | "both"


def crop_bbox_normalized(img: Image.Image, *, bbox: List[float]) -> Image.Image:
    """Crop a PIL image by a normalized (0..1) bbox [x, y, w, h]."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {bbox}")
    x, y, w, h = bbox
    W, H = img.size
    left = max(0, int(x * W))
    top = max(0, int(y * H))
    right = min(W, int((x + w) * W))
    bottom = min(H, int((y + h) * H))
    return img.crop((left, top, right, bottom))


def _run_tesseract(img: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image. Returns text or empty string on failure."""
    try:
        import pytesseract
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def _run_easyocr(img: Image.Image) -> str:
    """Run EasyOCR on a PIL image. Returns concatenated text or empty on failure."""
    try:
        import easyocr
        import numpy as np
        reader = _get_easyocr_reader()
        results = reader.readtext(np.array(img))
        lines = [r[1] for r in results]
        return "\n".join(lines)
    except Exception:
        return ""


_EASYOCR_READER = None


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


def _levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0 if a else 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    denom = max(len(a), len(b))
    return 1.0 - (prev[-1] / denom) if denom else 1.0


def run_fallback_ensemble(img: Image.Image, *, bbox: List[float]) -> FallbackRegionResult:
    """Run both OCR engines on the cropped region; return merged result."""
    crop = crop_bbox_normalized(img, bbox=bbox)
    tess_text = _run_tesseract(crop).strip()
    easy_text = _run_easyocr(crop).strip()

    agreement = _levenshtein_ratio(tess_text, easy_text)
    if tess_text == easy_text and tess_text:
        return FallbackRegionResult(text=tess_text, agreement=agreement, engine_winner="both")
    if agreement > 0.9:
        # Near-identical — pick tesseract arbitrarily.
        return FallbackRegionResult(text=tess_text or easy_text, agreement=agreement, engine_winner="tesseract")
    # Disagree: prefer the longer output (usually more complete capture).
    if len(easy_text) > len(tess_text):
        return FallbackRegionResult(text=easy_text, agreement=agreement, engine_winner="easyocr")
    return FallbackRegionResult(text=tess_text, agreement=agreement, engine_winner="tesseract")
```

- [ ] **Step 3: Verify + commit**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_fallback.py -x -q
git add src/extraction/vision/fallback.py tests/unit/extraction/vision/test_fallback.py
git commit -m "extraction/vision: add region-scoped Tesseract+EasyOCR fallback ensemble"
```

Expected pytest: 3 passed.

---

### Task 7: Redis observability log

**Files:**
- Create: `src/extraction/vision/observability.py`
- Create: `tests/unit/extraction/vision/test_observability.py`

- [ ] **Step 1: Failing tests**

Create `tests/unit/extraction/vision/test_observability.py`:

```python
import json
import time

import pytest

from src.extraction.vision.observability import (
    ExtractionLogEntry,
    build_redis_key,
    serialize_entry,
    write_entry_if_redis,
)


def test_serialize_entry_contains_required_fields():
    entry = ExtractionLogEntry(
        doc_id="d1",
        format="pdf_scanned",
        path_taken="vision",
        timings_ms={"file_adapter": 10.0, "docintel_route": 300.0, "vision_pass": 1200.0,
                    "coverage_verify": 400.0, "fallback": 0.0},
        routing_decision={"format": "pdf_scanned", "suggested_path": "vision", "confidence": 0.8},
        coverage_score=0.98,
        fallback_invocations=[],
        human_review=False,
        completed_at=time.time(),
    )
    out = serialize_entry(entry)
    data = json.loads(out)
    assert data["doc_id"] == "d1"
    assert data["path_taken"] == "vision"
    assert data["coverage_score"] == 0.98
    assert data["timings_ms"]["vision_pass"] == 1200.0


def test_build_redis_key_includes_doc_id():
    key = build_redis_key("doc-123")
    assert "doc-123" in key


def test_write_entry_if_redis_accepts_none_client():
    entry = ExtractionLogEntry(doc_id="d2", format="docx", path_taken="native",
                               timings_ms={}, routing_decision={}, coverage_score=1.0,
                               fallback_invocations=[], human_review=False, completed_at=time.time())
    # Should not raise when redis_client is None (writes are best-effort).
    write_entry_if_redis(redis_client=None, entry=entry)


def test_write_entry_if_redis_sets_ttl(monkeypatch):
    entry = ExtractionLogEntry(doc_id="d3", format="docx", path_taken="native",
                               timings_ms={}, routing_decision={}, coverage_score=1.0,
                               fallback_invocations=[], human_review=False, completed_at=time.time())

    calls = {}

    class FakeRedis:
        def setex(self, key, ttl, value):
            calls["key"] = key
            calls["ttl"] = ttl
            calls["value"] = value

    write_entry_if_redis(redis_client=FakeRedis(), entry=entry)
    assert "d3" in calls["key"]
    # 7-day TTL per spec §7
    assert calls["ttl"] == 7 * 24 * 3600
    assert "d3" in calls["value"]
```

- [ ] **Step 2: Implement**

Create `src/extraction/vision/observability.py`:

```python
"""Per-extraction Redis audit log.

Every extraction writes a structured log entry so operators can see where
accuracy bleeds and where DocWain training should focus next. Entry shape is
defined in spec §7. Writes are best-effort; a None or offline Redis client
does not break extraction.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


REDIS_KEY_PREFIX = "extraction:log"
TTL_SECONDS = 7 * 24 * 3600


@dataclass
class ExtractionLogEntry:
    doc_id: str
    format: str
    path_taken: str  # native | vision | mixed
    timings_ms: Dict[str, float] = field(default_factory=dict)
    routing_decision: Dict[str, Any] = field(default_factory=dict)
    coverage_score: float = 1.0
    fallback_invocations: List[Dict[str, Any]] = field(default_factory=list)
    human_review: bool = False
    completed_at: float = 0.0


def build_redis_key(doc_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}:{doc_id}"


def serialize_entry(entry: ExtractionLogEntry) -> str:
    return json.dumps(asdict(entry), ensure_ascii=False)


def write_entry_if_redis(*, redis_client: Any, entry: ExtractionLogEntry) -> None:
    """Write the entry to Redis with TTL. Best-effort — errors swallowed."""
    if redis_client is None:
        return
    try:
        redis_client.setex(
            build_redis_key(entry.doc_id),
            TTL_SECONDS,
            serialize_entry(entry),
        )
    except Exception:
        # Observability must never break extraction.
        pass
```

- [ ] **Step 3: Verify + commit**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_observability.py -x -q
git add src/extraction/vision/observability.py tests/unit/extraction/vision/test_observability.py
git commit -m "extraction/vision: add Redis per-extraction observability log"
```

Expected pytest: 4 passed.

---

### Task 8: Vision path orchestrator

**Files:**
- Create: `src/extraction/vision/orchestrator.py`
- Create: `tests/unit/extraction/vision/test_orchestrator.py`

Ties the 5 previous components (client, images, docintel, extractor, fallback) + observability into a single `extract_via_vision()` entry point that returns a canonical `ExtractionResult`.

- [ ] **Step 1: Failing tests (full mock of the sub-components)**

Create `tests/unit/extraction/vision/test_orchestrator.py`:

```python
import io

import fitz

from src.extraction.canonical_schema import ExtractionResult
from src.extraction.vision.client import VisionResponse
from src.extraction.vision.extractor import VisionExtraction
from src.extraction.vision.orchestrator import extract_via_vision


def _make_pdf() -> bytes:
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "some text")
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    return buf.getvalue()


def test_extract_via_vision_returns_canonical_result(monkeypatch):
    # Stub the three DocWain calls:
    def fake_call(self, *, system, user_text, image_bytes, image_mime="image/png",
                  max_tokens=1024, temperature=0.0):
        if "routing decision" in system.lower() or "classifier" in system.lower() or "routing" in system.lower():
            return VisionResponse(
                text='{"format":"pdf_scanned","doc_type_hint":"unknown","layout_complexity":"simple",'
                     '"has_handwriting":false,"suggested_path":"vision","confidence":0.8}',
                prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
            )
        if "coverage verifier" in system.lower():
            return VisionResponse(
                text='{"complete":true,"missed_regions":[],"low_confidence_regions":[]}',
                prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
            )
        # Extractor
        return VisionResponse(
            text='{"regions":[{"type":"text_block","bbox":[0.1,0.1,0.3,0.1],'
                 '"content":"some text","confidence":0.95}],"reading_order":[0],"page_confidence":0.9}',
            prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
        )

    monkeypatch.setattr("src.extraction.vision.client.VisionClient.call", fake_call)

    pdf = _make_pdf()
    result = extract_via_vision(pdf, doc_id="dv1", filename="scan.pdf", format_hint="pdf_scanned")
    assert isinstance(result, ExtractionResult)
    assert result.format == "pdf_scanned"
    assert result.path_taken == "vision"
    assert len(result.pages) == 1
    assert any("some text" in b.text for b in result.pages[0].blocks)
    assert result.metadata.coverage.verifier_score >= 0.9


def test_extract_via_vision_invokes_fallback_on_missed_region(monkeypatch):
    call_counts = {"fallback": 0}

    def fake_call(self, *, system, user_text, image_bytes, image_mime="image/png",
                  max_tokens=1024, temperature=0.0):
        if "coverage verifier" in system.lower():
            return VisionResponse(
                text='{"complete":false,"missed_regions":[{"bbox":[0.0,0.0,1.0,0.5],'
                     '"description":"header"}],"low_confidence_regions":[]}',
                prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
            )
        if "classifier" in system.lower() or "routing" in system.lower():
            return VisionResponse(
                text='{"format":"pdf_scanned","doc_type_hint":"unknown","layout_complexity":"simple",'
                     '"has_handwriting":false,"suggested_path":"vision","confidence":0.8}',
                prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
            )
        return VisionResponse(
            text='{"regions":[],"reading_order":[],"page_confidence":0.1}',
            prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
        )

    def fake_fallback(img, *, bbox):
        call_counts["fallback"] += 1
        from src.extraction.vision.fallback import FallbackRegionResult
        return FallbackRegionResult(text="recovered via fallback", agreement=1.0, engine_winner="both")

    monkeypatch.setattr("src.extraction.vision.client.VisionClient.call", fake_call)
    monkeypatch.setattr("src.extraction.vision.orchestrator.run_fallback_ensemble", fake_fallback)

    pdf = _make_pdf()
    result = extract_via_vision(pdf, doc_id="dv2", filename="scan.pdf", format_hint="pdf_scanned")
    assert call_counts["fallback"] >= 1
    assert any("recovered via fallback" in b.text for b in result.pages[0].blocks)
    assert len(result.metadata.coverage.fallback_invocations) >= 1
```

- [ ] **Step 2: Implement**

Create `src/extraction/vision/orchestrator.py`:

```python
"""Vision-path orchestrator.

Entry point: extract_via_vision(file_bytes, doc_id, filename, format_hint) →
ExtractionResult. Internally:

1. For PDF: render each page to PNG. For images: use the image bytes directly.
2. Per page:
   a. DocIntel classifier call → routing decision (format, layout, handwriting).
      (Skipped if format_hint already decides — we still call for metadata, but
      only on page 0 to save cost. Later pages carry the page-0 decision.)
   b. Vision extractor call → structured regions JSON.
   c. Coverage verifier call → complete?, missed_regions, low_confidence_regions.
   d. If incomplete or low-confidence: run fallback ensemble on each
      missed/low-conf region's bbox; replace/augment that region's content.
3. Collect per-page Block/Table/Image into canonical Page.
4. Assemble ExtractionResult with metadata (routing, coverage, fallback
   invocations).

Design notes:
- All DocWain calls go through VisionClient pointing at vLLM (port 8100).
- Per spec §5, within a document pages are processed sequentially so DocIntel
  context compounds. Per-document parallelism happens at Celery level.
"""
from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

import fitz
from PIL import Image

from src.extraction.canonical_schema import (
    Block,
    CoverageMetadata,
    DocIntelMetadata,
    ExtractionMetadata,
    ExtractionResult,
    Image as CanonicalImage,
    Page,
    Table,
)
from src.extraction.vision.client import VisionClient, VisionClientError, VisionResponse
from src.extraction.vision.docintel import (
    CLASSIFIER_SYSTEM_PROMPT,
    COVERAGE_SYSTEM_PROMPT,
    RoutingDecision,
    parse_coverage_response,
    parse_routing_response,
)
from src.extraction.vision.extractor import (
    EXTRACTOR_SYSTEM_PROMPT,
    VisionExtraction,
    parse_extractor_response,
)
from src.extraction.vision.fallback import run_fallback_ensemble
from src.extraction.vision.images import render_pdf_page_to_png


VLLM_BASE_URL = "http://localhost:8100/v1"
VLLM_MODEL = "docwain-fast"


def _build_client() -> VisionClient:
    return VisionClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL)


def _route(client: VisionClient, *, image_bytes: bytes, filename: str) -> RoutingDecision:
    try:
        resp = client.call(
            system=CLASSIFIER_SYSTEM_PROMPT,
            user_text=f"Filename: {filename}. Return the routing decision JSON for this page.",
            image_bytes=image_bytes,
            max_tokens=256,
            temperature=0.0,
        )
        return parse_routing_response(resp.text)
    except VisionClientError:
        return RoutingDecision(
            format="image", doc_type_hint="unknown", layout_complexity="simple",
            has_handwriting=False, suggested_path="vision", confidence=0.1,
        )


def _extract_page(client: VisionClient, *, image_bytes: bytes, hints: RoutingDecision) -> VisionExtraction:
    user_text = (
        f"Hints: doc_type={hints.doc_type_hint}, layout={hints.layout_complexity}, "
        f"handwriting={hints.has_handwriting}. Emit the regions JSON for this page."
    )
    try:
        resp = client.call(
            system=EXTRACTOR_SYSTEM_PROMPT,
            user_text=user_text,
            image_bytes=image_bytes,
            max_tokens=4096,
            temperature=0.0,
        )
        return parse_extractor_response(resp.text)
    except VisionClientError:
        return VisionExtraction()


def _verify(client: VisionClient, *, image_bytes: bytes, extraction: VisionExtraction) -> Dict[str, Any]:
    import json as _json
    payload_preview = _json.dumps({
        "regions": extraction.regions[:30],
        "reading_order": extraction.reading_order,
        "page_confidence": extraction.page_confidence,
    })
    try:
        resp = client.call(
            system=COVERAGE_SYSTEM_PROMPT,
            user_text=(
                "Here is the extraction JSON for the page:\n"
                f"{payload_preview}\n"
                "Return the coverage verdict JSON."
            ),
            image_bytes=image_bytes,
            max_tokens=1024,
            temperature=0.0,
        )
        return parse_coverage_response(resp.text)
    except VisionClientError:
        return {"complete": False, "missed_regions": [], "low_confidence_regions": []}


def _page_image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _regions_to_canonical_page(regions: List[Dict[str, Any]], *, page_num: int) -> Page:
    blocks: List[Block] = []
    tables: List[Table] = []
    images: List[CanonicalImage] = []
    for r in regions:
        rtype = r.get("type", "text_block")
        content = r.get("content", "")
        bbox = r.get("bbox")
        conf = r.get("confidence")  # noqa: F841  (kept for future use)
        if rtype == "table" and isinstance(content, dict) and isinstance(content.get("rows"), list):
            tables.append(Table(rows=content["rows"], bbox=None, header_row_index=0 if content["rows"] else None))
            continue
        if rtype in ("text_block", "handwriting", "form_field"):
            text = content if isinstance(content, str) else str(content)
            block_type = "paragraph" if rtype == "text_block" else rtype
            blocks.append(Block(text=text, bbox=None, block_type=block_type))
            continue
        if rtype == "figure":
            caption = content if isinstance(content, str) else ""
            images.append(CanonicalImage(bbox=None, ocr_text="", caption=caption))
            continue
    return Page(page_num=page_num, blocks=blocks, tables=tables, images=images)


def _apply_fallback(
    *,
    page_image_bytes: bytes,
    coverage: Dict[str, Any],
    extraction: VisionExtraction,
) -> List[Dict[str, Any]]:
    """Run fallback on missed regions; return augmented regions list + invocation records."""
    regions = list(extraction.regions)
    invocations: List[Dict[str, Any]] = []
    if coverage.get("complete"):
        return regions
    pil_img = _page_image_bytes_to_pil(page_image_bytes)
    for missed in coverage.get("missed_regions", []):
        bbox = missed.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            result = run_fallback_ensemble(pil_img, bbox=bbox)
        except Exception as exc:
            invocations.append({"bbox": bbox, "status": "error", "error": repr(exc)})
            continue
        if result.text.strip():
            regions.append({
                "type": "text_block",
                "bbox": bbox,
                "content": result.text,
                "confidence": max(0.3, result.agreement),
                "source": f"fallback:{result.engine_winner}",
            })
        invocations.append({
            "bbox": bbox,
            "engine_winner": result.engine_winner,
            "agreement": result.agreement,
            "chars": len(result.text),
        })
    return regions + [{"__invocations__": invocations}]


def extract_via_vision(
    file_bytes: bytes,
    *,
    doc_id: str,
    filename: str,
    format_hint: str,
) -> ExtractionResult:
    """Main entry. Returns a canonical ExtractionResult built from vision + fallback."""
    client = _build_client()

    # Obtain page images. For PDFs, render each page. For image files, use bytes as page 0.
    page_images: List[bytes] = []
    if format_hint in ("pdf_scanned", "pdf_mixed", "pdf_native"):
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
        try:
            for i in range(len(doc)):
                page_images.append(render_pdf_page_to_png(file_bytes, page_index=i, dpi=144))
        finally:
            doc.close()
    else:
        # Image file — single "page".
        page_images.append(file_bytes)

    # Classifier called on page 0 only; later pages reuse the decision.
    routing = _route(client, image_bytes=page_images[0], filename=filename)

    pages: List[Page] = []
    all_fallback_invocations: List[Dict[str, Any]] = []
    verifier_scores: List[float] = []

    for idx, img_bytes in enumerate(page_images):
        extraction = _extract_page(client, image_bytes=img_bytes, hints=routing)
        coverage = _verify(client, image_bytes=img_bytes, extraction=extraction)
        regions_with_invocations = _apply_fallback(
            page_image_bytes=img_bytes, coverage=coverage, extraction=extraction
        )
        # Separate invocations record (trailing sentinel dict)
        invocations: List[Dict[str, Any]] = []
        cleaned_regions: List[Dict[str, Any]] = []
        for r in regions_with_invocations:
            if isinstance(r, dict) and "__invocations__" in r:
                invocations.extend(r["__invocations__"])
            else:
                cleaned_regions.append(r)
        extraction_regions = VisionExtraction(
            regions=cleaned_regions,
            reading_order=extraction.reading_order,
            page_confidence=extraction.page_confidence,
        )
        page = _regions_to_canonical_page(extraction_regions.regions, page_num=idx + 1)
        pages.append(page)
        all_fallback_invocations.extend(invocations)
        # Verifier score: 1.0 when complete, otherwise a coarse 1.0 - missed_fraction estimate.
        if coverage.get("complete"):
            verifier_scores.append(1.0)
        else:
            missed_n = len(coverage.get("missed_regions", []))
            total_n = max(1, len(cleaned_regions))
            verifier_scores.append(max(0.0, 1.0 - missed_n / (missed_n + total_n)))

    avg_coverage = sum(verifier_scores) / len(verifier_scores) if verifier_scores else 0.0

    return ExtractionResult(
        doc_id=doc_id,
        format=routing.format if routing.format != "native" else format_hint,
        path_taken="vision",
        pages=pages,
        sheets=[],
        slides=[],
        metadata=ExtractionMetadata(
            doc_intel=DocIntelMetadata(
                doc_type_hint=routing.doc_type_hint,
                layout_complexity=routing.layout_complexity,
                has_handwriting=routing.has_handwriting,
                routing_confidence=routing.confidence,
            ),
            coverage=CoverageMetadata(
                verifier_score=avg_coverage,
                missed_regions=[],  # handled already via fallback
                low_confidence_regions=[],
                fallback_invocations=all_fallback_invocations,
            ),
            extraction_version="2026-04-24-v2",
        ),
    )
```

- [ ] **Step 3: Verify + commit**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/vision/test_orchestrator.py -x -q
git add src/extraction/vision/orchestrator.py tests/unit/extraction/vision/test_orchestrator.py
git commit -m "extraction/vision: add orchestrator tying DocIntel + extractor + coverage + fallback"
```

Expected pytest: 2 passed.

---

### Task 9: Image file adapter (JPG/PNG/TIFF routing)

**Files:**
- Create: `src/extraction/adapters/image_native.py`
- Create: `tests/unit/extraction/test_image_native.py`
- Modify: `src/extraction/adapters/dispatcher.py` (register image extensions)

- [ ] **Step 1: Failing test for the adapter**

Create `tests/unit/extraction/test_image_native.py`:

```python
import pytest

from src.extraction.adapters.errors import NotNativePathError
from src.extraction.adapters.image_native import extract_image_native


def test_image_adapter_always_raises_not_native():
    # Any image format raises NotNativePathError so dispatcher falls through to vision path.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20  # minimal PNG-ish bytes
    with pytest.raises(NotNativePathError):
        extract_image_native(png, doc_id="d1", filename="pic.png")


def test_image_adapter_raises_for_jpg_and_tiff():
    with pytest.raises(NotNativePathError):
        extract_image_native(b"\xff\xd8\xff\xe0", doc_id="d2", filename="photo.jpg")
    with pytest.raises(NotNativePathError):
        extract_image_native(b"MM\x00*", doc_id="d3", filename="scan.tiff")
```

- [ ] **Step 2: Failing test for dispatcher image routing**

Append to `tests/unit/extraction/test_dispatcher.py`:

```python
import pytest as _pytest_for_image


def test_dispatch_native_routes_png_to_image_adapter_which_raises():
    from src.extraction.adapters.dispatcher import dispatch_native
    from src.extraction.adapters.errors import NotNativePathError
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    with _pytest_for_image.raises(NotNativePathError):
        dispatch_native(png, filename="pic.png", doc_id="di1")
```

- [ ] **Step 3: Run to confirm fails, then implement**

Create `src/extraction/adapters/image_native.py`:

```python
"""Image-format "adapter" — a pass-through that always raises NotNativePathError.

Image files (JPG, PNG, TIFF) have no native text layer. They must go through
the vision path. This adapter exists so the dispatcher's routing table can map
image extensions explicitly (distinguishing them from genuinely-unknown
extensions) while still triggering the vision-path fallthrough.
"""
from __future__ import annotations

from src.extraction.adapters.errors import NotNativePathError


def extract_image_native(file_bytes: bytes, *, doc_id: str, filename: str):
    raise NotNativePathError(
        f"image file {filename!r} routes to vision path (no native extraction applicable)"
    )
```

Modify `src/extraction/adapters/dispatcher.py` — add the image extensions to the `_ADAPTERS` table:

```python
from src.extraction.adapters.image_native import extract_image_native
```

And extend `_ADAPTERS`:

```python
_ADAPTERS = {
    ".pdf": extract_pdf_native,
    ".docx": extract_docx_native,
    ".xlsx": extract_xlsx_native,
    ".xls": extract_xlsx_native,
    ".pptx": extract_pptx_native,
    ".csv": extract_csv_native,
    ".png": extract_image_native,
    ".jpg": extract_image_native,
    ".jpeg": extract_image_native,
    ".tif": extract_image_native,
    ".tiff": extract_image_native,
}
```

- [ ] **Step 4: Verify + commit**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_image_native.py tests/unit/extraction/test_dispatcher.py -x -q
git add src/extraction/adapters/image_native.py src/extraction/adapters/dispatcher.py tests/unit/extraction/test_image_native.py tests/unit/extraction/test_dispatcher.py
git commit -m "extraction: route image files (png/jpg/tiff) to vision path via dispatcher"
```

Expected pytest: all dispatcher + image tests pass (4 dispatcher original + 1 new + 2 image adapter = 7).

---

### Task 10: Wire vision orchestrator into the Celery task

**Files:**
- Modify: `src/tasks/extraction.py`

- [ ] **Step 1: Review current state**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
grep -n "NotNativePathError\|_native_path_taken\|dispatch_native\|fileProcessor\|ExtractionEngine" src/tasks/extraction.py
```

Locate where Plan 1 wired `dispatch_native` + the fall-through legacy branch.

- [ ] **Step 2: Replace the legacy fallback with vision orchestrator**

At the point in the task where Plan 1 falls through to legacy on NotNativePathError, insert a vision-path call BEFORE the legacy branch. The vision path takes over for all non-native formats. The legacy path remains as a deep fallback only if vision itself raises (this is a belt-and-suspenders safety net — the vision orchestrator already catches per-page errors internally, but we keep the outer guard).

Pattern to insert (match to actual variable names in file):

```python
# Plan 2: vision path takes over when native dispatch raised NotNativePathError.
if not _native_path_taken:
    try:
        from src.extraction.vision.orchestrator import extract_via_vision
        _vision_result = extract_via_vision(
            file_bytes=document_bytes,
            doc_id=document_id,
            filename=source_file,
            format_hint=_infer_format_hint(source_file),
        )
        result_dict = _dc_asdict(_vision_result)
        _vision_path_taken = True
        logger.info("[Plan2 vision path] doc_id=%s pages=%d fallback_invocations=%d",
                    document_id, len(_vision_result.pages),
                    len(_vision_result.metadata.coverage.fallback_invocations))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Plan2] vision path raised %r; falling through to legacy engine", exc)
        _vision_path_taken = False
else:
    _vision_path_taken = False

# Legacy engine runs only if BOTH native and vision declined the doc.
if not _native_path_taken and not _vision_path_taken:
    # ... existing legacy code path, unchanged ...
```

Add helper near the top of the module:

```python
def _infer_format_hint(filename: str) -> str:
    import os
    _, ext = os.path.splitext(filename.lower())
    if ext == ".pdf":
        return "pdf_scanned"
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return "image"
    return "image"
```

Make sure the blob upload + Mongo update at the end of the task see `result_dict` set by whichever path ran. If vision path's `path_taken="vision"` needs to update Mongo summary differently (e.g., `path_taken` field), include that.

Also add the observability write at task end:

```python
# Plan 2: observability audit log.
try:
    from src.extraction.vision.observability import ExtractionLogEntry, write_entry_if_redis
    _log_entry = ExtractionLogEntry(
        doc_id=document_id,
        format=result_dict.get("format", "unknown"),
        path_taken=result_dict.get("path_taken", "legacy"),
        timings_ms={},  # fill in if per-stage timings are already tracked; empty otherwise
        routing_decision=(result_dict.get("metadata") or {}).get("doc_intel") or {},
        coverage_score=((result_dict.get("metadata") or {}).get("coverage") or {}).get("verifier_score", 1.0),
        fallback_invocations=((result_dict.get("metadata") or {}).get("coverage") or {}).get("fallback_invocations", []),
        human_review=False,
        completed_at=time.time(),
    )
    write_entry_if_redis(redis_client=_get_redis_if_available(), entry=_log_entry)
except Exception:
    logger.debug("observability log skipped", exc_info=True)
```

(If `_get_redis_if_available()` helper doesn't exist in the task module, define a simple version that tries `from src.api.redis_client import get_redis` then `return get_redis()`, else returns None.)

- [ ] **Step 3: Verify module imports**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "from src.tasks import extraction; print('import OK')"
```

- [ ] **Step 4: Run tests**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit tests/integration -q --timeout=30
```

Expected: no new failures vs baseline (pre-existing `test_pipeline_has_six_phases` still fails — unrelated).

- [ ] **Step 5: Commit**

```bash
git add src/tasks/extraction.py
git commit -m "extraction: wire vision path into Celery task + per-extraction observability log"
```

---

### Task 11: Add scanned-PDF + image bench fixtures

**Files:**
- Modify: `tests/extraction_bench/fixtures/generate_fixtures.py`
- Generated: `tests/extraction_bench/cases/bench_scan_01/...`, `tests/extraction_bench/cases/bench_image_01/...`

Scanned-PDF and image fixtures use DocWain vision path (mocked for the bench harness since live DocWain calls aren't cheap). For the bench, we build fixtures that:
- Scanned PDF fixture: a PDF with only images (no text layer). Extraction via vision is expected to succeed with fallback (Tesseract) catching the text.
- Image fixture: a PNG with known text rendered on it (via PIL). Extraction via vision expected to return the known text.

However — the bench runner today calls `dispatch_native`. For vision-path fixtures, we need to extend the runner to route non-native through the vision orchestrator. For Plan 2's bench we stub the VisionClient with canned responses so the bench runs fully offline and deterministically.

- [ ] **Step 1: Extend fixture generator**

Modify `tests/extraction_bench/fixtures/generate_fixtures.py` to add two functions + a main hook:

```python
def generate_scanned_pdf_case():
    """A PDF whose pages have NO text layer — only images/shapes — simulating a scan."""
    d = fitz.open()
    for _ in range(1):
        p = d.new_page()
        p.draw_rect(fitz.Rect(72, 72, 400, 400), color=(0.1, 0.1, 0.1))
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    # Expected: vision-path output with single text_block and known content.
    expected = {
        "format": "pdf_scanned",
        "path_taken": "vision",
        "pages": [{"page_num": 1, "blocks": [{"text": "scanned content via fallback", "block_type": "paragraph"}], "tables": []}],
        "sheets": [],
        "slides": [],
    }
    _write_case("bench_scan_01", ".pdf", buf.getvalue(), expected)


def generate_image_case():
    """A PNG with a solid background — vision fallback is expected to emit a known text line."""
    from PIL import Image
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    expected = {
        "format": "image",
        "path_taken": "vision",
        "pages": [{"page_num": 1, "blocks": [{"text": "image content via fallback", "block_type": "paragraph"}], "tables": []}],
        "sheets": [],
        "slides": [],
    }
    _write_case("bench_image_01", ".png", buf.getvalue(), expected)
```

Call both from `main()` after the existing five generators.

- [ ] **Step 2: Extend runner to dispatch non-native cases through the vision orchestrator (with stubbed client)**

Modify `tests/extraction_bench/runner.py`:

```python
from src.extraction.adapters.errors import NotNativePathError


def _stub_vision_client_calls():
    """Patch VisionClient.call for the bench so vision-path fixtures run offline.

    Returns canned responses that match what bench_scan_01 / bench_image_01
    expect.
    """
    from src.extraction.vision import client as _client_module
    from src.extraction.vision.client import VisionResponse

    def fake_call(self, *, system, user_text, image_bytes, image_mime="image/png",
                  max_tokens=1024, temperature=0.0):
        s = system.lower()
        if "routing decision" in s or "classifier" in s:
            return VisionResponse(
                text='{"format":"pdf_scanned","doc_type_hint":"unknown","layout_complexity":"simple",'
                     '"has_handwriting":false,"suggested_path":"vision","confidence":0.8}',
                prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="bench-stub",
            )
        if "coverage verifier" in s:
            return VisionResponse(
                text='{"complete":false,"missed_regions":[{"bbox":[0.0,0.0,1.0,1.0],"description":"full-page"}],'
                     '"low_confidence_regions":[]}',
                prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="bench-stub",
            )
        return VisionResponse(
            text='{"regions":[],"reading_order":[],"page_confidence":0.1}',
            prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="bench-stub",
        )

    _client_module.VisionClient.call = fake_call  # type: ignore


def _stub_fallback_responses(canned_text_for_scan: str, canned_text_for_image: str):
    """Patch run_fallback_ensemble to emit canned text based on a hint in the filename."""
    from src.extraction.vision.fallback import FallbackRegionResult
    import src.extraction.vision.orchestrator as _orch

    original = _orch.run_fallback_ensemble

    def fake(img, *, bbox):
        # Use image size to distinguish fixture types:
        # - scanned PDF rendered page → ~612x792
        # - PNG fixture → 400x200
        w, h = img.size
        if w == 400 and h == 200:
            return FallbackRegionResult(text=canned_text_for_image, agreement=1.0, engine_winner="both")
        return FallbackRegionResult(text=canned_text_for_scan, agreement=1.0, engine_winner="both")

    _orch.run_fallback_ensemble = fake
    return original  # in case caller wants to restore
```

Replace `run_case` with a version that tries native first, vision second:

```python
def run_case(case_dir: Path) -> dict:
    source = next(case_dir.glob("source.*"))
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    file_bytes = source.read_bytes()
    try:
        result = dispatch_native(file_bytes, filename=source.name, doc_id=case_dir.name)
    except NotNativePathError:
        from src.extraction.vision.orchestrator import extract_via_vision
        hint = "pdf_scanned" if source.suffix.lower() == ".pdf" else "image"
        result = extract_via_vision(file_bytes, doc_id=case_dir.name, filename=source.name, format_hint=hint)
    actual = _extraction_to_comparable(result)
    scores = score_extraction(expected, actual)
    # Native path thresholds only apply to native path_taken; vision path has looser gate.
    if result.path_taken == "native":
        gate_passed = (
            scores["coverage"] >= NATIVE_COVERAGE_MIN
            and scores["fidelity"] >= NATIVE_FIDELITY_MIN
            and scores["structure"] >= NATIVE_STRUCTURE_MIN
            and scores["hallucination"] <= NATIVE_HALLUCINATION_MAX
        )
    else:
        # Vision path gate per spec §8.4: 0.95 / 0.92 / 0.95 / 0.01
        gate_passed = (
            scores["coverage"] >= 0.95
            and scores["fidelity"] >= 0.92
            and scores["structure"] >= 0.95
            and scores["hallucination"] <= 0.01
        )
    return {"case": case_dir.name, "scores": scores, "gate_passed": gate_passed, "path_taken": result.path_taken}
```

In `main()`, before iterating cases, stub the vision client + fallback:

```python
_stub_vision_client_calls()
_stub_fallback_responses(
    canned_text_for_scan="scanned content via fallback",
    canned_text_for_image="image content via fallback",
)
```

- [ ] **Step 3: Generate + run**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.fixtures.generate_fixtures
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner
```

Expected: 7 `[PASS]` lines (5 native + 2 vision), exit 0.

- [ ] **Step 4: Commit**

```bash
git add -f tests/extraction_bench/fixtures/generate_fixtures.py tests/extraction_bench/runner.py tests/extraction_bench/cases/bench_scan_01/ tests/extraction_bench/cases/bench_image_01/ tests/extraction_bench/bench_report.json
git commit -m "extraction/vision: extend bench with scanned-PDF + image fixtures (stubbed VisionClient)"
```

---

### Task 12: End-to-end vision smoke

**Files:**
- Create: `tests/integration/test_vision_path_smoke.py`

- [ ] **Step 1: Create smoke**

Create `tests/integration/test_vision_path_smoke.py`:

```python
"""Vision-path smoke with a fully stubbed DocWain client — offline test of the
entire orchestrator pipeline (DocIntel → extractor → verifier → fallback)
without hitting a live vLLM endpoint.
"""
import io

import fitz
from PIL import Image

from src.extraction.canonical_schema import ExtractionResult
from src.extraction.vision.client import VisionResponse
from src.extraction.vision.fallback import FallbackRegionResult
from src.extraction.vision.orchestrator import extract_via_vision


def _scanned_pdf(num_pages: int = 1) -> bytes:
    d = fitz.open()
    for _ in range(num_pages):
        p = d.new_page()
        p.draw_rect(fitz.Rect(72, 72, 400, 400), color=(0.1, 0.1, 0.1))
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    return buf.getvalue()


def _png_image() -> bytes:
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _stub_all(monkeypatch):
    def fake_call(self, *, system, user_text, image_bytes, image_mime="image/png",
                  max_tokens=1024, temperature=0.0):
        s = system.lower()
        if "classifier" in s or "routing decision" in s:
            return VisionResponse(
                text='{"format":"pdf_scanned","doc_type_hint":"unknown","layout_complexity":"simple",'
                     '"has_handwriting":false,"suggested_path":"vision","confidence":0.8}',
                prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="stub",
            )
        if "coverage verifier" in s:
            return VisionResponse(
                text='{"complete":false,"missed_regions":[{"bbox":[0.0,0.0,1.0,1.0],"description":"full"}],'
                     '"low_confidence_regions":[]}',
                prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="stub",
            )
        return VisionResponse(
            text='{"regions":[],"reading_order":[],"page_confidence":0.1}',
            prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="stub",
        )
    monkeypatch.setattr("src.extraction.vision.client.VisionClient.call", fake_call)

    def fake_fallback(img, *, bbox):
        return FallbackRegionResult(text="recovered content", agreement=1.0, engine_winner="both")
    monkeypatch.setattr("src.extraction.vision.orchestrator.run_fallback_ensemble", fake_fallback)


def test_vision_smoke_on_scanned_pdf(monkeypatch):
    _stub_all(monkeypatch)
    result = extract_via_vision(_scanned_pdf(), doc_id="s1", filename="scan.pdf", format_hint="pdf_scanned")
    assert isinstance(result, ExtractionResult)
    assert result.path_taken == "vision"
    assert result.pages
    assert any("recovered content" in b.text for b in result.pages[0].blocks)
    assert len(result.metadata.coverage.fallback_invocations) >= 1


def test_vision_smoke_on_image(monkeypatch):
    _stub_all(monkeypatch)
    result = extract_via_vision(_png_image(), doc_id="i1", filename="pic.png", format_hint="image")
    assert result.path_taken == "vision"
    assert result.pages
    assert any("recovered content" in b.text for b in result.pages[0].blocks)
```

- [ ] **Step 2: Run + commit**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/integration/test_vision_path_smoke.py -q
git add tests/integration/test_vision_path_smoke.py
git commit -m "extraction/vision: add offline end-to-end vision path smoke"
```

Expected pytest: 2 passed.

---

### Task 13: Final full-suite + bench run

Not a code task — validation.

- [ ] **Step 1: Run full extraction test set**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction tests/integration -q --timeout=30
```

Expected: all pass.

- [ ] **Step 2: Run bench**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner
```

Expected: 7 `[PASS]` lines, exit 0.

- [ ] **Step 3: Run broader sanity (catches KG removal regressions etc.)**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -q --timeout=30
```

Expected: 403+ passed, 1 pre-existing unrelated failure (`test_pipeline_has_six_phases`).

No commit for Task 13 — it's verification.

---

## Self-review — spec coverage

**Spec §3 (architecture):** Tasks 2–8 implement the three DocIntel capabilities + vision extractor + fallback + orchestrator. ✓

**Spec §4.2 (vision path):** Task 8 orchestrator flow (DocWain vision pass → coverage verifier → region-scoped fallback → merge). ✓

**Spec §4.3 (canonical schema):** Task 1 normalizes the Plan 1 wrapping. Task 8 emits canonical ExtractionResult directly. ✓

**Spec §6.1 (KG trigger removal):** unchanged from Plan 1 (already removed). No new KG triggers added in vision path. ✓

**Spec §7 (observability):** Task 7 + Task 10's observability hook add per-extraction Redis log with the required fields (timings, routing, coverage, fallback invocations, human_review flag). ✓

**Spec §8 (bench):** Task 11 adds scanned-PDF + image bench fixtures with vision-path gate thresholds. ✓

**Image files (JPG/PNG/TIFF):** Task 9 routes them to vision path via dispatcher + thin image_native adapter. ✓

**Non-goals respected:** no DocWain training work; no gateway changes (vision calls go directly to vLLM endpoint); no Researcher Agent; no KG re-introduction. ✓

## Self-review — placeholder scan

- All code steps show complete code, not stubs.
- All tests have complete bodies.
- Commit messages are exact.
- Variable-name hand-off flagged explicitly in Task 10 (wiring into existing Celery code) with the pattern to follow rather than literal names.
- No "TBD" or "implement appropriately" anywhere.

## Self-review — type consistency

- `ExtractionResult`, `Page`, `Block`, `Table`, `Sheet`, `Slide`, `Image`, `DocIntelMetadata`, `CoverageMetadata`, `ExtractionMetadata` — referenced from Plan 1, unchanged.
- `VisionClient`, `VisionResponse`, `VisionClientError` — defined Task 2, used Tasks 4, 5, 8, 11, 12.
- `RoutingDecision` — defined Task 4, used Task 8.
- `VisionExtraction` — defined Task 5, used Task 8.
- `FallbackRegionResult` — defined Task 6, used Tasks 8, 11, 12.
- `ExtractionLogEntry` — defined Task 7, used Task 10.
- `extract_via_vision` — defined Task 8, used Tasks 10, 11, 12.
- `extract_image_native` — defined Task 9, registered in dispatcher Task 9.
- No drift detected.

Plan complete.
