"""Curriculum data generation engine for DocWain V2 training pipeline.

Builds GenerationBrief objects that drive subagent data synthesis, and
provides utilities to parse, validate, and merge generated JSONL datasets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.finetune.v2.data_generator.base import DOCWAIN_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Area configuration
# ---------------------------------------------------------------------------

AREA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "excel_csv": {
        "initial_count": 900,
        "description": (
            "Spreadsheet and CSV document intelligence: tabular Q&A, "
            "aggregation, cross-sheet reasoning, formula interpretation, "
            "and anomaly detection in structured numerical data."
        ),
        "categories": [
            "tabular_qa",
            "multi_sheet",
            "aggregation",
            "formula_interpretation",
            "anomaly_detection",
            "trend_analysis",
            "pivot_summary",
        ],
        "difficulty_split": {"easy": 0.20, "medium": 0.50, "hard": 0.30},
    },
    "layout": {
        "initial_count": 800,
        "description": (
            "Document layout understanding: reading order, section extraction, "
            "header/footer detection, table-of-contents parsing, and "
            "multi-column document comprehension."
        ),
        "categories": [
            "reading_order",
            "section_extraction",
            "header_footer",
            "toc_parsing",
            "multi_column",
            "bounding_box_qa",
            "form_field_extraction",
        ],
        "difficulty_split": {"easy": 0.25, "medium": 0.50, "hard": 0.25},
    },
    "ocr_vision": {
        "initial_count": 800,
        "description": (
            "OCR and vision grounding: transcribing degraded text, correcting "
            "OCR errors, answering questions about scanned documents, chart "
            "image interpretation, and handwriting recognition."
        ),
        "categories": [
            "transcription",
            "ocr_correction",
            "scanned_qa",
            "chart_image_qa",
            "handwriting",
            "mixed_media",
        ],
        "difficulty_split": {"easy": 0.20, "medium": 0.50, "hard": 0.30},
    },
    "reasoning": {
        "initial_count": 900,
        "description": (
            "Multi-hop reasoning over documents: evidence grounding, "
            "contradictory claim detection, comparative analysis across "
            "multiple documents, temporal reasoning, and causal inference."
        ),
        "categories": [
            "multi_hop",
            "evidence_grounding",
            "contradiction_detection",
            "comparative_analysis",
            "temporal_reasoning",
            "causal_inference",
            "numeric_reasoning",
        ],
        "difficulty_split": {"easy": 0.15, "medium": 0.45, "hard": 0.40},
    },
    "kg": {
        "initial_count": 800,
        "description": (
            "Knowledge graph extraction and querying: entity extraction, "
            "relationship identification, graph traversal Q&A, entity "
            "disambiguation, and ontology-aware reasoning."
        ),
        "categories": [
            "entity_extraction",
            "relationship_identification",
            "graph_traversal_qa",
            "entity_disambiguation",
            "ontology_reasoning",
            "co_reference_resolution",
        ],
        "difficulty_split": {"easy": 0.20, "medium": 0.50, "hard": 0.30},
    },
    "visualization": {
        "initial_count": 900,
        "description": (
            "Data visualization generation and interpretation: producing "
            "chart specifications from data, interpreting existing charts, "
            "recommending visualization types, and explaining visual insights."
        ),
        "categories": [
            "chart_spec_generation",
            "chart_interpretation",
            "viz_recommendation",
            "insight_explanation",
            "dashboard_summary",
            "comparative_viz",
        ],
        "difficulty_split": {"easy": 0.20, "medium": 0.50, "hard": 0.30},
    },
}

# Validate counts at import time
_TOTAL = sum(cfg["initial_count"] for cfg in AREA_CONFIGS.values())
assert _TOTAL == 5100, f"AREA_CONFIGS initial counts must sum to 5100, got {_TOTAL}"

# ---------------------------------------------------------------------------
# GenerationBrief
# ---------------------------------------------------------------------------

_VALID_AREAS = set(AREA_CONFIGS.keys())
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class GenerationBrief:
    """Specification sent to a data-generation subagent for one capability area."""

    area: str
    count: int
    difficulty_split: Dict[str, float]
    categories: List[str]
    focus_instructions: str
    iteration: int

    def to_prompt(self) -> str:
        """Build a full subagent prompt for synthetic data generation."""
        cfg = AREA_CONFIGS.get(self.area, {})
        description = cfg.get("description", self.area)

        easy_n = round(self.count * self.difficulty_split.get("easy", 0.20))
        medium_n = round(self.count * self.difficulty_split.get("medium", 0.50))
        hard_n = self.count - easy_n - medium_n

        categories_str = "\n".join(f"  - {c}" for c in self.categories)
        focus_block = (
            f"\n## Focus Instructions (Iteration {self.iteration})\n{self.focus_instructions}\n"
            if self.focus_instructions.strip()
            else ""
        )

        return f"""You are a specialist data-generation agent for the DocWain V2 training pipeline.

## DocWain System Prompt (for reference — embed verbatim in every example)
{DOCWAIN_SYSTEM_PROMPT}

## Task
Generate exactly {self.count} high-quality synthetic training examples for the **{self.area}** capability area.

## Area Description
{description}

## Categories to cover
{categories_str}

## Difficulty Distribution
Generate approximately:
- Easy: {easy_n} examples ({int(self.difficulty_split.get('easy', 0.20)*100)}%)
- Medium: {medium_n} examples ({int(self.difficulty_split.get('medium', 0.50)*100)}%)
- Hard: {hard_n} examples ({int(self.difficulty_split.get('hard', 0.30)*100)}%)
{focus_block}
## Output Format
Output one JSON object per line (JSONL). Each line must be valid JSON with these fields:
  - "text": full conversation in Qwen3 chat template format (see below)
  - "area": "{self.area}"
  - "difficulty": one of "easy", "medium", "hard"
  - "category": one of the categories listed above

## Qwen3 Chat Template Format
Every "text" value must follow this exact structure:

```
<|im_start|>system
{DOCWAIN_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
[document context in markdown/XML + question here]<|im_end|>
<|im_start|>assistant
<think>
[step-by-step reasoning here — at least 3 sentences for easy, 5+ for medium/hard]
</think>

[final answer here, citing evidence]<|im_end|>
```

## Document Context Format
- Spreadsheet/tabular data: use markdown tables preceded by a `[SPREADSHEET: filename.xlsx]` header
- Multi-sheet: use separate tables with `### Sheet: <name>` headings
- KG context: use `<kg_context>` XML blocks listing entities and relationships
- Scanned/OCR documents: embed raw OCR text in `<ocr_text source="...">` tags
- Charts/visualisation: describe chart spec in JSON inside `<chart_spec>` tags

## Quality Requirements
1. **Realistic data** — use plausible domain-appropriate values (financial figures, names, dates)
2. **Natural questions** — questions an enterprise user would actually ask
3. **Deep reasoning** — the `<think>` block must show genuine step-by-step inference
4. **Diversity** — vary domains ({', '.join(['legal', 'financial', 'hr', 'medical', 'policy'])}),
   document types, question styles, and answer lengths
5. **Evidence grounding** — answers must cite specific cells, rows, entities, or passages
6. **No hallucination** — every claim in the answer must be derivable from the document context

## Output Instructions
- Output ONLY raw JSONL — no markdown fencing, no commentary, no preamble
- One valid JSON object per line, exactly {self.count} lines total
- Ensure every "text" value contains `<|im_start|>`, `<|im_end|>`, `<think>`, and `</think>` tokens
"""


# ---------------------------------------------------------------------------
# Brief builders
# ---------------------------------------------------------------------------


def build_initial_briefs() -> List[GenerationBrief]:
    """Return one GenerationBrief per capability area using initial counts."""
    briefs: List[GenerationBrief] = []
    for area, cfg in AREA_CONFIGS.items():
        briefs.append(
            GenerationBrief(
                area=area,
                count=cfg["initial_count"],
                difficulty_split=cfg["difficulty_split"],
                categories=list(cfg["categories"]),
                focus_instructions="",
                iteration=1,
            )
        )
    return briefs


def build_augmentation_briefs(
    failure_analysis: Dict[str, Any],
    iteration: int,
) -> List[GenerationBrief]:
    """Build targeted GenerationBriefs from an eval failure analysis dict.

    The failure_analysis dict must have:
      - "weak_areas": list of {area, dimension, avg_score, failure_patterns}
      - "total_augmentation_count": int — total examples to generate across all areas

    Count is distributed proportionally to the gap from 4.0 (larger gap = more examples).
    Augmentation uses a harder difficulty split: 10% easy, 40% medium, 50% hard.
    """
    weak_areas: List[Dict[str, Any]] = failure_analysis.get("weak_areas", [])
    total_count: int = int(failure_analysis.get("total_augmentation_count", 500))

    if not weak_areas:
        return []

    # Compute gap from 4.0 for each weak area
    TARGET_SCORE = 4.0
    gaps = [max(0.0, TARGET_SCORE - float(w.get("avg_score", 0.0))) for w in weak_areas]
    total_gap = sum(gaps) or 1.0

    augmentation_split = {"easy": 0.10, "medium": 0.40, "hard": 0.50}

    briefs: List[GenerationBrief] = []
    allocated = 0
    for idx, (weak, gap) in enumerate(zip(weak_areas, gaps)):
        area = weak["area"]
        dimension = weak.get("dimension", "")
        patterns = weak.get("failure_patterns", [])

        # Last item gets remainder to avoid off-by-one from rounding
        if idx == len(weak_areas) - 1:
            count = total_count - allocated
        else:
            count = round(total_count * (gap / total_gap))
        allocated += count

        # Build focus instructions from dimension and patterns
        focus_parts: List[str] = []
        if dimension:
            focus_parts.append(f"Focus on improving {dimension}.")
        if patterns:
            focus_parts.append(
                "Known failure patterns to address:\n"
                + "\n".join(f"  - {p}" for p in patterns)
            )

        categories = list(AREA_CONFIGS.get(area, {}).get("categories", [dimension]))

        briefs.append(
            GenerationBrief(
                area=area,
                count=max(1, count),
                difficulty_split=augmentation_split,
                categories=categories,
                focus_instructions="\n".join(focus_parts),
                iteration=iteration,
            )
        )

    return briefs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_example(example: Dict[str, Any]) -> bool:
    """Return True if example passes structural quality checks.

    Checks:
      - "text" field present and contains required Qwen3 chat tokens
      - "area" is one of the six known areas
      - "difficulty" is easy/medium/hard
    """
    text = example.get("text", "")
    if not isinstance(text, str) or not text:
        return False

    # Must have Qwen3 chat template tokens
    required_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    for token in required_tokens:
        if token not in text:
            return False

    # Must have a valid area
    area = example.get("area", "")
    if area not in _VALID_AREAS:
        return False

    # Must have a valid difficulty
    difficulty = example.get("difficulty", "")
    if difficulty not in _VALID_DIFFICULTIES:
        return False

    return True


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_generated_examples(raw_text: str) -> List[Dict[str, Any]]:
    """Parse JSONL output from a subagent, skipping markdown fencing and invalid lines.

    Returns only examples that pass validate_example().
    """
    examples: List[Dict[str, Any]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip markdown code fences
        if line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if validate_example(obj):
            examples.append(obj)
    return examples


# ---------------------------------------------------------------------------
# Dataset merging
# ---------------------------------------------------------------------------


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def merge_datasets(
    source_files: List[Path],
    output_path: Path,
    max_size: int = 15000,
) -> int:
    """Merge multiple JSONL files, deduplicate by SHA-256 of the text field,
    cap at max_size (keeping most recent — later files take precedence),
    and write the result to output_path.

    Returns the number of examples written.
    """
    # Process files in order; later entries overwrite earlier ones for same hash
    # (dict insertion order preserved in Python 3.7+, so "most recent" = last seen)
    seen: Dict[str, Dict[str, Any]] = {}

    for src in source_files:
        src = Path(src)
        if not src.exists():
            continue
        with open(src, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                text = obj.get("text", "")
                key = _sha256_of_text(text)
                seen[key] = obj  # overwrite duplicates; last file wins

    # Cap at max_size keeping most recent (end of dict)
    all_records = list(seen.values())
    if len(all_records) > max_size:
        all_records = all_records[-max_size:]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(all_records)
