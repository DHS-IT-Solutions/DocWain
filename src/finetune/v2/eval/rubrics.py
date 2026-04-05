"""Programmatic scoring rubrics for all 6 V2 finetuning tracks.

Each scoring function takes a model response string and a reference dict,
returning a dict of dimension scores on a 1.0-5.0 scale.  All scoring is
fully programmatic (regex, JSON parsing, keyword detection, F1, structural
analysis) -- no LLM judge required.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 1.0, hi: float = 5.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _ratio_to_score(ratio: float) -> float:
    """Map a 0.0-1.0 ratio to a 1.0-5.0 score."""
    return _clamp(1.0 + ratio * 4.0)


def _f1(predicted: Set[str], expected: Set[str]) -> float:
    """Compute token-level F1 between two sets.  Returns 0.0-1.0."""
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    tp = len(predicted & expected)
    precision = tp / len(predicted)
    recall = tp / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_numbers(text: str) -> List[float]:
    """Pull all numeric values (including negatives / decimals) from text."""
    return [float(n) for n in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))]


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from response text."""
    # Fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Bare JSON
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _char_accuracy(predicted: str, expected: str) -> float:
    """Character-level accuracy between two strings (0.0-1.0)."""
    if not expected:
        return 1.0 if not predicted else 0.0
    return SequenceMatcher(None, _normalize(predicted), _normalize(expected)).ratio()


def _keyword_overlap(text: str, keywords: List[str]) -> float:
    """Fraction of keywords found in text (case-insensitive)."""
    if not keywords:
        return 1.0
    text_low = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_low) / len(keywords)


def _count_reasoning_indicators(text: str) -> int:
    """Count reasoning step indicators in text."""
    count = 0
    # Numbered steps
    count += len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+)", text, re.IGNORECASE))
    # Causal connectors
    for word in ("because", "therefore", "thus", "hence", "since",
                 "as a result", "consequently", "given that", "this means",
                 "it follows", "we can conclude", "this indicates"):
        count += len(re.findall(re.escape(word), text, re.IGNORECASE))
    return count


def _count_evidence_citations(text: str) -> int:
    """Count evidence citation patterns."""
    count = 0
    count += len(re.findall(r'"[^"]{5,}"', text))
    count += len(re.findall(
        r"\[(?:source|doc|page|section|para|ref|evidence)\s*[\d:,\- ]+\]",
        text, re.IGNORECASE,
    ))
    count += len(re.findall(
        r"(?:according to|as stated|as mentioned|per the|from the|the document states)",
        text, re.IGNORECASE,
    ))
    return count


# ---------------------------------------------------------------------------
# Track 1: Excel / CSV Intelligence
# ---------------------------------------------------------------------------

def score_excel_csv(response: str, reference: dict) -> dict:
    """Score Track 1: Excel/CSV intelligence.

    Reference keys used:
        expected_answer (str|number) -- the correct answer
        expected_values (dict)       -- key-value pairs the response should contain
        data_types (list[str])       -- expected data-type mentions
        aggregation (str)            -- expected aggregation result
        cross_sheet_entities (list[str]) -- entities linking across sheets

    Returns dict with four dimension scores (1.0-5.0):
        tabular_qa_accuracy, cross_sheet_reasoning,
        data_type_correctness, aggregation_accuracy
    """
    resp_norm = _normalize(response)

    # --- tabular_qa_accuracy ---
    expected_answer = reference.get("expected_answer", "")
    expected_values = reference.get("expected_values", {})

    qa_hits = 0.0
    qa_total = 0

    if expected_answer:
        qa_total += 1
        ea_str = str(expected_answer).lower().strip()
        if ea_str in resp_norm:
            qa_hits += 1
        else:
            # Numeric proximity check
            try:
                ea_num = float(str(expected_answer).replace(",", ""))
                resp_nums = _extract_numbers(response)
                if any(
                    abs(n - ea_num) / max(abs(ea_num), 1e-9) < 0.05
                    for n in resp_nums
                ):
                    qa_hits += 1
            except (ValueError, TypeError):
                pass

    if expected_values:
        for key, val in expected_values.items():
            qa_total += 1
            val_str = str(val).lower().strip()
            if val_str in resp_norm:
                qa_hits += 1
            elif key.lower() in resp_norm:
                qa_hits += 0.3  # partial: key present, value wrong

    tabular_qa = _ratio_to_score(qa_hits / max(qa_total, 1))

    # --- cross_sheet_reasoning ---
    cross_entities = reference.get("cross_sheet_entities", [])
    if cross_entities:
        entity_found = _keyword_overlap(response, cross_entities)
        cross_language = _keyword_overlap(response, [
            "across sheets", "from sheet", "in sheet", "between sheets",
            "cross-reference", "linked", "matching", "corresponding",
            "other sheet", "sheet 1", "sheet 2", "tab",
        ])
        cross_sheet = _ratio_to_score(
            0.6 * entity_found + 0.4 * min(cross_language * 3, 1.0)
        )
    else:
        # Dimension not tested — default to gate-level so it doesn't penalise
        cross_sheet = 4.0 if len(response) > 20 else 1.0

    # --- data_type_correctness ---
    data_types = reference.get("data_types", [])
    if data_types:
        data_type_score = _ratio_to_score(_keyword_overlap(response, data_types))
    else:
        # Dimension not tested — default to gate-level so it doesn't penalise
        data_type_score = 4.0 if len(response) > 20 else 1.0

    # --- aggregation_accuracy ---
    aggregation = reference.get("aggregation", "")
    if aggregation:
        agg_str = str(aggregation).lower().strip()
        if agg_str in resp_norm:
            agg_score = 5.0
        else:
            try:
                agg_num = float(str(aggregation).replace(",", ""))
                resp_nums = _extract_numbers(response)
                diffs = [
                    abs(n - agg_num) / max(abs(agg_num), 1e-9)
                    for n in resp_nums
                ]
                if diffs:
                    best = min(diffs)
                    if best < 0.01:
                        agg_score = 5.0
                    elif best < 0.05:
                        agg_score = 4.0
                    elif best < 0.15:
                        agg_score = 3.0
                    else:
                        agg_score = 1.5
                else:
                    agg_score = 1.0
            except (ValueError, TypeError):
                agg_score = 1.0
    else:
        # Dimension not tested — default to gate-level so it doesn't penalise
        agg_score = 4.0 if len(response) > 20 else 1.0

    return {
        "tabular_qa_accuracy": round(tabular_qa, 2),
        "cross_sheet_reasoning": round(cross_sheet, 2),
        "data_type_correctness": round(data_type_score, 2),
        "aggregation_accuracy": round(agg_score, 2),
    }


# ---------------------------------------------------------------------------
# Track 2: Layout Intelligence
# ---------------------------------------------------------------------------

def score_layout(response: str, reference: dict) -> dict:
    """Score Track 2: Layout intelligence.

    Reference keys used:
        expected_fields (list[str])        -- field names to extract
        expected_relationships (list[str]) -- relationship descriptions
        noise_tokens (list[str])           -- noise that should NOT appear
        completeness_fields (list[str])    -- full field set for completeness

    Returns dict with four dimension scores (1.0-5.0):
        structure_accuracy, relationship_extraction,
        noise_robustness, completeness_score
    """
    resp_norm = _normalize(response)

    # --- structure_accuracy ---
    expected_fields = reference.get("expected_fields", [])
    if expected_fields:
        found = sum(1 for f in expected_fields if f.lower() in resp_norm)
        structure_accuracy = _ratio_to_score(found / len(expected_fields))
    else:
        structure_accuracy = 4.0

    # --- relationship_extraction ---
    expected_rels = reference.get("expected_relationships", [])
    if expected_rels:
        rel_found = sum(1 for r in expected_rels if r.lower() in resp_norm)
        relationship_extraction = _ratio_to_score(rel_found / len(expected_rels))
    else:
        relationship_extraction = 4.0

    # --- noise_robustness ---
    noise_tokens = reference.get("noise_tokens", [])
    if noise_tokens:
        noise_present = sum(1 for t in noise_tokens if t.lower() in resp_norm)
        noise_robustness = _ratio_to_score(1.0 - noise_present / len(noise_tokens))
    else:
        noise_robustness = 4.0

    # --- completeness_score ---
    comp_fields = reference.get("completeness_fields",
                                reference.get("expected_fields", []))
    if comp_fields:
        comp_found = sum(1 for f in comp_fields if f.lower() in resp_norm)
        completeness_score = _ratio_to_score(comp_found / len(comp_fields))
    else:
        completeness_score = 4.0

    return {
        "structure_accuracy": round(structure_accuracy, 2),
        "relationship_extraction": round(relationship_extraction, 2),
        "noise_robustness": round(noise_robustness, 2),
        "completeness_score": round(completeness_score, 2),
    }


# ---------------------------------------------------------------------------
# Track 3: OCR & Vision
# ---------------------------------------------------------------------------

def score_ocr_vision(response: str, reference: dict) -> dict:
    """Score Track 3: OCR & Vision.

    Reference keys used:
        printed_text (str)            -- expected printed text extraction
        handwritten_text (str)        -- expected handwritten text extraction
        diagram_elements (list[str])  -- expected diagram element descriptions
        table_data (list[list[str]])  -- expected table rows
        overlay_text (str)            -- expected text from overlay regions

    Returns dict with five dimension scores (1.0-5.0):
        printed_accuracy, handwriting_accuracy, diagram_understanding,
        image_table_reconstruction, overlay_handling
    """
    # --- printed_accuracy ---
    printed = reference.get("printed_text", "")
    if printed:
        printed_accuracy = _ratio_to_score(_char_accuracy(response, printed))
    else:
        printed_accuracy = 4.0

    # --- handwriting_accuracy ---
    handwritten = reference.get("handwritten_text", "")
    if handwritten:
        handwriting_accuracy = _ratio_to_score(_char_accuracy(response, handwritten))
    else:
        handwriting_accuracy = 4.0

    # --- diagram_understanding ---
    diagram_elements = reference.get("diagram_elements", [])
    if diagram_elements:
        diagram_understanding = _ratio_to_score(
            _keyword_overlap(response, diagram_elements)
        )
    else:
        diagram_understanding = 4.0

    # --- image_table_reconstruction ---
    table_data = reference.get("table_data", [])
    if table_data:
        total_cells = 0
        matched_cells = 0
        r_norm = _normalize(response)
        for row in table_data:
            for cell in row:
                total_cells += 1
                if _normalize(str(cell)) in r_norm:
                    matched_cells += 1
        table_ratio = matched_cells / max(total_cells, 1)
        # Bonus for markdown table formatting
        if re.search(r"^\|.+\|", response, re.MULTILINE):
            table_ratio = min(1.0, table_ratio + 0.1)
        image_table_reconstruction = _ratio_to_score(table_ratio)
    else:
        image_table_reconstruction = 4.0

    # --- overlay_handling ---
    overlay = reference.get("overlay_text", "")
    if overlay:
        overlay_handling = _ratio_to_score(_char_accuracy(response, overlay))
    else:
        overlay_handling = 4.0

    return {
        "printed_accuracy": round(printed_accuracy, 2),
        "handwriting_accuracy": round(handwriting_accuracy, 2),
        "diagram_understanding": round(diagram_understanding, 2),
        "image_table_reconstruction": round(image_table_reconstruction, 2),
        "overlay_handling": round(overlay_handling, 2),
    }


# ---------------------------------------------------------------------------
# Track 4: Context & Reasoning
# ---------------------------------------------------------------------------

def score_reasoning(response: str, reference: dict) -> dict:
    """Score Track 4: Context & Reasoning.

    Reference keys used:
        expected_conclusion (str)  -- the correct final answer / conclusion
        key_evidence (list[str])   -- evidence items to cite
        reasoning_steps (int)      -- expected minimum reasoning steps
        key_terms (list[str])      -- domain terms that should appear

    Returns dict with three dimension scores (1.0-5.0):
        reasoning_depth, evidence_grounding, synthesis_coherence
    """
    # --- reasoning_depth ---
    expected_steps = reference.get("reasoning_steps", 3)
    actual_indicators = _count_reasoning_indicators(response)
    step_ratio = min(actual_indicators / max(expected_steps, 1), 1.5)
    word_count = len(response.split())
    length_bonus = min(word_count / 200.0, 0.3)
    reasoning_depth = _ratio_to_score(min(step_ratio * 0.7 + length_bonus, 1.0))

    # --- evidence_grounding ---
    key_evidence = reference.get("key_evidence", [])
    if key_evidence:
        evidence_overlap = _keyword_overlap(response, key_evidence)
        citation_count = _count_evidence_citations(response)
        citation_bonus = min(citation_count / max(len(key_evidence), 1), 1.0) * 0.3
        evidence_grounding = _ratio_to_score(
            min(evidence_overlap * 0.7 + citation_bonus, 1.0)
        )
    else:
        citation_count = _count_evidence_citations(response)
        evidence_grounding = _ratio_to_score(min(citation_count / 3.0, 1.0))

    # --- synthesis_coherence ---
    expected_conclusion = reference.get("expected_conclusion", "")
    key_terms = reference.get("key_terms", [])

    coherence = 0.0
    checks = 0

    _STOP_WORDS = frozenset(
        "the a an is are was were of in to and or for on at by with".split()
    )

    if expected_conclusion:
        checks += 1
        conc_words = set(_normalize(expected_conclusion).split()) - _STOP_WORDS
        resp_words = set(_normalize(response).split())
        if conc_words:
            coherence += len(conc_words & resp_words) / len(conc_words)

    if key_terms:
        checks += 1
        coherence += _keyword_overlap(response, key_terms)

    if checks > 0:
        synthesis_coherence = _ratio_to_score(coherence / checks)
    else:
        has_structure = bool(re.search(r"(?:^|\n)\s*\d+[\.\)]", response))
        synthesis_coherence = 4.0 if has_structure else 3.0

    return {
        "reasoning_depth": round(reasoning_depth, 2),
        "evidence_grounding": round(evidence_grounding, 2),
        "synthesis_coherence": round(synthesis_coherence, 2),
    }


# ---------------------------------------------------------------------------
# Track 5: KG-Augmented
# ---------------------------------------------------------------------------

def score_kg(response: str, reference: dict) -> dict:
    """Score Track 5: KG-Augmented.

    Reference keys used:
        expected_entities (list[str])      -- entity names/IDs to reference
        expected_relationships (list[str]) -- relationship descriptions
        expected_citations (list[str])     -- source doc / entity citations

    Returns dict with three dimension scores (1.0-5.0):
        entity_usage, relationship_reasoning, citation_accuracy
    """
    resp_norm = _normalize(response)

    _STOP_WORDS = frozenset(
        "the a an is are was were of in to and or for on at by with".split()
    )

    # --- entity_usage ---
    expected_entities = reference.get("expected_entities", [])
    if expected_entities:
        entity_found = sum(1 for e in expected_entities if e.lower() in resp_norm)
        entity_usage = _ratio_to_score(entity_found / len(expected_entities))
    else:
        entity_usage = 4.0

    # --- relationship_reasoning ---
    expected_rels = reference.get("expected_relationships", [])
    if expected_rels:
        rel_hits = 0.0
        for rel in expected_rels:
            rel_words = set(_normalize(rel).split()) - _STOP_WORDS
            if rel_words:
                overlap = len(rel_words & set(resp_norm.split())) / len(rel_words)
                if overlap >= 0.6:
                    rel_hits += 1
                elif overlap >= 0.3:
                    rel_hits += 0.5
        relationship_reasoning = _ratio_to_score(rel_hits / len(expected_rels))
    else:
        relationship_reasoning = 4.0

    # --- citation_accuracy ---
    expected_citations = reference.get("expected_citations", [])
    bracket_cites = len(re.findall(
        r"\[(?:ENT|REL|DOC|SRC|KG)[-_]?\d*\]", response, re.IGNORECASE
    ))
    if expected_citations:
        cite_found = sum(1 for c in expected_citations if c.lower() in resp_norm)
        cite_ratio = cite_found / len(expected_citations)
        cite_bonus = min(bracket_cites / max(len(expected_citations), 1), 0.3)
        citation_accuracy = _ratio_to_score(min(cite_ratio + cite_bonus, 1.0))
    else:
        citation_accuracy = _ratio_to_score(min(bracket_cites / 3.0, 1.0))

    return {
        "entity_usage": round(entity_usage, 2),
        "relationship_reasoning": round(relationship_reasoning, 2),
        "citation_accuracy": round(citation_accuracy, 2),
    }


# ---------------------------------------------------------------------------
# Track 6: Visualization
# ---------------------------------------------------------------------------

_VIZ_DIRECTIVE_RE = re.compile(r"<!--DOCWAIN_VIZ\s*(\{.*?\})\s*-->", re.DOTALL)

_VALID_CHART_TYPES = frozenset({
    "bar", "horizontal_bar", "line", "area", "pie", "donut",
    "scatter", "radar", "waterfall", "treemap", "gauge",
    "grouped_bar", "stacked_bar", "multi_line", "heatmap", "funnel",
})

_COMPATIBLE_VIZ_GROUPS: list[frozenset[str]] = [
    frozenset({"bar", "horizontal_bar"}),
    frozenset({"donut", "pie"}),
    frozenset({"line", "area"}),
    frozenset({"grouped_bar", "stacked_bar"}),
    frozenset({"line", "multi_line"}),
]


def score_visualization(response: str, reference: dict) -> dict:
    """Score Track 6: Visualization.

    Reference keys used:
        expects_chart (bool)           -- whether a chart directive is expected
        expected_chart_type (str)      -- the chart type
        expected_labels (list[str])    -- data labels
        expected_values (list[float])  -- data values

    Returns dict with four dimension scores (1.0-5.0):
        trigger_judgment, spec_correctness, data_accuracy, type_selection
    """
    expects_chart = reference.get("expects_chart", True)
    expected_type = reference.get("expected_chart_type", "")
    expected_labels = reference.get("expected_labels", [])
    expected_values = reference.get("expected_values", [])

    viz_match = _VIZ_DIRECTIVE_RE.search(response)
    has_directive = viz_match is not None
    parsed_spec: Optional[Dict[str, Any]] = None

    if viz_match:
        try:
            parsed_spec = json.loads(viz_match.group(1))
            if parsed_spec and "chart_type" not in parsed_spec:
                for alias in ("type", "chart"):
                    if alias in parsed_spec:
                        parsed_spec["chart_type"] = parsed_spec.pop(alias)
                        break
        except json.JSONDecodeError:
            parsed_spec = None

    # --- trigger_judgment ---
    if expects_chart:
        trigger_judgment = 5.0 if has_directive else 1.0
    else:
        trigger_judgment = 5.0 if not has_directive else 1.5

    # --- spec_correctness ---
    if expects_chart:
        if parsed_spec is not None:
            has_type = "chart_type" in parsed_spec
            has_data = bool(
                parsed_spec.get("labels")
                or parsed_spec.get("values")
                or parsed_spec.get("data")
            )
            if has_type and has_data:
                spec_correctness = 5.0
            elif has_type:
                spec_correctness = 3.5
            else:
                spec_correctness = 2.0
        elif has_directive:
            spec_correctness = 1.5  # directive present but bad JSON
        else:
            spec_correctness = 1.0
    else:
        spec_correctness = 5.0 if not has_directive else 2.0

    # --- data_accuracy ---
    if expects_chart and parsed_spec:
        label_score = 0.0
        value_score = 0.0
        checks = 0

        if expected_labels:
            checks += 1
            actual_labels = {str(l).lower() for l in parsed_spec.get("labels", [])}
            expected_set = {l.lower() for l in expected_labels}
            if expected_set and actual_labels:
                overlap = len(expected_set & actual_labels)
                label_score = overlap / max(len(expected_set), len(actual_labels))
            elif not expected_set and not actual_labels:
                label_score = 1.0

        if expected_values:
            checks += 1
            actual_values = parsed_spec.get("values", [])
            if actual_values and len(actual_values) == len(expected_values):
                matches = 0
                for av, ev in zip(actual_values, expected_values):
                    try:
                        if abs(float(av) - float(ev)) < 0.01 * max(abs(float(ev)), 1):
                            matches += 1
                    except (ValueError, TypeError):
                        pass
                value_score = matches / len(expected_values)
            elif actual_values:
                value_score = 0.3

        if checks > 0:
            data_accuracy = _ratio_to_score((label_score + value_score) / checks)
        else:
            data_accuracy = (
                3.5
                if parsed_spec.get("labels") or parsed_spec.get("values")
                else 2.0
            )
    elif expects_chart:
        data_accuracy = 1.0
    else:
        data_accuracy = 5.0 if not has_directive else 2.0

    # --- type_selection ---
    if expects_chart and parsed_spec and expected_type:
        actual_type = parsed_spec.get("chart_type", "")
        if actual_type == expected_type:
            type_selection = 5.0
        elif any(
            actual_type in g and expected_type in g
            for g in _COMPATIBLE_VIZ_GROUPS
        ):
            type_selection = 3.5
        elif actual_type in _VALID_CHART_TYPES:
            type_selection = 2.0
        else:
            type_selection = 1.0
    elif expects_chart:
        type_selection = 1.0
    else:
        type_selection = 5.0 if not has_directive else 2.0

    return {
        "trigger_judgment": round(trigger_judgment, 2),
        "spec_correctness": round(spec_correctness, 2),
        "data_accuracy": round(data_accuracy, 2),
        "type_selection": round(type_selection, 2),
    }


# ---------------------------------------------------------------------------
# Track name -> scorer mapping
# ---------------------------------------------------------------------------

TRACK_SCORERS = {
    "excel_csv": score_excel_csv,
    "layout": score_layout,
    "ocr_vision": score_ocr_vision,
    "reasoning": score_reasoning,
    "kg": score_kg,
    "visualization": score_visualization,
}
