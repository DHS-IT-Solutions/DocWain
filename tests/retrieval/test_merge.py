"""Phase 3 Task 4 — merge / rerank / MMR over PackedItem."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.merge import (
    merge_layers,
    mmr_select,
    rerank_merged_candidates,
)
from src.retrieval.types import PackedItem, RetrievalBundle


# ---------------------------------------------------------------------------
# merge_layers
# ---------------------------------------------------------------------------


def _mk_chunk(doc_id: str, chunk_id: str, text: str = "t", score: float = 0.8) -> dict:
    return {
        "kind": "chunk",
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "confidence": score,
    }


def _mk_kg_direct(src: str, dst: str, rel: str = "cites") -> dict:
    return {
        "kind": "kg_direct",
        "src": src,
        "dst": dst,
        "type": rel,
        "evidence": [],
        "confidence": 0.9,
    }


def _mk_kg_inferred(src: str, dst: str, rel: str = "correlates_with") -> dict:
    return {
        "kind": "kg_inferred",
        "src": src,
        "dst": dst,
        "relation_type": rel,
        "confidence": 0.82,
        "evidence": [],
        "inference_path": [],
    }


def _mk_sme(snippet_id: str, text: str = "SME claim", atype: str = "dossier") -> dict:
    return {
        "kind": "sme_artifact",
        "snippet_id": snippet_id,
        "artifact_type": atype,
        "text": text,
        "confidence": 0.9,
        "evidence": ["d1#c1"],
    }


def test_merge_preserves_all_items_when_unique() -> None:
    a = [_mk_chunk("d1", "c1"), _mk_chunk("d2", "c2")]
    b = [_mk_kg_direct("n1", "n2")]
    c = [_mk_sme("s1")]
    merged = merge_layers({"a": a, "b": b, "c": c, "d": []})
    assert len(merged) == 4


def test_merge_dedups_on_doc_chunk_key() -> None:
    a = [_mk_chunk("d1", "c1", text="A")]
    c = [
        # Same (doc, chunk) key — SME variant wins the sme_backed flag.
        {
            "kind": "sme_artifact",
            "doc_id": "d1",
            "chunk_id": "c1",
            "text": "A",
            "confidence": 0.85,
            "artifact_type": "dossier",
        }
    ]
    merged = merge_layers({"a": a, "b": [], "c": c, "d": []})
    assert len(merged) == 1
    assert merged[0].sme_backed is True


def test_merge_marks_kg_inferred_as_sme_backed() -> None:
    b = [_mk_kg_direct("n1", "n2"), _mk_kg_inferred("n1", "n3")]
    merged = merge_layers({"a": [], "b": b, "c": [], "d": []})
    # Two distinct KG rows survive (different relation types).
    by_rel = {p.metadata.get("relation_type"): p for p in merged}
    assert by_rel["cites"].sme_backed is False
    assert by_rel["correlates_with"].sme_backed is True


def test_merge_layer_c_always_sme_backed() -> None:
    c = [_mk_sme("s1"), _mk_sme("s2")]
    merged = merge_layers({"a": [], "b": [], "c": c, "d": []})
    assert all(p.sme_backed is True for p in merged)
    assert all(p.layer == "c" for p in merged)


def test_merge_layer_a_never_sme_backed_by_default() -> None:
    merged = merge_layers({"a": [_mk_chunk("d1", "c1")], "b": [], "c": [], "d": []})
    assert merged[0].sme_backed is False
    assert merged[0].layer == "a"


def test_merge_kg_direct_never_sme_backed() -> None:
    merged = merge_layers({"a": [], "b": [_mk_kg_direct("n1", "n2")], "c": [], "d": []})
    assert merged[0].sme_backed is False


def test_merge_accepts_retrieval_bundle() -> None:
    bundle = RetrievalBundle(
        layer_a_chunks=[_mk_chunk("d1", "c1")],
        layer_c_sme=[_mk_sme("s1")],
    )
    merged = merge_layers(bundle)
    assert len(merged) == 2
    assert any(p.layer == "a" for p in merged)
    assert any(p.layer == "c" and p.sme_backed for p in merged)


def test_merge_provenance_populated_from_evidence_refs() -> None:
    c = [
        {
            "kind": "sme_artifact",
            "snippet_id": "s1",
            "text": "x",
            "confidence": 0.9,
            "evidence": ["d1#c1", "d2#c2"],
        }
    ]
    merged = merge_layers({"a": [], "b": [], "c": c, "d": []})
    assert merged[0].provenance == (("d1", "c1"), ("d2", "c2"))


def test_merge_degraded_double_append_not_present() -> None:
    """ERRATA §11 — no short-name 'c' in degraded list."""
    bundle = RetrievalBundle(degraded_layers=["layer_c"])
    assert "c" not in bundle.degraded_layers


# ---------------------------------------------------------------------------
# rerank_merged_candidates
# ---------------------------------------------------------------------------


def _p(text: str, layer: str = "a", *, conf: float = 0.8, rr: float = 0.5,
       sme: bool = False, meta: dict | None = None) -> PackedItem:
    return PackedItem(
        text=text,
        provenance=(("d", "c"),),
        layer=layer,
        confidence=conf,
        rerank_score=rr,
        sme_backed=sme,
        metadata=meta or {},
    )


def test_cross_encoder_reshuffles_top_k() -> None:
    ce = MagicMock()
    # CE says the third candidate is best; the others are worse.
    ce.predict.return_value = [0.1, 0.4, 0.9]
    cands = [
        _p("irrelevant", rr=0.95),
        _p("middling", rr=0.80),
        _p("very relevant", rr=0.60),
    ]
    ranked = rerank_merged_candidates(
        "q", cands, cross_encoder=ce, top_k=3, intent="lookup"
    )
    assert ranked[0].text == "very relevant"
    assert ranked[-1].text == "irrelevant"


def test_cross_encoder_bypassed_when_flag_off() -> None:
    ce = MagicMock()
    cands = [_p("a", rr=0.9), _p("b", rr=0.8)]
    ranked = rerank_merged_candidates(
        "q", cands, cross_encoder=ce, top_k=2, enable_cross_encoder=False
    )
    ce.predict.assert_not_called()
    assert [r.text for r in ranked] == ["a", "b"]


def test_cross_encoder_none_is_no_op() -> None:
    cands = [_p("a", rr=0.9), _p("b", rr=0.8)]
    ranked = rerank_merged_candidates(
        "q", cands, cross_encoder=None, top_k=2
    )
    assert [r.text for r in ranked] == ["a", "b"]


def test_sme_layer_gets_bonus_on_analytical_intent() -> None:
    ce = MagicMock()
    # Tied CE → bonus wins for SME-backed on an analytical intent.
    ce.predict.return_value = [0.7, 0.7]
    cands = [
        _p("chunk", layer="a", rr=0.9, conf=0.9, sme=False),
        _p("dossier", layer="c", rr=0.7, conf=0.92, sme=True),
    ]
    ranked = rerank_merged_candidates(
        "analyze revenue trends",
        cands,
        cross_encoder=ce,
        top_k=2,
        intent="analyze",
    )
    # SME-backed wins on the intent bonus.
    assert ranked[0].text == "dossier"


def test_intent_without_bonus_does_not_promote_sme() -> None:
    ce = MagicMock()
    ce.predict.return_value = [0.9, 0.7]
    cands = [
        _p("chunk", layer="a", rr=0.9, sme=False),
        _p("sme",   layer="c", rr=0.7, sme=True),
    ]
    # 'lookup' is not in the SME intent bonus table.
    ranked = rerank_merged_candidates(
        "q", cands, cross_encoder=ce, top_k=2, intent="lookup"
    )
    assert ranked[0].text == "chunk"


def test_empty_candidates_returns_empty() -> None:
    ce = MagicMock()
    assert rerank_merged_candidates("q", [], ce, top_k=5) == []
    ce.predict.assert_not_called()


def test_rerank_does_not_mutate_inputs() -> None:
    ce = MagicMock()
    ce.predict.return_value = [0.5, 0.5]
    cands = [_p("a", rr=0.9), _p("b", rr=0.8)]
    out = rerank_merged_candidates("q", cands, cross_encoder=ce, top_k=2)
    # Frozen dataclass → we got fresh instances, inputs unchanged.
    assert cands[0].rerank_score == 0.9
    assert cands[1].rerank_score == 0.8
    assert {o.text for o in out} == {"a", "b"}


# ---------------------------------------------------------------------------
# mmr_select
# ---------------------------------------------------------------------------


def test_mmr_picks_highest_score_when_lam_one() -> None:
    items = [
        _p("alpha", rr=0.99),
        _p("alpha beta", rr=0.98),
        _p("completely unrelated", rr=0.50),
    ]
    picks = mmr_select(items, top_k=2, lam=1.0)
    # lam=1 → pure score.
    texts = [p.text for p in picks]
    assert texts == ["alpha", "alpha beta"]


def test_mmr_spans_clusters_when_lam_low() -> None:
    # Two near-duplicate "alpha" items and one outlier; low lam → diversity.
    items = [
        _p("alpha one", rr=0.95),
        _p("alpha two", rr=0.93),
        _p("completely unrelated theme", rr=0.40),
    ]
    picks = mmr_select(items, top_k=2, lam=0.2)
    texts = [p.text for p in picks]
    assert "alpha one" in texts
    assert "completely unrelated theme" in texts


def test_mmr_empty_returns_empty() -> None:
    assert mmr_select([], top_k=5) == []


def test_mmr_top_k_greater_than_len_returns_all_sorted() -> None:
    items = [_p("a", rr=0.5), _p("b", rr=0.9)]
    picks = mmr_select(items, top_k=5, lam=0.7)
    assert [p.text for p in picks] == ["b", "a"]
