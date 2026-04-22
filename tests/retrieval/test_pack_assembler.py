"""Phase 3 Task 5 — budget-aware pack assembly tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval.pack_assembler import PackAssembler
from src.retrieval.types import PackedItem


def _adapter(caps: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(retrieval_caps={"max_pack_tokens": caps})


def _p(
    text: str,
    *,
    layer: str = "a",
    conf: float = 0.9,
    rr: float = 0.5,
    sme: bool = False,
    prov: tuple[tuple[str, str], ...] | None = None,
    meta: dict | None = None,
) -> PackedItem:
    return PackedItem(
        text=text,
        provenance=prov or (("d1", "c1"),),
        layer=layer,
        confidence=conf,
        rerank_score=rr,
        sme_backed=sme,
        metadata=meta or {},
    )


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_pack_under_budget_keeps_all() -> None:
    ad = _adapter({"analyze": 1000})
    pa = PackAssembler(adapter=ad)
    items = [_p("short alpha"), _p("short beta", prov=(("d2", "c2"),))]
    pack = pa.assemble(items, intent="analyze")
    assert len(pack) == 2


def test_pack_over_budget_drops_lowest_confidence_first() -> None:
    # Each item = 4 tokens (18 chars / 4); budget 9 fits exactly 2.
    ad = _adapter({"analyze": 9})
    pa = PackAssembler(adapter=ad)
    # Distinct surface forms so dedup doesn't collapse any of them.
    items = [
        _p("alpha one two thr4", conf=0.95, rr=0.6, prov=(("d1", "c1"),)),
        _p("delta fre one mil5", conf=0.50, rr=0.9, prov=(("d2", "c2"),)),
        _p("gamma sev bil thr8", conf=0.80, rr=0.7, prov=(("d3", "c3"),)),
    ]
    pack = pa.assemble(items, intent="analyze")
    confidences = sorted(p.confidence for p in pack)
    # Low-confidence variant was dropped by the budget gate.
    assert 0.50 not in confidences
    assert confidences == [0.80, 0.95]


def test_unknown_intent_falls_back_to_generic_cap() -> None:
    ad = _adapter({"generic": 60})
    pa = PackAssembler(adapter=ad)
    # Each item ≈ 30 tokens (120 chars / 4). Budget 60 fits 2.
    items = [
        _p("a" * 120, prov=(("d1", "c1"),)),
        _p("b" * 120, prov=(("d2", "c2"),)),
        _p("c" * 120, prov=(("d3", "c3"),)),
    ]
    pack = pa.assemble(items, intent="exotic")
    assert len(pack) == 2


def test_unknown_intent_no_generic_falls_back_to_default() -> None:
    # Caps table has nothing relevant → DEFAULT_CAP (4000) takes over.
    ad = _adapter({"diagnose": 500})
    pa = PackAssembler(adapter=ad)
    items = [_p("x", prov=(("d1", "c1"),))]
    pack = pa.assemble(items, intent="weird_new_intent")
    assert len(pack) == 1


# ---------------------------------------------------------------------------
# SME compression
# ---------------------------------------------------------------------------


def test_layer_c_items_compressed_not_full_narrative() -> None:
    ad = _adapter({"analyze": 4000})
    pa = PackAssembler(adapter=ad)
    long_narrative = "narrative " * 500  # ~5000 chars
    it = _p(
        long_narrative,
        layer="c",
        sme=True,
        prov=(("d1", "c1"), ("d2", "c2")),
        meta={
            "artifact_type": "dossier",
            "key_claims": ["Q3 revenue up 12%", "margin expanded 2pts"],
        },
    )
    pack = pa.assemble([it], intent="analyze")
    assert len(pack) == 1
    compressed = pack[0].text
    # Compressed form keeps the key claims.
    assert "Q3 revenue up 12%" in compressed
    assert "margin expanded 2pts" in compressed
    # Evidence refs come through as [doc#chunk].
    assert "[d1#c1]" in compressed
    # The compressed body is dramatically shorter than the narrative.
    assert len(compressed) < len(long_narrative)
    # And it carries the SME/dossier tag.
    assert "[SME/dossier]" in compressed


def test_layer_c_without_key_claims_uses_narrative_clip() -> None:
    ad = _adapter({"analyze": 4000})
    pa = PackAssembler(adapter=ad)
    long = "word " * 1000
    it = _p(long, layer="c", sme=True, meta={"artifact_type": "dossier"})
    pack = pa.assemble([it], intent="analyze")
    assert pack[0].text.startswith("[SME/dossier]")
    # Body is clipped to the SME body budget (1200 chars by default).
    assert len(pack[0].text) < len(long) + 100


def test_layer_a_text_passes_through_unchanged() -> None:
    ad = _adapter({"analyze": 4000})
    pa = PackAssembler(adapter=ad)
    it = _p("chunk text stays exactly the same")
    pack = pa.assemble([it], intent="analyze")
    assert pack[0].text == "chunk text stays exactly the same"


# ---------------------------------------------------------------------------
# Provenance + dedup
# ---------------------------------------------------------------------------


def test_provenance_preserved_on_every_item() -> None:
    ad = _adapter({"analyze": 1000})
    pa = PackAssembler(adapter=ad)
    pack = pa.assemble(
        [_p("a", prov=(("d1", "c1"), ("d2", "c2")))],
        intent="analyze",
    )
    assert pack[0].provenance == (("d1", "c1"), ("d2", "c2"))


def test_semantic_dedup_collapses_near_duplicates() -> None:
    ad = _adapter({"analyze": 4000})
    pa = PackAssembler(adapter=ad)
    # Same tokens → Jaccard=1.0 > 0.85 threshold.
    text = "alpha beta gamma delta"
    items = [
        _p(text, prov=(("d1", "c1"),)),
        _p(text, prov=(("d2", "c2"),), sme=True),
        _p("totally different words here", prov=(("d3", "c3"),)),
    ]
    pack = pa.assemble(items, intent="analyze")
    # Two kept: the SME-backed variant wins the dedup collapse.
    assert len(pack) == 2
    # SME-backed variant survived the near-dup collision.
    sme_kept = [p for p in pack if p.sme_backed]
    assert len(sme_kept) == 1


# ---------------------------------------------------------------------------
# Drop-order determinism
# ---------------------------------------------------------------------------


def test_drop_order_stable_by_confidence_then_rerank() -> None:
    # Budget fits exactly one 30-token item.
    ad = _adapter({"analyze": 30})
    pa = PackAssembler(adapter=ad)
    txt = "y" * 120  # ~30 tokens
    items = [
        _p(txt, conf=0.5, rr=0.9, prov=(("d1", "c1"),)),  # low conf, high rr
        _p(txt, conf=0.5, rr=0.3, prov=(("d2", "c2"),)),  # low conf, low rr
        _p(txt, conf=0.9, rr=0.4, prov=(("d3", "c3"),)),  # highest conf
    ]
    pack = pa.assemble(items, intent="analyze")
    # Highest confidence wins, even though its rerank is middling.
    assert len(pack) == 1
    assert pack[0].provenance == (("d3", "c3"),)


def test_empty_items_returns_empty() -> None:
    ad = _adapter({"analyze": 1000})
    pa = PackAssembler(adapter=ad)
    assert pa.assemble([], intent="analyze") == []


def test_missing_caps_uses_default() -> None:
    ad = SimpleNamespace(retrieval_caps=None)
    pa = PackAssembler(adapter=ad)
    # No crash; a small pack fits under default 4000 cap.
    pack = pa.assemble([_p("x")], intent="analyze")
    assert len(pack) == 1
