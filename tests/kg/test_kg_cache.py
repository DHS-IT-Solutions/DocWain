"""Tests for KG entity cache."""

from unittest.mock import MagicMock

from src.kg.kg_cache import KGCache


def test_kg_cache_loads_entity_catalog():
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


def test_kg_cache_survives_neo4j_failure():
    mock_store = MagicMock()
    mock_store.run_query.side_effect = Exception("Neo4j down")
    cache = KGCache()
    cache.warm(neo4j_store=mock_store)
    assert cache.entity_count == 0
    assert cache.is_warmed is False


def test_kg_cache_entity_lookup():
    mock_store = MagicMock()
    mock_store.run_query.return_value = [
        {"name": "Acme Corp", "type": "ORG", "doc_count": 5},
    ]
    cache = KGCache()
    cache.warm(neo4j_store=mock_store)
    match = cache.lookup_entity("Acme Corp")
    assert match is not None
    assert match["type"] == "ORG"
    assert cache.lookup_entity("Nonexistent") is None
