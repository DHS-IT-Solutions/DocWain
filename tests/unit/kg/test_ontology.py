"""Unit tests for src/kg/ontology.py"""

import pytest

from src.kg.ontology import (
    ALL_RELATIONSHIPS,
    DOMAINS,
    detect_domain,
    get_domain_relationships,
    get_relationship_schema,
)


# ---------------------------------------------------------------------------
# DOMAINS list
# ---------------------------------------------------------------------------


class TestAllDomainsExist:
    def test_expected_domains_present(self):
        expected = {"legal", "financial", "hr", "medical", "generic"}
        assert expected == set(DOMAINS)

    def test_domains_is_list(self):
        assert isinstance(DOMAINS, list)

    def test_domains_non_empty(self):
        assert len(DOMAINS) > 0


# ---------------------------------------------------------------------------
# get_domain_relationships
# ---------------------------------------------------------------------------


class TestGetDomainReturnsRelationshipTypes:
    def test_legal_returns_expected_names(self):
        names = get_domain_relationships("legal")
        expected = {
            "party_to",
            "signatory_of",
            "governed_by",
            "amends",
            "supersedes",
            "terminates",
            "effective_from",
            "expires_on",
        }
        assert expected == set(names)

    def test_financial_returns_expected_names(self):
        names = get_domain_relationships("financial")
        expected = {"invoiced_by", "paid_to", "line_item_of", "totals_to", "billed_on", "due_on"}
        assert expected == set(names)

    def test_hr_returns_expected_names(self):
        names = get_domain_relationships("hr")
        expected = {"employed_by", "reports_to", "holds_certification", "worked_during", "role_of"}
        assert expected == set(names)

    def test_medical_returns_expected_names(self):
        names = get_domain_relationships("medical")
        expected = {"diagnosed_with", "prescribed", "treated_by", "allergic_to", "admitted_on"}
        assert expected == set(names)

    def test_generic_returns_expected_names(self):
        names = get_domain_relationships("generic")
        expected = {"related_to", "mentioned_in", "part_of", "located_at"}
        assert expected == set(names)

    def test_returns_list_of_strings(self):
        names = get_domain_relationships("legal")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# Unknown domain falls back to generic
# ---------------------------------------------------------------------------


class TestUnknownDomainReturnsGeneric:
    def test_unknown_domain_matches_generic(self):
        generic = get_domain_relationships("generic")
        unknown = get_domain_relationships("nonexistent_domain")
        assert set(generic) == set(unknown)

    def test_empty_string_domain_matches_generic(self):
        generic = get_domain_relationships("generic")
        result = get_domain_relationships("")
        assert set(generic) == set(result)

    def test_none_like_string_matches_generic(self):
        generic = get_domain_relationships("generic")
        result = get_domain_relationships("undefined")
        assert set(generic) == set(result)


# ---------------------------------------------------------------------------
# get_relationship_schema
# ---------------------------------------------------------------------------


class TestRelationshipHasSchema:
    def test_known_relationship_returns_dict(self):
        schema = get_relationship_schema("party_to")
        assert isinstance(schema, dict)

    def test_schema_has_required_keys(self):
        schema = get_relationship_schema("party_to")
        assert "name" in schema
        assert "source_types" in schema
        assert "target_types" in schema
        assert "domain" in schema

    def test_schema_name_matches_lookup_key(self):
        schema = get_relationship_schema("invoiced_by")
        assert schema["name"] == "invoiced_by"

    def test_schema_source_and_target_types_are_lists(self):
        schema = get_relationship_schema("employed_by")
        assert isinstance(schema["source_types"], list)
        assert isinstance(schema["target_types"], list)

    def test_schema_domain_is_correct(self):
        assert get_relationship_schema("party_to")["domain"] == "legal"
        assert get_relationship_schema("invoiced_by")["domain"] == "financial"
        assert get_relationship_schema("employed_by")["domain"] == "hr"
        assert get_relationship_schema("diagnosed_with")["domain"] == "medical"
        assert get_relationship_schema("related_to")["domain"] == "generic"

    def test_unknown_relationship_returns_none(self):
        assert get_relationship_schema("does_not_exist") is None

    def test_all_domains_have_schemas_retrievable(self):
        for domain in DOMAINS:
            for name in get_domain_relationships(domain):
                schema = get_relationship_schema(name)
                assert schema is not None, f"Missing schema for '{name}' in domain '{domain}'"


# ---------------------------------------------------------------------------
# ALL_RELATIONSHIPS uniqueness
# ---------------------------------------------------------------------------


class TestAllRelationshipTypesUnique:
    def test_all_relationship_names_unique(self):
        names = [r["name"] for r in ALL_RELATIONSHIPS]
        assert len(names) == len(set(names)), "Duplicate relationship names found in ALL_RELATIONSHIPS"

    def test_all_relationships_have_required_keys(self):
        required = {"name", "source_types", "target_types", "domain"}
        for rel in ALL_RELATIONSHIPS:
            missing = required - rel.keys()
            assert not missing, f"Relationship {rel.get('name')} missing keys: {missing}"

    def test_all_relationships_cover_all_domains(self):
        domains_in_all = {r["domain"] for r in ALL_RELATIONSHIPS}
        assert set(DOMAINS) == domains_in_all


# ---------------------------------------------------------------------------
# detect_domain
# ---------------------------------------------------------------------------


class TestDetectDomainFromEntities:
    def test_legal_entities_detected(self):
        entities = [{"type": "CONTRACT"}, {"type": "CLAUSE"}, {"type": "PERSON"}]
        assert detect_domain(entities) == "legal"

    def test_financial_entities_detected(self):
        entities = [{"type": "INVOICE"}, {"type": "AMOUNT"}, {"type": "PAYMENT"}]
        assert detect_domain(entities) == "financial"

    def test_hr_entities_detected(self):
        entities = [{"type": "ROLE"}, {"type": "CERTIFICATION"}]
        assert detect_domain(entities) == "hr"

    def test_medical_entities_detected(self):
        entities = [{"type": "CONDITION"}, {"type": "MEDICATION"}, {"type": "SUBSTANCE"}]
        assert detect_domain(entities) == "medical"

    def test_empty_entities_returns_generic(self):
        assert detect_domain([]) == "generic"

    def test_no_hint_match_returns_generic(self):
        entities = [{"type": "UNKNOWN_TYPE"}, {"type": "ANOTHER_UNKNOWN"}]
        assert detect_domain(entities) == "generic"

    def test_dominant_domain_wins_over_minority(self):
        # 3 financial hints vs 1 legal hint → financial wins
        entities = [
            {"type": "INVOICE"},
            {"type": "PAYMENT"},
            {"type": "AMOUNT"},
            {"type": "CONTRACT"},
        ]
        assert detect_domain(entities) == "financial"

    def test_entities_missing_type_key_handled_gracefully(self):
        entities = [{"label": "foo"}, {"type": "INVOICE"}, {"type": "PAYMENT"}]
        assert detect_domain(entities) == "financial"

    def test_single_strong_hint_detected(self):
        entities = [{"type": "REGULATION"}]
        assert detect_domain(entities) == "legal"
