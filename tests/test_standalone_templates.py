"""Tests for standalone_templates.py — prompt template registry."""

import pytest
from src.api.standalone_templates import (
    PromptTemplate,
    get_template,
    list_templates,
    apply_template,
)


class TestGetTemplate:
    def test_get_template_by_name(self):
        tmpl = get_template("invoice")
        assert tmpl is not None
        assert tmpl.name == "invoice"
        assert tmpl.description != ""
        assert "table" in tmpl.modes
        assert "entities" in tmpl.modes
        assert tmpl.system_prompt != ""
        assert tmpl.extraction_hints != ""

    def test_get_template_unknown_returns_none(self):
        result = get_template("nonexistent")
        assert result is None

    def test_get_template_all_six_exist(self):
        expected = [
            "invoice",
            "contract_clauses",
            "compliance_checklist",
            "medical_record",
            "financial_report",
            "resume",
        ]
        for name in expected:
            tmpl = get_template(name)
            assert tmpl is not None, f"Template '{name}' not found"

    def test_get_template_returns_prompt_template_instance(self):
        tmpl = get_template("resume")
        assert isinstance(tmpl, PromptTemplate)


class TestListTemplates:
    def test_list_templates_returns_at_least_six(self):
        templates = list_templates()
        assert len(templates) >= 6

    def test_list_templates_contains_all_names(self):
        templates = list_templates()
        names = {t.name for t in templates}
        expected = {
            "invoice",
            "contract_clauses",
            "compliance_checklist",
            "medical_record",
            "financial_report",
            "resume",
        }
        assert expected.issubset(names)

    def test_list_templates_all_have_modes(self):
        for tmpl in list_templates():
            assert isinstance(tmpl.modes, list), f"{tmpl.name} modes should be a list"
            assert len(tmpl.modes) >= 1, f"{tmpl.name} should have at least one mode"

    def test_list_templates_all_have_output_schema(self):
        for tmpl in list_templates():
            assert isinstance(
                tmpl.output_schema, dict
            ), f"{tmpl.name} output_schema should be a dict"


class TestApplyTemplate:
    def test_apply_template_with_user_prompt_preserved(self):
        tmpl = get_template("invoice")
        result = apply_template(tmpl, user_prompt="Extract totals only.")
        assert result["user_prompt"] == "Extract totals only."

    def test_apply_template_system_prompt_non_empty(self):
        tmpl = get_template("invoice")
        result = apply_template(tmpl)
        assert "system_prompt" in result
        assert len(result["system_prompt"]) > 0

    def test_apply_template_extraction_hints_appended_to_system_prompt(self):
        tmpl = get_template("invoice")
        result = apply_template(tmpl)
        assert tmpl.extraction_hints in result["system_prompt"]

    def test_apply_template_no_user_prompt_uses_default(self):
        tmpl = get_template("contract_clauses")
        result = apply_template(tmpl)
        assert "user_prompt" in result
        assert result["user_prompt"] != ""

    def test_apply_template_returns_dict_with_required_keys(self):
        tmpl = get_template("medical_record")
        result = apply_template(tmpl)
        assert set(result.keys()) >= {"system_prompt", "user_prompt"}

    def test_apply_template_user_prompt_overrides_default(self):
        tmpl = get_template("financial_report")
        custom = "Summarise only the revenue figures."
        result_custom = apply_template(tmpl, user_prompt=custom)
        result_default = apply_template(tmpl)
        assert result_custom["user_prompt"] == custom
        assert result_default["user_prompt"] != custom


class TestTemplateContents:
    """Spot-check modes and descriptions for each template."""

    def test_invoice_modes(self):
        t = get_template("invoice")
        assert set(t.modes) == {"table", "entities"}

    def test_contract_clauses_modes(self):
        t = get_template("contract_clauses")
        assert set(t.modes) == {"entities", "summary"}

    def test_compliance_checklist_modes(self):
        t = get_template("compliance_checklist")
        assert set(t.modes) == {"qa", "entities"}

    def test_medical_record_modes(self):
        t = get_template("medical_record")
        assert set(t.modes) == {"entities", "table"}

    def test_financial_report_modes(self):
        t = get_template("financial_report")
        assert set(t.modes) == {"table", "summary"}

    def test_resume_modes(self):
        t = get_template("resume")
        assert set(t.modes) == {"entities", "table"}
