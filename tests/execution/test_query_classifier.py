"""Tests for the rule-based query classifier."""

from src.execution.query_classifier import QueryClassification, classify_query


class TestConversational:
    def test_greeting_classified_as_conversational(self):
        result = classify_query("Hello!")
        assert result.query_type == "CONVERSATIONAL"

    def test_thanks_classified_as_conversational(self):
        result = classify_query("Thanks, that helps")
        assert result.query_type == "CONVERSATIONAL"

    def test_bye_classified_as_conversational(self):
        result = classify_query("Bye!")
        assert result.query_type == "CONVERSATIONAL"

    def test_affirmation_classified_as_conversational(self):
        result = classify_query("Ok")
        assert result.query_type == "CONVERSATIONAL"

    def test_yes_classified_as_conversational(self):
        result = classify_query("Yes")
        assert result.query_type == "CONVERSATIONAL"

    def test_short_non_question_classified_as_conversational(self):
        result = classify_query("Sounds good")
        assert result.query_type == "CONVERSATIONAL"


class TestSimple:
    def test_simple_lookup(self):
        result = classify_query("What is the total revenue?")
        assert result.query_type == "SIMPLE"

    def test_who_is_query(self):
        result = classify_query("Who is the CEO?")
        assert result.query_type == "SIMPLE"

    def test_short_question(self):
        result = classify_query("When was the contract signed?")
        assert result.query_type == "SIMPLE"


class TestAnalytical:
    def test_analytical_suggestion(self):
        result = classify_query("Based on the financial data, what areas need improvement?")
        assert result.query_type == "ANALYTICAL"

    def test_analytical_recommendation(self):
        result = classify_query("Can you suggest ways to improve our margins?")
        assert result.query_type == "ANALYTICAL"

    def test_analytical_identify_risks(self):
        result = classify_query("Identify risks in the current portfolio")
        assert result.query_type == "ANALYTICAL"


class TestComplex:
    def test_complex_comparison(self):
        result = classify_query(
            "Compare the revenue trends between Q1 and Q3 across all departments"
        )
        assert result.query_type == "COMPLEX"

    def test_complex_multi_doc(self):
        result = classify_query(
            "What are the differences between the 2023 and 2024 annual reports?"
        )
        assert result.query_type == "COMPLEX"

    def test_unknown_defaults_to_complex(self):
        result = classify_query(
            "Tell me everything about the organizational restructuring and its "
            "impact on various departments including finance, HR, and operations "
            "over the past fiscal year"
        )
        assert result.query_type == "COMPLEX"

    def test_multiple_questions_complex(self):
        result = classify_query("What is the revenue? And what about the costs?")
        assert result.query_type == "COMPLEX"


class TestMetadata:
    def test_classification_has_signals(self):
        result = classify_query("Hello!")
        assert len(result.signals) > 0

    def test_classification_has_confidence(self):
        result = classify_query("What is the revenue?")
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_query(self):
        result = classify_query("")
        assert result.query_type == "CONVERSATIONAL"

    def test_returns_dataclass(self):
        result = classify_query("Hello!")
        assert isinstance(result, QueryClassification)
