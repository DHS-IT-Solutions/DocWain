"""Unit tests for parse_viz_directives() (Task 29)."""

import pytest

from src.visualization.enhancer import parse_viz_directives


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_tuple(self):
        result = parse_viz_directives("no tags here")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_list(self):
        directives, _ = parse_viz_directives("no tags here")
        assert isinstance(directives, list)

    def test_second_element_is_str(self):
        _, clean = parse_viz_directives("no tags here")
        assert isinstance(clean, str)


# ---------------------------------------------------------------------------
# No tags
# ---------------------------------------------------------------------------

class TestNoTags:
    def test_empty_string(self):
        directives, clean = parse_viz_directives("")
        assert directives == []
        assert clean == ""

    def test_plain_text_unchanged(self):
        text = "This is just regular text with no viz tags."
        directives, clean = parse_viz_directives(text)
        assert directives == []
        assert clean == text

    def test_html_comment_not_matched(self):
        text = "<!-- viz type='bar' --> some text"
        directives, clean = parse_viz_directives(text)
        assert directives == []


# ---------------------------------------------------------------------------
# Single tag — attribute extraction
# ---------------------------------------------------------------------------

class TestSingleTag:
    def test_single_tag_parsed(self):
        text = '<viz type="bar" title="Sales" />'
        directives, clean = parse_viz_directives(text)
        assert len(directives) == 1
        assert directives[0]["type"] == "bar"
        assert directives[0]["title"] == "Sales"

    def test_tag_removed_from_clean(self):
        text = 'Before <viz type="bar" /> after.'
        _, clean = parse_viz_directives(text)
        assert "<viz" not in clean
        assert "/>" not in clean

    def test_surrounding_text_preserved(self):
        text = 'Hello <viz type="pie" /> world.'
        _, clean = parse_viz_directives(text)
        assert "Hello" in clean
        assert "world." in clean

    def test_single_attribute(self):
        text = '<viz chart="line" />'
        directives, _ = parse_viz_directives(text)
        assert directives[0]["chart"] == "line"

    def test_many_attributes(self):
        text = '<viz type="bar" title="Rev" unit="$" color="blue" animated="true" />'
        directives, _ = parse_viz_directives(text)
        d = directives[0]
        assert d["type"] == "bar"
        assert d["title"] == "Rev"
        assert d["unit"] == "$"
        assert d["color"] == "blue"
        assert d["animated"] == "true"

    def test_empty_attribute_value(self):
        text = '<viz type="bar" title="" />'
        directives, _ = parse_viz_directives(text)
        assert directives[0]["title"] == ""

    def test_attribute_value_with_spaces(self):
        text = '<viz title="Monthly Revenue Report" type="bar" />'
        directives, _ = parse_viz_directives(text)
        assert directives[0]["title"] == "Monthly Revenue Report"


# ---------------------------------------------------------------------------
# Multiple tags
# ---------------------------------------------------------------------------

class TestMultipleTags:
    def test_two_tags_both_parsed(self):
        text = 'First <viz type="bar" /> middle <viz type="pie" /> last.'
        directives, _ = parse_viz_directives(text)
        assert len(directives) == 2
        assert directives[0]["type"] == "bar"
        assert directives[1]["type"] == "pie"

    def test_all_tags_removed_from_clean(self):
        text = '<viz type="bar" /> some text <viz type="line" />'
        _, clean = parse_viz_directives(text)
        assert "<viz" not in clean

    def test_three_tags(self):
        text = (
            '<viz type="a" /> text1 '
            '<viz type="b" /> text2 '
            '<viz type="c" />'
        )
        directives, _ = parse_viz_directives(text)
        assert len(directives) == 3

    def test_order_preserved(self):
        text = '<viz id="first" /> <viz id="second" /> <viz id="third" />'
        directives, _ = parse_viz_directives(text)
        assert [d["id"] for d in directives] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Whitespace cleanup
# ---------------------------------------------------------------------------

class TestWhitespaceCleanup:
    def test_double_spaces_collapsed(self):
        text = "word1  <viz type='x' />  word2"
        # The viz tag is removed leaving double spaces
        _, clean = parse_viz_directives(text)
        assert "  " not in clean

    def test_multiple_blank_lines_collapsed(self):
        text = "line1\n\n\n\nline2"
        _, clean = parse_viz_directives(text)
        assert "\n\n\n" not in clean

    def test_no_leading_trailing_whitespace(self):
        text = "  <viz type='bar' />  "
        _, clean = parse_viz_directives(text)
        assert clean == clean.strip()

    def test_clean_text_after_inline_removal(self):
        text = "Revenue <viz type='bar' title='Revenue' /> data shows growth."
        _, clean = parse_viz_directives(text)
        # Should not start or end with extra spaces
        assert not clean.startswith(" ")
        assert not clean.endswith(" ")
        assert "Revenue" in clean
        assert "data shows growth." in clean


# ---------------------------------------------------------------------------
# Case insensitivity on tag name
# ---------------------------------------------------------------------------

class TestCaseInsensitivity:
    def test_uppercase_VIZ_matched(self):
        text = '<VIZ type="bar" />'
        directives, _ = parse_viz_directives(text)
        assert len(directives) == 1
        assert directives[0]["type"] == "bar"

    def test_mixed_case_VIZ_matched(self):
        text = '<Viz type="line" />'
        directives, _ = parse_viz_directives(text)
        assert len(directives) == 1


# ---------------------------------------------------------------------------
# Edge cases — malformed or partial tags
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unclosed_tag_not_matched(self):
        """<viz type="bar"> without /> should not be matched."""
        text = '<viz type="bar"> some text'
        directives, clean = parse_viz_directives(text)
        assert len(directives) == 0

    def test_mixed_valid_and_invalid(self):
        """Only the self-closing tag should be matched."""
        text = '<viz type="bar"> invalid <viz type="pie" /> valid'
        directives, _ = parse_viz_directives(text)
        assert len(directives) == 1
        assert directives[0]["type"] == "pie"

    def test_tag_with_no_attributes(self):
        """<viz /> with no attributes returns an empty dict."""
        text = "<viz />"
        directives, clean = parse_viz_directives(text)
        assert len(directives) == 1
        assert directives[0] == {}
        assert "<viz" not in clean

    def test_text_only_after_removal_is_correct(self):
        before = "start"
        after = "end"
        text = f"{before} <viz type='x' /> {after}"
        _, clean = parse_viz_directives(text)
        assert before in clean
        assert after in clean
