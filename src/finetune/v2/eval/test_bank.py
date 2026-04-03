"""Test bank for V2 track evaluation.

Each track has a set of test examples with queries, reference answers,
and scoring criteria.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _excel_csv_tests() -> List[Dict[str, Any]]:
    return [
        {
            "id": "excel_csv_001",
            "query": "Extract all line items from this invoice and compute the total amount.",
            "reference": {
                "keywords": ["line item", "total", "amount", "quantity", "price"],
                "expects_table": True,
                "expects_tool_call": True,
            },
        },
        {
            "id": "excel_csv_002",
            "query": "Parse the spreadsheet data and identify the top 5 vendors by total spend.",
            "reference": {
                "keywords": ["vendor", "spend", "total", "rank"],
                "expects_table": True,
                "expects_tool_call": True,
            },
        },
        {
            "id": "excel_csv_003",
            "query": "Convert the CSV data into a structured table showing employee names, departments, and salaries.",
            "reference": {
                "keywords": ["employee", "department", "salary", "name"],
                "expects_table": True,
                "expects_tool_call": False,
            },
        },
        {
            "id": "excel_csv_004",
            "query": "Summarize the quarterly revenue figures from this financial spreadsheet.",
            "reference": {
                "keywords": ["quarter", "revenue", "Q1", "Q2", "total"],
                "expects_table": True,
                "expects_tool_call": True,
            },
        },
        {
            "id": "excel_csv_005",
            "query": "Find any duplicate entries in this purchase order spreadsheet.",
            "reference": {
                "keywords": ["duplicate", "purchase", "order", "entry"],
                "expects_table": False,
                "expects_tool_call": True,
            },
        },
    ]


def _layout_tests() -> List[Dict[str, Any]]:
    return [
        {
            "id": "layout_001",
            "query": "Analyze the structure of this document and identify all section headings.",
            "reference": {
                "keywords": ["heading", "section", "structure", "document"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "layout_002",
            "query": "Detect all tables and figures in this document and describe their locations.",
            "reference": {
                "keywords": ["table", "figure", "page", "location"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "layout_003",
            "query": "Extract the header and footer content from each page of this document.",
            "reference": {
                "keywords": ["header", "footer", "page", "content"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "layout_004",
            "query": "Identify the multi-column layout regions in this document.",
            "reference": {
                "keywords": ["column", "layout", "region", "text"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "layout_005",
            "query": "Map the document hierarchy showing how sections relate to subsections.",
            "reference": {
                "keywords": ["hierarchy", "section", "subsection", "structure"],
                "expects_tool_call": False,
            },
        },
    ]


def _ocr_vision_tests() -> List[Dict[str, Any]]:
    return [
        {
            "id": "ocr_001",
            "query": "Extract all text from this scanned document image.",
            "reference": {
                "keywords": ["text", "extract", "document"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "ocr_002",
            "query": "Read the handwritten notes in this image and transcribe them.",
            "reference": {
                "keywords": ["handwritten", "notes", "transcribe"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "ocr_003",
            "query": "Extract the data from this table in the scanned PDF page.",
            "reference": {
                "keywords": ["table", "data", "extract", "row", "column"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "ocr_004",
            "query": "Identify and correct any OCR errors in the extracted text.",
            "reference": {
                "keywords": ["correct", "error", "OCR", "text"],
                "expects_tool_call": False,
            },
        },
        {
            "id": "ocr_005",
            "query": "What does the stamp on this document say?",
            "reference": {
                "keywords": ["stamp", "document", "text"],
                "expects_tool_call": True,
            },
        },
    ]


def _reasoning_tests() -> List[Dict[str, Any]]:
    return [
        {
            "id": "reasoning_001",
            "query": "Compare the terms of Contract A and Contract B. Which offers better indemnification protection?",
            "reference": {
                "keywords": ["contract", "indemnification", "protection", "compare", "clause"],
            },
        },
        {
            "id": "reasoning_002",
            "query": "Based on the financial statements from 2023 and 2024, what is the revenue growth trend?",
            "reference": {
                "keywords": ["revenue", "growth", "trend", "2023", "2024", "increase"],
            },
        },
        {
            "id": "reasoning_003",
            "query": "Cross-reference the audit report findings with the compliance policy. Are there any gaps?",
            "reference": {
                "keywords": ["audit", "compliance", "gap", "finding", "policy"],
            },
        },
        {
            "id": "reasoning_004",
            "query": "Synthesize information from all uploaded documents to create a risk assessment.",
            "reference": {
                "keywords": ["risk", "assessment", "document", "finding", "severity"],
            },
        },
        {
            "id": "reasoning_005",
            "query": "Timeline the key events mentioned across all documents and identify any contradictions.",
            "reference": {
                "keywords": ["timeline", "event", "contradiction", "date", "document"],
            },
        },
    ]


def _kg_tests() -> List[Dict[str, Any]]:
    return [
        {
            "id": "kg_001",
            "query": "Identify all entities mentioned in this document and their relationships.",
            "reference": {
                "keywords": ["entity", "relationship", "person", "organization"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "kg_002",
            "query": "Build a knowledge graph of the parties, obligations, and dates in this contract.",
            "reference": {
                "keywords": ["party", "obligation", "date", "contract", "relationship"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "kg_003",
            "query": "Find all cross-references between sections in this document.",
            "reference": {
                "keywords": ["cross-reference", "section", "reference", "link"],
                "expects_tool_call": True,
            },
        },
        {
            "id": "kg_004",
            "query": "Extract the organizational hierarchy mentioned in this employee handbook.",
            "reference": {
                "keywords": ["organization", "hierarchy", "department", "report"],
                "expects_tool_call": False,
            },
        },
        {
            "id": "kg_005",
            "query": "Map the dependencies between the requirements listed in this specification document.",
            "reference": {
                "keywords": ["dependency", "requirement", "specification", "depends"],
                "expects_tool_call": True,
            },
        },
    ]


def _visualization_tests() -> List[Dict[str, Any]]:
    return [
        {
            "id": "viz_001",
            "query": "Create a bar chart showing the quarterly revenue from this financial data.",
            "reference": {
                "keywords": ["quarter", "revenue", "chart"],
                "expects_viz": True,
                "expected_chart_type": "bar",
            },
        },
        {
            "id": "viz_002",
            "query": "Generate a pie chart showing the expense distribution by category.",
            "reference": {
                "keywords": ["expense", "category", "distribution"],
                "expects_viz": True,
                "expected_chart_type": "donut",
            },
        },
        {
            "id": "viz_003",
            "query": "Plot the monthly trend of employee headcount over the past year.",
            "reference": {
                "keywords": ["monthly", "trend", "headcount", "employee"],
                "expects_viz": True,
                "expected_chart_type": "line",
            },
        },
        {
            "id": "viz_004",
            "query": "Compare the performance metrics across all departments using a radar chart.",
            "reference": {
                "keywords": ["performance", "department", "metric", "compare"],
                "expects_viz": True,
                "expected_chart_type": "radar",
            },
        },
        {
            "id": "viz_005",
            "query": "Visualize the project budget allocation as a treemap.",
            "reference": {
                "keywords": ["budget", "allocation", "project"],
                "expects_viz": True,
                "expected_chart_type": "treemap",
            },
        },
    ]


def get_test_bank(track: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """Return test bank examples, optionally filtered to a single track.

    Parameters
    ----------
    track:
        If provided, return only tests for this track.
        If None, return all tracks.

    Returns
    -------
    Dict mapping track name to list of test examples.
    """
    all_tests = {
        "excel_csv": _excel_csv_tests(),
        "layout": _layout_tests(),
        "ocr_vision": _ocr_vision_tests(),
        "reasoning": _reasoning_tests(),
        "kg": _kg_tests(),
        "visualization": _visualization_tests(),
    }
    if track is not None:
        if track not in all_tests:
            raise ValueError(f"Unknown track: {track!r}")
        return {track: all_tests[track]}
    return all_tests
