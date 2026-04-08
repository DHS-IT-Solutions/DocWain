"""Prompt template registry for common document processing use cases.

Templates are pre-built configurations that pair a system prompt with
extraction hints and mode suggestions. The `apply_template` helper assembles
the final prompt dict consumed by the standalone processor and router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    name: str
    description: str
    modes: List[str]
    system_prompt: str
    extraction_hints: str
    output_schema: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[str, PromptTemplate] = {}


def _register(template: PromptTemplate) -> None:
    _TEMPLATES[template.name] = template


# 1. Invoice
_register(
    PromptTemplate(
        name="invoice",
        description="Extract invoice fields including vendor, amounts, dates, and line items.",
        modes=["table", "entities"],
        system_prompt=(
            "You are an expert invoice processing assistant. "
            "Your task is to extract all relevant financial and vendor information "
            "from the invoice document provided. Be precise and structured."
        ),
        extraction_hints=(
            "Extract: vendor name, vendor address, invoice number, invoice date, "
            "due date, subtotal, tax amount, total amount, currency, payment terms, "
            "and a full list of line items (description, quantity, unit price, total). "
            "Return amounts as numbers, not strings."
        ),
        output_schema={
            "vendor": "str",
            "invoice_number": "str",
            "invoice_date": "str",
            "due_date": "str",
            "subtotal": "float",
            "tax": "float",
            "total": "float",
            "currency": "str",
            "line_items": [{"description": "str", "quantity": "float", "unit_price": "float", "total": "float"}],
        },
    )
)

# 2. Contract clauses
_register(
    PromptTemplate(
        name="contract_clauses",
        description="Identify and extract contract clauses with risk assessment.",
        modes=["entities", "summary"],
        system_prompt=(
            "You are a legal document analyst specialising in contract review. "
            "Identify all material clauses, flag high-risk provisions, and provide "
            "a balanced risk summary for each clause."
        ),
        extraction_hints=(
            "Extract each named clause (e.g. termination, indemnification, limitation of liability, "
            "governing law, payment terms, confidentiality, intellectual property). "
            "For each clause note: clause_type, verbatim_text, risk_level (low/medium/high), "
            "and a one-sentence risk_note."
        ),
        output_schema={
            "clauses": [
                {
                    "clause_type": "str",
                    "verbatim_text": "str",
                    "risk_level": "str",
                    "risk_note": "str",
                }
            ],
            "overall_risk": "str",
        },
    )
)

# 3. Compliance checklist
_register(
    PromptTemplate(
        name="compliance_checklist",
        description="Check a document against compliance requirements and flag gaps.",
        modes=["qa", "entities"],
        system_prompt=(
            "You are a compliance auditor. Evaluate the document against regulatory "
            "and policy requirements. Produce a structured checklist indicating which "
            "requirements are met, partially met, or missing."
        ),
        extraction_hints=(
            "For each compliance requirement found, record: requirement_id, description, "
            "status (met/partial/missing), evidence (verbatim excerpt or 'none'), "
            "and a remediation_note where status is not 'met'."
        ),
        output_schema={
            "checklist": [
                {
                    "requirement_id": "str",
                    "description": "str",
                    "status": "str",
                    "evidence": "str",
                    "remediation_note": "str",
                }
            ],
            "compliance_score": "float",
        },
    )
)

# 4. Medical record
_register(
    PromptTemplate(
        name="medical_record",
        description="Extract patient info, diagnoses, medications, and procedures from medical records.",
        modes=["entities", "table"],
        system_prompt=(
            "You are a clinical data extraction specialist. Extract structured clinical "
            "information from the medical record while preserving clinical accuracy. "
            "Do not infer or fabricate clinical details not present in the source document."
        ),
        extraction_hints=(
            "Extract: patient demographics (name if present, DOB, sex, MRN), "
            "encounter date, attending physician, primary and secondary diagnoses (ICD codes if present), "
            "active medications (drug, dose, frequency, route), procedures performed, "
            "allergies, and relevant lab results."
        ),
        output_schema={
            "patient": {"dob": "str", "sex": "str", "mrn": "str"},
            "encounter_date": "str",
            "diagnoses": [{"code": "str", "description": "str", "type": "str"}],
            "medications": [{"drug": "str", "dose": "str", "frequency": "str", "route": "str"}],
            "procedures": ["str"],
            "allergies": ["str"],
            "lab_results": [{"test": "str", "value": "str", "unit": "str", "flag": "str"}],
        },
    )
)

# 5. Financial report
_register(
    PromptTemplate(
        name="financial_report",
        description="Extract financial tables, KPIs, and executive summary from financial reports.",
        modes=["table", "summary"],
        system_prompt=(
            "You are a financial analyst assistant. Extract key financial data, performance "
            "indicators, and the executive summary from the financial report. Preserve numeric "
            "precision and note the reporting currency and period."
        ),
        extraction_hints=(
            "Extract: reporting period, currency, revenue, gross profit, EBITDA, net income, "
            "EPS (basic and diluted), total assets, total liabilities, equity, operating cash flow. "
            "Also extract all named KPIs and the verbatim executive summary if present. "
            "Include year-over-year change where available."
        ),
        output_schema={
            "reporting_period": "str",
            "currency": "str",
            "income_statement": {
                "revenue": "float",
                "gross_profit": "float",
                "ebitda": "float",
                "net_income": "float",
                "eps_basic": "float",
                "eps_diluted": "float",
            },
            "balance_sheet": {
                "total_assets": "float",
                "total_liabilities": "float",
                "equity": "float",
            },
            "cash_flow": {"operating": "float"},
            "kpis": [{"name": "str", "value": "str"}],
            "executive_summary": "str",
        },
    )
)

# 6. Resume
_register(
    PromptTemplate(
        name="resume",
        description="Extract skills, work experience, education, and contact information from resumes.",
        modes=["entities", "table"],
        system_prompt=(
            "You are a recruitment data extraction assistant. Extract all structured "
            "information from the resume or CV provided. Be thorough and preserve the "
            "original wording of skills and job titles."
        ),
        extraction_hints=(
            "Extract: candidate name, email, phone, location, LinkedIn/portfolio URLs, "
            "professional summary (verbatim), skills (categorised where possible), "
            "work experience (company, title, start_date, end_date, bullet_points), "
            "education (institution, degree, field, graduation_year, gpa if present), "
            "certifications, and languages."
        ),
        output_schema={
            "contact": {"name": "str", "email": "str", "phone": "str", "location": "str"},
            "summary": "str",
            "skills": [{"category": "str", "items": ["str"]}],
            "experience": [
                {
                    "company": "str",
                    "title": "str",
                    "start_date": "str",
                    "end_date": "str",
                    "highlights": ["str"],
                }
            ],
            "education": [
                {
                    "institution": "str",
                    "degree": "str",
                    "field": "str",
                    "graduation_year": "str",
                }
            ],
            "certifications": ["str"],
            "languages": ["str"],
        },
    )
)

# ---------------------------------------------------------------------------
# Default user prompts per template
# ---------------------------------------------------------------------------

_DEFAULT_USER_PROMPTS: Dict[str, str] = {
    "invoice": "Please extract all invoice data from the document.",
    "contract_clauses": "Identify and assess all material clauses in this contract.",
    "compliance_checklist": "Evaluate this document for compliance and produce a checklist.",
    "medical_record": "Extract all clinical information from this medical record.",
    "financial_report": "Extract financial tables, KPIs, and the executive summary.",
    "resume": "Extract all candidate information from this resume.",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_template(name: str) -> Optional[PromptTemplate]:
    """Return the named template or None if it does not exist."""
    return _TEMPLATES.get(name)


def list_templates() -> List[PromptTemplate]:
    """Return all registered templates."""
    return list(_TEMPLATES.values())


def apply_template(
    template: PromptTemplate,
    user_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Assemble the final prompt dict for the given template.

    The extraction_hints are appended to the system_prompt.
    If *user_prompt* is supplied it takes precedence; otherwise a sensible
    default is used.
    """
    combined_system = f"{template.system_prompt}\n\n{template.extraction_hints}"

    effective_user = (
        user_prompt
        if user_prompt is not None
        else _DEFAULT_USER_PROMPTS.get(template.name, "Process this document.")
    )

    return {
        "system_prompt": combined_system,
        "user_prompt": effective_user,
    }
