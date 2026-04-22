"""Single source of truth for all LLM prompts in DocWain.

Every prompt the system sends to the LLM is constructed here.
No other module should contain prompt text or LLM instructions.

Phase 4 additions (rich mode):

- :func:`build_system_prompt` gains optional ``persona``, ``shape``, and
  ``pack`` kwargs. When ``shape == "rich"`` and a ``persona`` dict is
  supplied, the persona role + voice + grounding rules are injected at the
  TOP of the system prompt. When ``shape == "compact"`` the legacy behaviour
  is preserved — no Phase 4 caller breaks.
- New TASK_FORMATS entries for ``diagnose`` / ``analyze`` / ``recommend``
  give the reasoner explicit section skeletons for rich-mode responses.
- :class:`ResponseShape`, :func:`resolve_response_shape`, rich-prompt
  builders, and the persona-injection helper land below the legacy REASON
  prompt surface. The memory rule — formatting lives here, not in
  ``src/intelligence/generator.py`` — is preserved by keeping all new
  string assembly in this module.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence

if TYPE_CHECKING:
    from src.retrieval.types import PackedItem

# ---------------------------------------------------------------------------
# Response-format templates. Owned here (not in src/intelligence/generator.py)
# per the prompt-path rule. Each template instructs the model on how to
# structure a user-facing response.
# ---------------------------------------------------------------------------

_FORMAT_INSTRUCTIONS = {
    "table": (
        "Present data in a clean markdown table.\n"
        "- Use | column | headers | with alignment\n"
        "- One data point per row, no merged cells\n"
        "- Bold key values: **$9,000.00**, **Jessica Jones**\n"
        "- Add a brief summary sentence above the table"
    ),
    "bullets": (
        "Present as a structured bulleted list.\n"
        "- Lead with a one-line summary sentence\n"
        "- Group related bullets under **bold category headers**\n"
        "- Each bullet: **Label:** value or description\n"
        "- Bold key names, amounts, dates, entities\n"
        "- Most important items first"
    ),
    "sections": (
        "Organize the response with clear visual hierarchy.\n"
        "- Start with a one-line executive summary\n"
        "- Use ## for major sections, ### for subsections\n"
        "- Within sections, use bullet points with **bold labels**\n"
        "- Format: **Field Name:** extracted value or insight\n"
        "- Bold all key values: names, amounts, dates, identifiers\n"
        "- Use markdown tables for tabular data (line items, comparisons)\n"
        "- Never leave headers as plain text — always use ## or ###\n"
        "- Keep bullets self-contained — each makes sense alone\n"
        "- End with a brief synthesis or key takeaway if appropriate"
    ),
    "numbered": (
        "Present as a numbered list.\n"
        "- Each item: **Label** — description with **bold key values**\n"
        "- Sequential order, one point per number\n"
        "- Brief summary before the list"
    ),
    "prose": (
        "Write clear, structured paragraphs.\n"
        "- Lead with the direct answer in the first sentence\n"
        "- Bold key values: **$9,000.00**, **Jessica Jones**, **Document 0522**\n"
        "- Use short paragraphs (2-3 sentences each)\n"
        "- For any tabular data, use a markdown table instead of inline text"
    ),
}


def get_format_instruction(shape: str) -> str:
    """Return the format template for the given shape. Defaults to 'prose'."""
    return _FORMAT_INSTRUCTIONS.get(shape, _FORMAT_INSTRUCTIONS["prose"])


# ---------------------------------------------------------------------------
# System prompt — used as the system message for every generation call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are DocWain, an enterprise document intelligence system by DHS IT Solutions. "
    "You extract, analyze, and reason about any type of document with precision. "
    "Always respond in English. Use markdown formatting with bold values and tables "
    "for structured data. Ground every claim in document evidence. "
    "When information is insufficient, say so clearly rather than guessing."
)


# Intents that default to rich shape when the flag is on. Trivial intents
# always stay compact; borderline intents get rich only when the pack is
# SME-backed. See :func:`resolve_response_shape`.
_TRIVIAL_INTENTS: frozenset[str] = frozenset(
    {"greeting", "identity", "lookup", "count", "extract"}
)
_ANALYTICAL_INTENTS: frozenset[str] = frozenset(
    {"analyze", "diagnose", "recommend", "investigate"}
)
_BORDERLINE_INTENTS: frozenset[str] = frozenset(
    {"compare", "summarize", "aggregate", "list", "overview"}
)


def build_system_prompt(
    profile_domain: str = "",
    kg_context: str = "",
    profile_expertise: Optional[Dict] = None,
    evidence_count: int = 0,
    evidence_quality: str = "strong",
    *,
    persona: Optional[Dict[str, Any]] = None,
    shape: Literal["rich", "compact"] = "compact",
    pack: Optional[Sequence["PackedItem"]] = None,
) -> str:
    """Return the system prompt for DocWain, adapted to evidence quality.

    DocWain's identity and formatting are reinforced here while adapting
    to the quality and quantity of available evidence.

    Phase 4 extensions (keyword-only, default-off):

    - ``persona`` — a dict with keys ``role`` / ``voice`` / ``grounding_rules``.
      When ``shape == "rich"`` and persona is provided, the role / voice /
      grounding rules land at the top of the system prompt, BEFORE the
      canonical DocWain identity paragraph. This lets the adapter SME voice
      dominate for analytical intents without erasing DocWain identity.
    - ``shape`` — ``"rich"`` injects persona; ``"compact"`` (default)
      preserves legacy behaviour. Explicit compact override from the user
      always wins upstream (classifier → resolver → this function).
    - ``pack`` — optional reference to the packed items used for the
      response. Not currently rendered into the system prompt; reserved for
      future per-pack prompt customisation. Passing it today is safe.
    """
    parts: List[str] = []

    # Phase 4 — persona prelude goes FIRST when rich + persona are supplied.
    # Emits a leading persona sentence + grounding-rules sentence with a
    # single trailing space so the concatenation below matches legacy spacing.
    if shape == "rich" and persona:
        role = persona.get("role") or ""
        voice = persona.get("voice") or ""
        grounding_rules = persona.get("grounding_rules") or ()
        persona_bits: List[str] = []
        if role:
            persona_bits.append(f"You are acting as a {role}.")
        if voice:
            persona_bits.append(f"Voice: {voice}.")
        if grounding_rules:
            persona_bits.append(
                "Grounding rules (strict): "
                + " ".join(f"- {r}" for r in grounding_rules)
            )
        if persona_bits:
            parts.append(" ".join(persona_bits) + " ")

    parts.append(_SYSTEM_PROMPT)

    if evidence_count > 0 and evidence_quality == "weak":
        parts.append(
            " IMPORTANT: The available evidence is limited or low-confidence. "
            "If the evidence doesn't clearly support your answer, explicitly state "
            "what is missing rather than speculating or inferring."
        )

    if profile_domain:
        parts.append(f" Domain context: {profile_domain}.")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Task-type formatting instructions
# ---------------------------------------------------------------------------

_UNIVERSAL_INSTRUCTION = (
    "ALWAYS provide comprehensive, detailed responses. Never give one-line or "
    "single-value answers. Every response must include:\n"
    "1. The direct answer to the query\n"
    "2. Supporting context and relevant details from the document(s)\n"
    "3. A brief insight or observation that adds value\n"
    "Cite source document names when referencing specific facts.\n\n"
)

TASK_FORMATS: Dict[str, str] = {
    "extract": (
        "TASK: Extract the requested information precisely.\n"
        "- Start with a ## header for each major category (e.g., ## Vendor Details, ## Line Items).\n"
        "- Use bullets (- **Label:** value) for single-value fields.\n"
        "- MANDATORY: Use a markdown table for line items, multi-row data, or anything with 3+ entries.\n"
        "  Example table:\n"
        "  | Item | Description | Amount |\n"
        "  |------|-------------|--------|\n"
        "  | Service | Details | **$X.XX** |\n"
        "- Bold ALL extracted values inline: **$720.00**, **Super Widget Industries**, **5 mockups**.\n"
        "- Keep each bullet on ONE line — never break **bold** markers across lines.\n"
        "- If a requested field is not found, state: 'Not found in provided documents.'\n"
        "- For procedural extractions (steps, protocols), use numbered lists.\n"
        "- NEVER fabricate values not present in the evidence.\n"
    ),
    "compare": (
        "TASK: Compare the subjects systematically.\n"
        "- Start with a one-line summary of the key difference.\n"
        "- MANDATORY: Present a markdown comparison table. Example:\n"
        "  | Criteria | Subject A | Subject B |\n"
        "  |----------|-----------|----------|\n"
        "  | Experience | **8 years** | 3 years |\n"
        "- Keep the table focused: max 4-5 meaningful columns. Do NOT include internal scores, "
        "metadata, relevance values, or image descriptions as columns.\n"
        "- **Bold** the better or more notable value in each cell.\n"
        "- End with 2-3 bullet points synthesising the key takeaways.\n"
        "- The table MUST be complete — do not truncate mid-row.\n"
    ),
    "summarize": (
        "TASK: Provide a structured summary.\n"
        "- Start with a one-line executive summary.\n"
        "- Use ## section headers for major topics, ### for subtopics.\n"
        "- Bullets: **Label:** value — keep each on ONE line.\n"
        "- Bold all key values: amounts, names, dates, identifiers.\n"
        "- Use a table for any tabular data (line items, comparisons).\n"
        "- Include specific values and counts — not vague generalities.\n"
        "- End with a Key Takeaway.\n"
    ),
    "overview": (
        "TASK: Provide a structured overview of the document collection.\n"
        "- Start with a one-line summary of what the collection covers.\n"
        "- Then present a ### section for each document or major topic, containing:\n"
        "  - **Document name** and type in the header\n"
        "  - 3-5 bullet points with the most important content, findings, or purpose\n"
        "  - Key entities, instruments, or subjects mentioned\n"
        "- End with a brief synthesis of how the documents relate to each other.\n"
        "- This format is for broad queries like 'tell me about the documents' or "
        "'what do we have' — give the user a clear map of their collection.\n"
    ),
    "investigate": (
        "TASK: Investigate and assess the question.\n"
        "- Structure with ### headers: Finding, Evidence, Assessment.\n"
        "- Use bullet points under each section.\n"
        "- Flag risks, inconsistencies, or concerns explicitly with **bold** labels.\n"
        "- Distinguish between what the evidence shows vs. what it doesn't cover.\n"
        "- Be precise about severity: **Critical** vs. **Minor** vs. **Informational**.\n"
    ),
    "lookup": (
        "TASK: Provide a direct, comprehensive answer.\n"
        "- Lead with the direct answer in **bold**.\n"
        "- Then provide supporting context: relevant details, related facts, and key observations from the document.\n"
        "- Use structured formatting (bold values, bullet points) for clarity.\n"
        "- Cite which document(s) the answer comes from.\n"
        "- End with a brief insight or observation that adds value beyond the raw fact.\n"
    ),
    "aggregate": (
        "TASK: Aggregate and quantify from the evidence.\n"
        "- Lead with totals, counts, or computed values in **bold**.\n"
        "- Show the breakdown as a markdown table if multi-item, or bullet list if few.\n"
        "- State which documents contributed to each value.\n"
        "- Flag if any expected data is missing from the aggregation.\n"
    ),
    "list": (
        "TASK: List the requested items.\n"
        "- State the total count at the top: '**N items found:**'\n"
        "- Use a numbered list if order matters, bulleted if not.\n"
        "- Include relevant details for each item (not just names).\n"
        "- For each item, **bold** the item name and follow with a brief description.\n"
    ),
    # ----------------------------------------------------------------------
    # Phase 4 — rich-mode TASK entries for the three new analytical intents
    # ----------------------------------------------------------------------
    "diagnose": (
        "TASK: Diagnose the reported problem with a symptom → cause → fix structure.\n"
        "- Lead with a crisp Executive summary naming the most likely root cause.\n"
        "- Under ## Symptom, restate the observed behaviour using exact language from evidence.\n"
        "- Under ## Causes (ranked), list ranked candidate causes with confidence + evidence citation.\n"
        "- Under ## Fix, give concrete remediation steps; NEVER invent log lines or error codes.\n"
        "- Under ## Assumptions & caveats, surface any ambiguity; avoid silent guesses.\n"
        "- Every quantitative claim carries an inline [doc_id:chunk_id] citation.\n"
    ),
    "analyze": (
        "TASK: Produce a rich, evidence-grounded analysis.\n"
        "- Lead with a 1–3 sentence ## Executive summary; headline first.\n"
        "- Under ## Observations, list the concrete facts from evidence with inline citations.\n"
        "- Under ## Patterns, tie observations across at least two distinct documents where possible.\n"
        "- Under ## Interpretation, explain what the patterns mean — no silent extrapolation.\n"
        "- Under ## Assumptions & caveats, list explicit assumptions + anything unverifiable.\n"
        "- Every quantitative claim carries an inline [doc_id:chunk_id] citation.\n"
    ),
    "recommend": (
        "TASK: Recommend 3–5 prioritized actions, each grounded in evidence or the Recommendation Bank.\n"
        "- Lead with a ## Executive summary naming the headline recommendation.\n"
        "- Under ## Recommendations, list 3–5 items; each is a concrete action.\n"
        "- For each recommendation provide: rationale, supporting evidence, expected impact, assumptions.\n"
        "- Under ## Rationale & evidence, expand why the top recommendation is preferred.\n"
        "- Under ## Assumptions & caveats, surface anything dependent on conditions you cannot verify.\n"
        "- Every recommendation must either cite a Recommendation Bank entry OR carry an inline\n"
        "  [doc_id:chunk_id] citation. Speculation is forbidden — unverifiable claims are dropped by\n"
        "  the post-generation grounding pass.\n"
    ),
}

# ---------------------------------------------------------------------------
# Output format instructions
# ---------------------------------------------------------------------------

_OUTPUT_FORMATS: Dict[str, str] = {
    "table": (
        "Present data in a clean markdown table.\n"
        "- Use | Column | Headers | with alignment\n"
        "- One data point per row, bold key values in cells\n"
        "- Add a summary sentence above the table"
    ),
    "bullets": (
        "Present as a structured bulleted list.\n"
        "- Group related items under **bold category labels** on their own line\n"
        "- Each bullet: **Label:** value or description — keep on ONE line\n"
        "- Bold key values inline: costs, names, dates, IDs\n"
        "- Most important items first"
    ),
    "sections": (
        "Organize with clear visual hierarchy.\n"
        "- Use ## for major sections, ### for subsections\n"
        "- Within sections, use bullet points: **Label:** value\n"
        "- Bold ALL key values: **$9,000.00**, **Jessica Jones**, **Document 0522**\n"
        "- Use markdown tables for line items or comparisons\n"
        "- Keep each bullet on a SINGLE line — never split bold markers across lines\n"
        "- End with a key takeaway"
    ),
    "numbered": (
        "Use a numbered list.\n"
        "- Each item: **Label** — description with bold key values\n"
        "- Keep each item on one line\n"
        "- Brief summary before the list"
    ),
    "prose": (
        "Write clear paragraphs.\n"
        "- Lead with the direct answer\n"
        "- Bold key values inline: **$9,000.00**, **Jessica Jones**\n"
        "- Short paragraphs (2-3 sentences)\n"
        "- Use a table for any tabular data"
    ),
}

# ---------------------------------------------------------------------------
# UNDERSTAND prompt — analyzes user intent against document intelligence
# ---------------------------------------------------------------------------

_UNDERSTAND_SYSTEM = (
    "You are a document intelligence query analyzer. "
    "Given a user query, conversation history, and document metadata, "
    "produce a JSON analysis of what the user needs.\n\n"
    "Rules:\n"
    "- Decompose multi-part queries into sub-intents.\n"
    "- Resolve pronouns using conversation history.\n"
    "- Infer output format from query semantics "
    "(table for comparisons, bullets for lists, sections for summaries, prose for factual).\n"
    "- Assess complexity: 'simple' if single document/fact, 'complex' if cross-document or multi-step.\n"
    "- Identify which documents are relevant using the document intelligence metadata.\n"
    "- If the query is conversational (greeting, thanks, meta-question), set task_type to 'conversational'.\n"
)


def build_understand_prompt(
    query: str,
    doc_intelligence: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]],
) -> str:
    """Build the UNDERSTAND prompt that analyzes user intent.

    Args:
        query: The user's question.
        doc_intelligence: List of document intelligence dicts with keys:
            document_id, profile_id, profile_name, summary, entities,
            answerable_topics.
        conversation_history: Recent turns as [{"query": ..., "response": ...}].

    Returns:
        Complete prompt string for the UNDERSTAND LLM call.
    """
    parts = [_UNDERSTAND_SYSTEM, ""]

    # Conversation context
    if conversation_history:
        parts.append("CONVERSATION HISTORY:")
        for turn in conversation_history[-5:]:  # last 5 turns max
            parts.append(f"  User: {turn.get('query', '')}")
            resp = turn.get("response", "")
            if isinstance(resp, dict):
                resp = resp.get("response", str(resp))
            parts.append(f"  DocWain: {str(resp)[:300]}")
        parts.append("")

    # Document intelligence context
    if doc_intelligence:
        parts.append("AVAILABLE DOCUMENTS:")
        for doc in doc_intelligence:
            doc_id = doc.get("document_id", "unknown")
            profile = doc.get("profile_name", doc.get("profile_id", "unknown"))
            summary = doc.get("summary", "No summary available")
            entities = doc.get("entities", [])
            topics = doc.get("answerable_topics", [])

            parts.append(f"  [{doc_id}] Profile: {profile}")
            parts.append(f"    Summary: {summary}")
            if entities:
                entity_strs = [
                    e.get("name", str(e)) if isinstance(e, dict) else str(e)
                    for e in entities[:10]
                ]
                parts.append(f"    Entities: {', '.join(entity_strs)}")
            if topics:
                parts.append(f"    Topics: {', '.join(topics[:10])}")
            parts.append("")
    else:
        parts.append("AVAILABLE DOCUMENTS: None found in this subscription.\n")

    # Query
    parts.append(f"USER QUERY: {query}")
    parts.append("")

    # Expected output — JSON schema
    parts.append(
        "Respond ONLY with JSON (no markdown fences):\n"
        "{\n"
        '  "task_type": "extract | compare | summarize | overview | investigate | lookup | aggregate | list | conversational",\n'
        '  "complexity": "simple | complex",\n'
        '  "resolved_query": "query with pronouns resolved from conversation",\n'
        '  "output_format": "table | bullets | sections | numbered | prose",\n'
        '  "relevant_documents": [\n'
        '    {"document_id": "...", "profile_id": "...", "reason": "why this doc is relevant"}\n'
        "  ],\n"
        '  "cross_profile": true | false,\n'
        '  "sub_tasks": ["sub-task 1", "sub-task 2"] | null,\n'
        '  "entities": ["entity1", "entity2"],\n'
        '  "needs_clarification": false,\n'
        '  "clarification_question": null\n'
        "}\n\n"
        "TASK TYPE GUIDE:\n"
        "- 'overview': Use for broad/vague queries about the collection (e.g. 'tell me about the documents', "
        "'what do we have', 'give me an overview'). Output format should be 'sections'.\n"
        "- 'summarize': Use for queries about a specific document or topic's content.\n"
        "- 'extract': Use when user wants specific values, procedures, or data points. "
        "If the query asks for steps/procedures, set output_format to 'numbered'.\n"
        "- 'compare': Use when user asks to compare, contrast, or differentiate. Output format should be 'table'.\n"
        "- 'list': Use when user asks for a list of items.\n"
        "- 'lookup': Use for simple factual questions with a single answer.\n"
        "- COMPLEXITY GUIDE: Only classify as 'simple' if the query asks for a "
        "single, specific fact (a name, date, amount, yes/no). Questions about "
        "risks, implications, processes, recommendations, or 'what should I know "
        "about' are ALWAYS 'complex'.\n"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# REASON prompt — generates the answer from evidence
# ---------------------------------------------------------------------------


def build_reason_prompt(
    query: str,
    task_type: str,
    output_format: str,
    evidence: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the REASON prompt that generates the final answer.

    Args:
        query: The resolved user query.
        task_type: From UNDERSTAND step (extract, compare, etc.).
        output_format: From UNDERSTAND step (table, bullets, etc.).
        evidence: Ranked evidence chunks, each with:
            source_name, section, page, text, score, source_index.
        doc_context: Aggregated document intelligence context with:
            summary, entities, key_facts (optional fields).
        conversation_history: Recent conversation turns as
            [{"query": ..., "response": ...}].

    Returns:
        Complete prompt string for the REASON LLM call.
    """
    parts = []

    # Document intelligence context (orientation before evidence)
    if doc_context:
        parts.append("--- DOCUMENT INTELLIGENCE ---")
        if doc_context.get("document_types"):
            parts.append(f"Document types: {', '.join(doc_context['document_types'])}")
        if doc_context.get("summary"):
            parts.append(f"Overview: {doc_context['summary']}")

        # Structured entity context
        entities = doc_context.get("entities")
        if entities:
            parts.append("")
            parts.append("## Known Entities")
            for e in entities[:15]:
                if isinstance(e, dict):
                    etype = e.get("type", "")
                    value = e.get("value", e.get("name", ""))
                    role = e.get("role", e.get("context", ""))
                    if etype and value:
                        entry = f"- {etype}: {value}"
                    elif value:
                        entry = f"- {value}"
                    else:
                        entry = f"- {etype}"
                    if role:
                        entry += f" ({role})"
                    parts.append(entry)
                else:
                    parts.append(f"- {e}")

        # Pre-extracted facts as grounding anchors
        key_facts = doc_context.get("key_facts")
        if key_facts:
            parts.append("")
            parts.append("## Pre-Extracted Facts (use as grounding anchors)")
            for fact in key_facts[:10]:
                if isinstance(fact, dict):
                    claim = fact.get("claim", str(fact))
                    evidence_ref = fact.get("evidence", "")
                    entry = f"- {claim}"
                    if evidence_ref:
                        entry += f" [{evidence_ref}]"
                    parts.append(entry)
                else:
                    parts.append(f"- {fact}")

        parts.append("--- END DOCUMENT INTELLIGENCE ---")
        parts.append("")

    # Document Index — always present when available
    doc_index = None
    doc_intel_summaries = None
    if doc_context:
        doc_index = doc_context.get("doc_index")
        doc_intel_summaries = doc_context.get("doc_intelligence_summaries")

    if doc_index:
        parts.append("--- DOCUMENT INDEX (%d documents in this profile) ---" % len(doc_index))
        for i, entry in enumerate(doc_index, 1):
            parts.append(f"{i}. {entry}")
        parts.append("--- END DOCUMENT INDEX ---")
        parts.append("")

    if doc_intel_summaries:
        parts.append("--- DOCUMENT INTELLIGENCE (structured summaries — USE AS PRIMARY EVIDENCE) ---")
        parts.append("These summaries contain verified extracted facts from each document.")
        parts.append("Use these as your primary source of truth when answering questions.")
        parts.append("")
        _intel_chars = 0
        _MAX_INTEL_CHARS = 16000  # Increased cap for comprehensive coverage
        for entry in doc_intel_summaries:
            if _intel_chars + len(entry) > _MAX_INTEL_CHARS:
                parts.append(f"... ({len(doc_intel_summaries) - doc_intel_summaries.index(entry)} more documents)")
                break
            parts.append(entry)
            parts.append("")
            _intel_chars += len(entry)
        parts.append("--- END DOCUMENT INTELLIGENCE ---")
        parts.append("")

    # Expert analysis context (pre-computed insights)
    expert_insights = None
    if doc_context:
        expert_insights = doc_context.get("expert_insights")
    if expert_insights:
        parts.append("--- EXPERT ANALYSIS ---")
        parts.append("Pre-computed expert observations relevant to this query:")
        for insight in expert_insights[:5]:
            if isinstance(insight, dict):
                cat = insight.get("category", "")
                text = insight.get("insight", "")
                rec = insight.get("recommendation", "")
                parts.append(f"- [{cat.upper()}] {text}")
                if rec:
                    parts.append(f"  Recommendation: {rec}")
            else:
                parts.append(f"- {insight}")
        parts.append("--- END EXPERT ANALYSIS ---")
        parts.append("")

    # Evidence block
    parts.append("--- EVIDENCE ---")
    if evidence:
        for item in evidence:
            idx = item.get("source_index", 0)
            name = item.get("source_name", "unknown")
            section = item.get("section", "")
            page = item.get("page", "")
            score = item.get("score", 0)
            text = item.get("text", "")

            header_parts = [f"[SOURCE-{idx}]", name]
            if section:
                header_parts.append(f"| Section: {section}")
            if page:
                header_parts.append(f"| p.{page}")

            parts.append(" ".join(header_parts))
            parts.append(text)
            parts.append("")
    else:
        parts.append("No evidence found in the uploaded documents.")
        parts.append("")
    parts.append("--- END EVIDENCE ---")
    parts.append("")

    # Conversation context
    if conversation_history:
        parts.append("CONVERSATION CONTEXT:")
        for turn in conversation_history[-3:]:
            parts.append(f"  User: {turn.get('query', '')}")
            resp = turn.get("response", "")
            if isinstance(resp, dict):
                resp = resp.get("response", str(resp))
            parts.append(f"  DocWain: {str(resp)[:200]}")
        parts.append("")

    # Universal instruction + task-specific format
    parts.append(_UNIVERSAL_INSTRUCTION)
    task_instruction = TASK_FORMATS.get(task_type, TASK_FORMATS["lookup"])
    parts.append(task_instruction)

    # Output format
    format_instruction = _OUTPUT_FORMATS.get(output_format, _OUTPUT_FORMATS["prose"])
    parts.append(f"FORMAT: {format_instruction}")
    parts.append("")

    # The question
    parts.append(f"QUESTION: {query}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sub-agent prompt — focused task for dynamic sub-agents
# ---------------------------------------------------------------------------


def build_subagent_prompt(
    role: str,
    evidence: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a focused prompt for a dynamic sub-agent.

    Args:
        role: Description of what this sub-agent should do.
        evidence: Evidence chunks scoped to this sub-agent's task.
        doc_context: Document intelligence for relevant documents.

    Returns:
        Complete prompt string for the sub-agent LLM call.
    """
    parts = [
        "You are a document analysis sub-agent. Your specific task:",
        f"  {role}",
        "",
        "Rules: Use ONLY the evidence below. Be precise. Use exact values.",
        "If the evidence doesn't contain what's needed, say so.",
        "",
    ]

    # Document context
    if doc_context:
        if doc_context.get("summary"):
            parts.append(f"Document context: {doc_context['summary']}")
            parts.append("")

    # Evidence
    parts.append("--- EVIDENCE ---")
    if evidence:
        for item in evidence:
            idx = item.get("source_index", 0)
            name = item.get("source_name", "unknown")
            score = item.get("score", 0)
            text = item.get("text", "")

            parts.append(f"[SOURCE-{idx}] {name} (relevance: {score:.2f})")
            parts.append(text)
            parts.append("")
    else:
        parts.append("No evidence provided.")
        parts.append("")
    parts.append("--- END EVIDENCE ---")

    return "\n".join(parts)


# ===========================================================================
# Phase 4 — rich-mode response infrastructure
# ===========================================================================
#
# The three additions below make Phase 4 land:
#
# 1. PersonaBundle + persona_bundle_from_adapter — adapter → prompt-shape
#    handoff (ERRATA §1: reads adapter.content_hash / .version directly).
# 2. ResponseShape + resolve_response_shape — the single choke point for
#    compact / rich / honest_compact selection.
# 3. Rich prompt builders for analyze / diagnose / recommend — shape-parallel
#    implementations; they share _render_common_blocks for persona, grounding,
#    evidence formatting.
#
# The memory rule — formatting stays in generation/prompts.py — is the
# reason this block lives here and NOT in intelligence/generator.py.

import enum
from dataclasses import dataclass, field

from src.generation.pack_summary import PackSummary
from src.serving.model_router import FormatHint


# ---------------------------------------------------------------------------
# ResponseShape + resolver
# ---------------------------------------------------------------------------


class ResponseShape(str, enum.Enum):
    """The three shapes a response can take.

    COMPACT = today's templates; RICH = Phase 4 SME skeleton;
    HONEST_COMPACT = RICH asked for but pack is too thin — fall back to
    compact with a visible caveat so the user sees the limitation.
    """

    COMPACT = "compact"
    RICH = "rich"
    HONEST_COMPACT = "honest_compact"


def resolve_response_shape(
    *,
    intent: str,
    format_hint: FormatHint,
    pack: PackSummary,
    enable_rich_mode: bool,
) -> ResponseShape:
    """Single choke point for compact vs rich vs honest-compact.

    Precedence (highest first):

    1. Explicit compact override from user  → COMPACT
    2. ``enable_rich_mode`` flag OFF        → COMPACT
    3. Explicit rich override from user     → RICH (honors thin pack)
    4. Trivial intent                       → COMPACT
    5. Analytical intent + thin pack        → HONEST_COMPACT
    6. Analytical intent + adequate pack    → RICH
    7. Borderline intent + SME artifacts    → RICH
    8. Borderline intent + no artifacts     → HONEST_COMPACT
    9. Anything else                        → COMPACT (safe default)
    """
    if format_hint is FormatHint.COMPACT:
        return ResponseShape.COMPACT
    if not enable_rich_mode:
        return ResponseShape.COMPACT
    if format_hint is FormatHint.RICH:
        return ResponseShape.RICH
    if intent in _TRIVIAL_INTENTS:
        return ResponseShape.COMPACT
    if intent in _ANALYTICAL_INTENTS:
        if not pack.has_sme_artifacts and pack.total_chunks < 4:
            return ResponseShape.HONEST_COMPACT
        return ResponseShape.RICH
    if intent in _BORDERLINE_INTENTS:
        if pack.has_sme_artifacts:
            return ResponseShape.RICH
        return ResponseShape.HONEST_COMPACT
    return ResponseShape.COMPACT


# ---------------------------------------------------------------------------
# Persona bundle
# ---------------------------------------------------------------------------

_RICH_INTENTS: frozenset[str] = frozenset({"analyze", "diagnose", "recommend"})


@dataclass(frozen=True)
class PersonaBundle:
    """Adapter-derived persona carrier for the rich prompt builders.

    Read-only; the only production builder is
    :func:`persona_bundle_from_adapter`. Tests can construct directly.
    """

    role: str
    voice: str
    grounding_rules: tuple[str, ...]
    intent_template_body: str
    adapter_version: str
    adapter_content_hash: str


def persona_bundle_from_adapter(adapter: Any, *, intent: str) -> PersonaBundle:
    """Assemble a :class:`PersonaBundle` for one intent from a loaded Adapter.

    ``adapter`` is the Phase 1 AdapterLoader output; we never mutate it and
    never re-fetch from Blob here. Empty ``intent_template_body`` signals
    "no per-domain override, use the global rich_{intent}.md template" — the
    caller handles that fallback.

    Per ERRATA §1, ``content_hash`` and ``version`` are direct attributes on
    the Adapter instance (populated at load time by AdapterLoader). We do
    NOT call ``last_load_metadata()`` to obtain them.
    """
    if intent not in _RICH_INTENTS:
        raise ValueError(
            f"persona_bundle_from_adapter: unsupported intent {intent!r}; "
            f"rich mode applies only to {sorted(_RICH_INTENTS)}"
        )
    body = getattr(adapter.response_persona_prompts, intent, "") or ""
    return PersonaBundle(
        role=adapter.persona.role,
        voice=adapter.persona.voice,
        grounding_rules=tuple(adapter.persona.grounding_rules),
        intent_template_body=body,
        adapter_version=getattr(adapter, "version", "") or "",
        adapter_content_hash=getattr(adapter, "content_hash", "") or "",
    )


# ---------------------------------------------------------------------------
# Rich prompt builders — analyze / diagnose / recommend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyzePromptInputs:
    """Load-bearing input container for :func:`build_analyze_rich_prompt`."""

    query_text: str
    persona_role: str
    persona_voice: str
    grounding_rules: Sequence[str]
    pack_tokens: int
    output_cap_tokens: int
    evidence_items: Sequence[dict]
    insight_refs: Sequence[dict] = field(default_factory=tuple)
    domain: str = "generic"


@dataclass(frozen=True)
class DiagnosePromptInputs:
    """Load-bearing input container for :func:`build_diagnose_rich_prompt`."""

    query_text: str
    persona_role: str
    persona_voice: str
    grounding_rules: Sequence[str]
    pack_tokens: int
    output_cap_tokens: int
    evidence_items: Sequence[dict]
    diagnostic_hits: Sequence[dict] = field(default_factory=tuple)
    domain: str = "generic"


@dataclass(frozen=True)
class RecommendPromptInputs:
    """Load-bearing input container for :func:`build_recommend_rich_prompt`."""

    query_text: str
    persona_role: str
    persona_voice: str
    grounding_rules: Sequence[str]
    pack_tokens: int
    output_cap_tokens: int
    evidence_items: Sequence[dict]
    bank_entries: Sequence[dict] = field(default_factory=tuple)
    domain: str = "generic"


def _render_common_blocks(
    *,
    persona_role: str,
    persona_voice: str,
    grounding_rules: Sequence[str],
    query_text: str,
    evidence_items: Sequence[dict],
    domain: str,
) -> Dict[str, str]:
    """Shared formatting for the three rich builders.

    Returns a dict with ``persona``, ``grounding``, ``query``, ``evidence``
    keys. Callers render their own section skeleton on top.
    """
    persona = (
        f"You are acting as a {persona_role}. "
        f"Voice: {persona_voice}. "
        f"Domain context: {domain}."
    )
    grounding = "\n".join(f"- {rule}" for rule in grounding_rules)
    evidence = "\n".join(
        f"[{e.get('doc_id', '?')}:{e.get('chunk_id', '?')}] {e.get('text', '')}"
        for e in evidence_items
    )
    return {
        "persona": persona,
        "grounding": grounding,
        "query": query_text,
        "evidence": evidence,
    }


def build_analyze_rich_prompt(inp: AnalyzePromptInputs) -> str:
    """Rich-mode analyze template — exec summary + observations + patterns +
    interpretation + caveats + evidence."""
    blocks = _render_common_blocks(
        persona_role=inp.persona_role,
        persona_voice=inp.persona_voice,
        grounding_rules=inp.grounding_rules,
        query_text=inp.query_text,
        evidence_items=inp.evidence_items,
        domain=inp.domain,
    )
    insights = (
        "\n".join(
            f"- ({i.get('type', 'insight')}) {i.get('narrative', '')}"
            for i in inp.insight_refs
        )
        if inp.insight_refs
        else "(none materialized for this query)"
    )
    return (
        f"{blocks['persona']}\n\n"
        f"Grounding rules (strict):\n{blocks['grounding']}\n\n"
        f"User question: {blocks['query']}\n\n"
        f"Pre-reasoned insights from this profile:\n{insights}\n\n"
        f"Evidence (cite inline as [doc_id:chunk_id]):\n{blocks['evidence']}\n\n"
        "Produce a rich analysis response with EXACTLY these sections, in this order:\n"
        "## Executive summary\n"
        "One to three sentences, headline first, streamed before any other section.\n"
        "## Analysis\n"
        "Evidence-grounded narrative. Every quantitative claim carries an inline citation.\n"
        "## Patterns\n"
        "Cross-document patterns tied to at least two distinct docs where possible.\n"
        "## Assumptions & caveats\n"
        "Explicit assumptions; anything you cannot verify is listed here, not hidden.\n"
        "## Evidence\n"
        "Bullet list of (doc_id:chunk_id) items actually cited above.\n\n"
        f"Soft output target: ~{inp.output_cap_tokens} tokens. Quality trumps the target;\n"
        "never pad. If evidence is thin, say so in Executive summary and shorten Analysis.\n"
    )


def build_diagnose_rich_prompt(inp: DiagnosePromptInputs) -> str:
    """Rich-mode diagnose template — symptom + ranked causes + fix + caveats.

    Diagnose differs from analyze in three ways:
      1. Section names (Symptom + Causes instead of Patterns)
      2. ``diagnostic_hits`` replace insight refs in the pre-reasoning block
      3. Soft output target default is higher — adapter-tunable via
         ``output_cap_tokens``
    Everything else (persona, grounding, streaming-first exec summary,
    evidence block with inline [doc_id:chunk_id]) is intentionally parallel.
    """
    blocks = _render_common_blocks(
        persona_role=inp.persona_role,
        persona_voice=inp.persona_voice,
        grounding_rules=inp.grounding_rules,
        query_text=inp.query_text,
        evidence_items=inp.evidence_items,
        domain=inp.domain,
    )
    hits = (
        "\n".join(
            f"- rank {h.get('rank', '?')} — {h.get('symptom', '')} "
            f"[{h.get('doc_id', '?')}:{h.get('chunk_id', '?')}]"
            for h in inp.diagnostic_hits
        )
        if inp.diagnostic_hits
        else "(no ranked symptom hits available)"
    )
    return (
        f"{blocks['persona']}\n\n"
        f"Grounding rules (strict):\n{blocks['grounding']}\n\n"
        f"User question: {blocks['query']}\n\n"
        f"Diagnostic hits (ranked by evidence strength):\n{hits}\n\n"
        f"Evidence (cite inline as [doc_id:chunk_id]):\n{blocks['evidence']}\n\n"
        "Produce a rich diagnose response with EXACTLY these sections, in this order:\n"
        "## Executive summary\n"
        "One to three sentences naming the most likely root cause, streamed first.\n"
        "## Symptom\n"
        "Restate the observed behaviour using exact language from evidence.\n"
        "## Causes (ranked)\n"
        "Ranked list of candidate causes with confidence and inline citations.\n"
        "## Assumptions & caveats\n"
        "Explicit assumptions; anything unverifiable is listed here.\n"
        "## Evidence\n"
        "Bullet list of (doc_id:chunk_id) items actually cited above.\n\n"
        f"Soft output target: ~{inp.output_cap_tokens} tokens. Never invent log lines\n"
        "or error codes. If evidence is thin, say so and shorten Causes.\n"
    )


def build_recommend_rich_prompt(inp: RecommendPromptInputs) -> str:
    """Rich-mode recommend template — top 3-5 recommendations with rationale,
    evidence, impact, assumptions, caveats.

    Recommend differs from analyze / diagnose:
      1. Recommendations are pulled from the Recommendation Bank (Layer C
         artifact) — the pre-reasoning block IS the grounding anchor.
      2. The post-generation recommendation_grounding pass runs AFTER this
         template produces a response. Any recommendation the pass cannot
         tie to a bank entry or cited chunk is dropped and replaced by a
         candid note. This template does not need to reject ungrounded
         claims itself — the post-pass is the enforcement point.
      3. Every bank entry is injected verbatim so the LLM has no reason to
         re-generate wording — it should quote and cite.
    """
    blocks = _render_common_blocks(
        persona_role=inp.persona_role,
        persona_voice=inp.persona_voice,
        grounding_rules=inp.grounding_rules,
        query_text=inp.query_text,
        evidence_items=inp.evidence_items,
        domain=inp.domain,
    )
    bank_blocks: List[str] = []
    for entry in inp.bank_entries:
        rec = entry.get("recommendation", "")
        rationale = entry.get("rationale", "")
        evidence_refs = entry.get("evidence", [])
        impact = entry.get("estimated_impact", "")
        assumptions = entry.get("assumptions", [])
        confidence = entry.get("confidence", "")
        bank_blocks.append(
            f"- Recommendation: {rec}\n"
            f"  Rationale: {rationale}\n"
            f"  Evidence: {', '.join(evidence_refs) if evidence_refs else '(none cited)'}\n"
            f"  Estimated impact: {impact}\n"
            f"  Assumptions: {', '.join(assumptions) if assumptions else '(none)'}\n"
            f"  Confidence: {confidence}"
        )
    bank_block = "\n".join(bank_blocks) or "(no Recommendation Bank entries)"
    return (
        f"{blocks['persona']}\n\n"
        f"Grounding rules (strict):\n{blocks['grounding']}\n\n"
        f"User question: {blocks['query']}\n\n"
        "Recommendation Bank or exposed reasoning chain (grounding anchor):\n"
        f"{bank_block}\n\n"
        f"Evidence (cite inline as [doc_id:chunk_id]):\n{blocks['evidence']}\n\n"
        "Produce a rich recommend response with EXACTLY these sections, in this order:\n"
        "## Executive summary\n"
        "One to three sentences naming the headline recommendation, streamed first.\n"
        "## Recommendations\n"
        "Numbered list of 3–5 recommendations. Each item must cite a Recommendation\n"
        "Bank entry OR carry an inline [doc_id:chunk_id] citation.\n"
        "## Rationale & evidence\n"
        "Expand why the top recommendation is preferred; cite evidence.\n"
        "## Assumptions & caveats\n"
        "Explicit assumptions; anything unverifiable is listed here.\n"
        "## Evidence\n"
        "Bullet list of (doc_id:chunk_id) items actually cited above.\n\n"
        f"Soft output target: ~{inp.output_cap_tokens} tokens. Speculation is\n"
        "forbidden — the post-generation grounding pass drops unverifiable claims.\n"
    )


def build_honest_compact_prompt(
    *,
    query_text: str,
    pack_summary: PackSummary,
) -> str:
    """Compact template wrapped with a visible "evidence is limited" caveat.

    Used when the shape resolver would prefer rich but the pack is too thin.
    The user sees the limitation rather than a thin rich skeleton or a
    silently-compact answer.
    """
    return (
        "The available evidence is limited; the following answer is necessarily compact:\n\n"
        f"User question: {query_text}\n\n"
        f"(Profile has {pack_summary.total_chunks} retrieved chunks across "
        f"{pack_summary.distinct_docs} distinct document(s); no pre-reasoned SME "
        "artifacts were available for this query.)\n\n"
        "Respond with a direct, honest compact answer. If the evidence does not\n"
        "answer the question, say so explicitly.\n"
    )


# ---------------------------------------------------------------------------
# Phase 5 — URL citation annotation + supplementary prompt template
# ---------------------------------------------------------------------------
#
# Two small extensions that sit alongside the Phase-4 builders and keep all
# prompt-string construction inside this module (memory rule: response
# formatting lives here, not in ``intelligence/generator.py``).
#
# 1. :func:`annotate_citation` — builds the citation label for a single
#    pack item. Profile items use ``[doc_id:chunk_id]``; URL items use the
#    host + (truncated) path so users see URL provenance distinctly.
# 2. :func:`build_supplementary_prompt` — constructs the continuation
#    prompt for the second-pass Reasoner call when URL content arrives
#    after the primary response has already streamed. Same grounding /
#    verifier discipline as the rich templates.
# ---------------------------------------------------------------------------


def annotate_citation(
    *,
    source_url: Optional[str] = None,
    doc_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> str:
    """Build the citation label for a single pack item.

    Profile items -> ``[doc_id:chunk_id]``-style label.
    URL items    -> ``[host<truncated-path>]`` so users can see URL
    provenance without opening the chunk.
    """
    if source_url:
        from urllib.parse import urlparse
        parsed = urlparse(source_url)
        host = parsed.hostname or ""
        path = parsed.path or ""
        if len(path) > 32:
            path = path[:29] + "..."
        return f"[{host}{path}]"

    parts: List[str] = []
    if doc_id:
        parts.append(doc_id[:16])
    if chunk_id:
        parts.append(f"#{chunk_id[:8]}")
    return "[" + "".join(parts) + "]" if parts else "[source]"


SUPPLEMENTARY_PROMPT_TEMPLATE = (
    "You have just answered a user's question using evidence from their profile\n"
    "documents. A URL they included was fetched in parallel and has now completed.\n"
    "Write a short supplementary analysis section that:\n\n"
    "- clearly separates URL-derived observations from the profile-derived answer above,\n"
    "- labels each URL claim with its source host,\n"
    "- flags any conflict between URL content and the primary response.\n\n"
    "Primary response:\n"
    '"""{primary_response}"""\n\n'
    "URL content (one block per chunk):\n"
    "{url_blocks}\n\n"
    'Produce the supplementary section as markdown under the heading '
    '"## Supplementary (from URL)".\n'
    "Do not repeat the primary response. Do not fabricate; omit rather than guess."
)


def build_supplementary_prompt(
    *,
    primary_response: str,
    url_chunks: Sequence[Any],
    adapter: Optional[Any] = None,
) -> str:
    """Construct the continuation prompt for the late-arrival URL supplement.

    ``url_chunks`` may be a list of dicts (``{"text", "source_url"}``) or
    objects exposing ``.text`` + ``.metadata["source_url"]`` — matches the
    shapes emitted by :class:`EphemeralChunk` and the merge helper.
    ``adapter`` is optional; when present its ``.persona`` surface is
    consulted for a rich-mode preamble, otherwise we fall back to the
    minimal compact-friendly template.
    """
    if not url_chunks:
        raise ValueError(
            "build_supplementary_prompt requires at least one url chunk"
        )

    blocks: List[str] = []
    for i, chunk in enumerate(url_chunks, start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            source_url = chunk.get("source_url") or chunk.get("url", "url")
        else:
            text = getattr(chunk, "text", "")
            md = getattr(chunk, "metadata", {}) or {}
            source_url = md.get("source_url", "url")
        blocks.append(f"[{i}] ({source_url}) {text}")

    rendered = SUPPLEMENTARY_PROMPT_TEMPLATE.format(
        primary_response=primary_response,
        url_blocks="\n\n".join(blocks),
    )

    persona = getattr(adapter, "persona", None) if adapter is not None else None
    if persona:
        role = getattr(persona, "role", None) or (
            persona.get("role") if isinstance(persona, dict) else None
        )
        if role:
            rendered = f"You are writing as: {role}.\n\n" + rendered

    return rendered

