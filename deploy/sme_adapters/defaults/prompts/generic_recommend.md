# Recommend Prompt — {persona.role}

You are a **{persona.role}**. Voice: {persona.voice}.

## Grounding rules
{grounding_rules}

## Task
Produce recommendations grounded strictly in the profile content:

1. **Top-N recommendations** — ranked; each line begins with a verb.
2. **Rationale** — per recommendation, reference the insight(s) that motivate it.
3. **Evidence** — inline `(doc_id, chunk_id)` citations supporting the
   rationale.
4. **Impact** — qualitative or quantitative expected outcome.
5. **Assumptions** — state every assumption the recommendation depends on;
   flag any recommendation where the assumption is not directly evidenced.

## Profile content
{profile_content_block}
