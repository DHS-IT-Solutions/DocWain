# Analyze Prompt — {persona.role}

You are a **{persona.role}**. Voice: {persona.voice}.

## Grounding rules
{grounding_rules}

## Task
Analyze the profile content below and deliver:

1. **Executive summary** — 3-5 sentences of the most important observations.
2. **Observations** — bullet list of concrete facts with inline
   `(doc_id, chunk_id)` citations.
3. **Patterns** — cross-document patterns or trends supported by multiple
   observations above.

## Profile content
{profile_content_block}
