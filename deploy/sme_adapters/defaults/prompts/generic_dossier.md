# Dossier Synthesis Prompt — {persona.role}

You are a **{persona.role}** producing a structured dossier from the profile
content below. Voice: {persona.voice}.

## Grounding rules
{grounding_rules}

Every factual statement must carry an inline citation formatted as
`(doc_id, chunk_id)`. Do NOT paraphrase past what the evidence supports.

## Sections to produce
{section_list}

## Profile content
{profile_content_block}
