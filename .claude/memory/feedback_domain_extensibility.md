---
name: DocWain Must Stay Domain-Agnostic and Open-Set
description: DocWain is a generic document-intelligence product; every new feature (intelligence layer, SME personas, artifact templates, KG ontology, domain adapters) must be extensible to new domains without code changes and must not be fixed to a closed list of domains. A generic/default path must always work for unknown or unclassified content.
type: feedback
originSessionId: 56b70947-9824-48b4-9a97-b3d2d50b0d88
---
DocWain serves any type of documents provided and must not get locked to a fixed, closed set of domains or reasoning approaches. Every design that involves domain-specific behavior (SME personas, artifact shapes, reasoning templates, KG ontology extensions, prompts) must be plugin-shaped: adding a new domain should mean dropping in a config/adapter file, not editing core code or redeploying. A `generic` default path must always work for unknown or unclassified profiles so the system degrades gracefully rather than failing.

**Why:** Stated by the user on 2026-04-20 during the Profile-SME reasoning design: "DocWain is a generic product that works for any type of documents provided and should not get fixed to a single or closed approaches." Echoes the existing canonical pipeline rule (domain-agnostic) but extends it: the system must stay open-set as customer domains grow, and generic behavior must never fail.

**How to apply:**
- Domain-specific logic lives in adapters (YAML/JSON + optional Python helper) loaded at runtime, not in core modules.
- Every artifact builder, reasoning template, and persona ships a `generic` default that works on any profile.
- Auto-detection plus user override is the pattern; `unknown → generic` is always safe.
- KG ontology extensions register new node/edge types via config, not by editing ontology.py's core.
- Avoid hard-coded `if domain == 'finance'` branches in core paths; route through adapter lookup instead.
- When reviewing designs, flag any closed enum of domains or reasoning modes that doesn't expose an extension point.
