---
name: preprod_v01 produces better responses than main
description: As of 2026-04-22 the preprod_v01 branch generates more intelligent / better-quality responses than main, despite main having 417 newer commits
type: project
originSessionId: dc7597b6-0d4a-464a-8305-e7a3b998992a
---
On 2026-04-22 Muthu identified that the `preprod_v01` branch produces better, more intelligent responses than `main`, despite `main` having ~417 commits of fixes landed after `preprod_v01` diverged (demo hints, extract-progress, prompt compaction, structured-answer, json-extract, gateway cleanups, wire-vllm-gateway-primary, etc.). The decision was to run production on `preprod_v01` (standalone codebase, no merge from main).

**Why:** A month+ of work on main regressed output quality. User frame: "wasted more than a month of effort, time and money." `preprod_v01` is the known-good baseline for response quality.

**How to apply:**
- Do NOT blindly merge `main` into `preprod_v01`. Treat `preprod_v01` as the current production truth.
- Before reintroducing anything from main's 417 commits, prove it doesn't regress response quality against a `preprod_v01` baseline.
- When investigating quality regressions, bisect against commits added to `main` after `preprod_v01` diverged — especially prompt / structured-answer / RAG-generation commits (PRs #10–#19).
- Suspect commits to audit first (touch the generation/prompt path):
  - `efa8a20` fix(generation): compact prompt payload + converge vLLM on vllm_manager
  - `a1cd0d5` fix(prompt): require natural-language answer field in structured output
  - `aec9450` fix(generation): cap max_tokens + relax LLM-output validation
  - `d38e3fd` fix(structured-answer): stop asking LLM to echo documents; graft server-side
  - `ec37f05` fix(demo): include snippets in prompt, skip router LLM, cap compare pairs
  - `8027caa` fix(demo): explicit schema hint + doc count + name-index in prompt
  - `5501a71` / `cb07632` fix(extract-progress): scope aggregates, rename common_data.uploaded → Extracted
- `standalone/` and `teams_app/` do not exist on `preprod_v01`; those services cannot run on this branch.
