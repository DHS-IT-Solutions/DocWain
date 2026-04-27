---
name: preprod_v02 is the implementation branch for post-audit upgrades
description: Created 2026-04-23 off preprod_v01 as the next-iteration baseline — all post-preprod roadmap implementations land here
type: project
originSessionId: dc7597b6-0d4a-464a-8305-e7a3b998992a
---
On 2026-04-23, after the backend-quality audit and the extraction-accuracy brainstorming, Muthu directed that all roadmap implementations go on a new branch `preprod_v02`, branched off `preprod_v01`.

**Worktree location:** `~/.config/superpowers/worktrees/DocWain/preprod_v02` (global worktrees pattern established by prior work).

**Branch base:** `preprod_v01` at commit `1da060c`.

**Purpose:** preprod_v02 is the long-lived implementation line for the post-audit roadmap items:
- Extraction accuracy overhaul (spec: `docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md`)
- Researcher Agent + RAG integration (separate spec, later)
- KG consolidation into training stage (separate spec, later)
- Serving swap to vLLM-primary (separate spec, later)
- teams_app cherry-pick + standalone rebuild (last)

Once the extraction overhaul is mature and validated against its bench, `preprod_v02` becomes the next production baseline, replacing `preprod_v01`.

**Why:** Separate long-lived branch keeps preprod_v01 as the known-good quality baseline while work proceeds. Does not touch main (main is quarantined per `project_preprod_v01_quality.md`).

**How to apply:**
- All implementation work for the 5-item roadmap lands on `preprod_v02` or short-lived feature branches off it — never direct to preprod_v01, never to main.
- Specs live in `docs/superpowers/specs/` on preprod_v02 (force-added because `docs/` is gitignored; precedent set earlier).
- Plans live in `docs/superpowers/plans/` on preprod_v02.
- Test benches (e.g., `tests/extraction_bench/`) are tracked and version-controlled on this branch.
- No Claude/Anthropic/Co-Authored-By attribution in commit messages.
