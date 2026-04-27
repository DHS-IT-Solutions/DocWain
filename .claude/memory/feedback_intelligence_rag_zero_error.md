---
name: Intelligence/RAG Re-integration — Zero-Error Operating Rule
description: During the 2026-04-21 intelligence+RAG re-integration workstream, the user explicitly set a no-room-for-errors bar; every batch must be verifiable, pipeline-isolated, and single-flag revertible
type: feedback
originSessionId: 97be8d9c-e3ba-42b5-9ff0-d1704500f888
---
For the intelligence layer + RAG re-integration workstream (spec: `docs/superpowers/specs/2026-04-21-intelligence-rag-redesign-design.md`), the user set a hard operating bar: "I have no room for errors." This shapes how every batch is prepared and merged.

**Why:** The user has already spent significant time on V5 (which failed, reverted on 2026-04-20), on the pipeline fix that shipped today, and on the SME branch. A further regression on top of today's accuracy drop would compound the lost time. Quality and reversibility beat speed on every trade-off.

**How to apply:**

- Never merge a batch without its §6.1 exit criteria mechanically verified. "Looks right" is not an exit criterion; commands with pass/fail outputs are.
- Every batch ships on its own branch `batch-N-<desc>`, never pushed to `main` directly. One batch per day minimum, never two on the same day.
- Every PR description includes both the `git revert <sha>` rollback command and the flag-writes that would roll back without a revert.
- The document-processing pipeline is off-limits in every batch. If a cherry-pick touches the do-not-touch list in the spec §4.4, quarantine that change to a follow-up PR — do not merge it in the intelligence batch.
- The "flag-OFF within tolerance" invariant (spec §6.1 + §6.1.1) must be proven per batch with both numbers and a retrieval-call trace diff. Numbers alone are insufficient — two batches can produce identical numbers while silently changing code paths.
- Batch 0 gets a canary on the owner's profile for ≥ 1 hour of real queries before rolling to prod.
- If an eval gate fails, the batch does not merge. No "merge and we'll fix it in the next PR" allowances.
