---
name: MongoDB Status Values Are Immutable
description: UI relies on exact MongoDB pipeline_status and status field values — never rename, remap, or remove them.
type: feedback
originSessionId: 93168fda-607e-4c51-b06c-5b5e0f18a6b1
---
MongoDB `pipeline_status` and `status` field values (defined in `src/api/statuses.py`) are a stable contract between backend and UI. User reaffirmed on 2026-04-17: "Ensure not to change any of the existing mongo db status changes as UI relies on these changes."

Never:
- Rename status constants (e.g., `TRAINING_COMPLETED` stays `TRAINING_COMPLETED`).
- Remove legacy aliases — even deprecated ones may still be read by UI.
- Change the transition points where each status is set (UI polls and reacts to these specific moments).
- Remap one status string to another without explicit UI coordination.

Canonical status values in `src/api/statuses.py`:
- Pipeline: `UPLOADED`, `EXTRACTION_IN_PROGRESS`, `EXTRACTION_COMPLETED`, `EXTRACTION_FAILED`, `SCREENING_IN_PROGRESS`, `SCREENING_COMPLETED`, `SCREENING_FAILED`, `EMBEDDING_IN_PROGRESS`, `TRAINING_COMPLETED`, `EMBEDDING_FAILED`
- HITL gates: `AWAITING_REVIEW_1`, `AWAITING_REVIEW_2`, `REJECTED`, `PROCESSING_IN_PROGRESS`, `PROCESSING_COMPLETED`, `PROCESSING_FAILED`
- Legacy/terminal: `UNDER_REVIEW`, `TRAINING_PARTIALLY_COMPLETED`, `TRAINING_BLOCKED_SECURITY`, `TRAINING_BLOCKED_CONFIDENTIAL`, `DELETED`
- KG (independent): `PENDING`, `IN_PROGRESS`, `COMPLETED`, `FAILED`

**Why:** Breaking these silently breaks the UI — user cannot see or act on their documents.

**How to apply:**
- When touching any code in `src/api/document_status.py`, `src/api/statuses.py`, or pipeline handlers, preserve every existing string value and every transition point.
- If a genuinely new status is required, ADD it — never repurpose an old one.
- Before adding a new status, check whether the UI side also needs an update; the two must change together.
