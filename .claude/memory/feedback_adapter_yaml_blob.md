---
name: SME Adapter YAML (and Referenced Templates) Live in Azure Blob, Not Repo
description: Domain adapter YAML files and any prompt/template files they reference are stored in Azure Blob, not the code repository. Enables hot-swap without redeploy, per-subscription overrides, versioning, and audit trail. Extends the existing storage separation rule (blob = document content AND configuration as data, Mongo = control plane only, Qdrant = vectors, Neo4j = graph).
type: feedback
originSessionId: 56b70947-9824-48b4-9a97-b3d2d50b0d88
---
DocWain's domain adapter YAML files (generic, finance, legal, hr, medical, it_support, and any custom domain adapters added later) live in Azure Blob Storage. Any prompt templates or reference files an adapter loads also live in Blob under the same scheme. Code ships with NO hardcoded adapter YAMLs baked into the repository.

**Why:** Stated by the user on 2026-04-20 while reviewing the Profile-SME design: "just ensure all the .yaml files are stored in azure blob." Matches the user's existing storage separation discipline (document content and now configuration-as-data belongs in Blob, not repo or MongoDB). Gives operators: hot-swap without redeploy, per-subscription adapter overrides for enterprise multi-tenancy, native versioning via Blob's snapshot feature, and full audit trail for compliance.

**How to apply:**
- Adapter files live at `sme_adapters/global/{domain}.yaml` for defaults and `sme_adapters/subscription/{subscription_id}/{domain}.yaml` for per-subscription overrides. Resolution: subscription-specific first, fall back to global.
- Referenced templates (e.g., `prompts/finance_dossier.md`) live alongside under `sme_adapters/global/prompts/` and `sme_adapters/subscription/{subscription_id}/prompts/`.
- Service loads adapters on first use, caches in-memory with TTL (~5 min), and supports invalidation via admin API when an adapter is updated.
- Bootstrap: deployment playbook uploads default YAMLs to Blob on first install; code never embeds them.
- Integrity: every load logs the adapter version and content hash for audit; SME synthesis traces record which adapter version produced each artifact.
- Failure mode: if Blob fetch fails, fall back to last-cached version; if no cache, fall back to the `generic` adapter (must still succeed); surface the failure in health checks.
- When adding a new domain, operators upload the YAML to Blob; no code deploy required.
