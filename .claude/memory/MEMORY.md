# DocWain Project Memory

## User
- [Muthu - Project Owner](user_muthu.md) — DocWain owner, values accuracy over latency, wants enterprise-grade production quality

## Feedback
- [Pipeline Flow Rules](feedback_pipeline_flow.md) — Upload auto-triggers extraction; screening and training/embedding are HITL-triggered; KG belongs in training (not screening)
- [MongoDB Status Values Immutable](feedback_mongo_status_stability.md) — Never rename/remap/remove pipeline_status strings; UI contract
- [Intelligence Precomputed at Ingestion](feedback_intelligence_precompute.md) — All intelligence built in training stage, only looked up at query time
- [Measure Before You Change](feedback_measure_before_change.md) — No accuracy/quality work without a fresh baseline + measurement harness in place first
- [Storage Separation Rules](feedback_storage_separation.md) — MongoDB is control plane only, document content goes to Azure Blob and Qdrant
- [LLM Prompt Code Paths](feedback_prompt_paths.md) — Response formatting goes in generation/prompts.py (Reasoner path), NOT intelligence/generator.py
- [Finetuning Pipeline Trigger](feedback_finetuning_trigger.md) — "initiate finetuning pipeline" in prompt → start iterative SFT→DPO→eval→retrain loop
- [No Claude Attribution](feedback_no_claude_attribution.md) — Never include Co-Authored-By, Claude, or Anthropic references in commits, code, or docs
- [No Customer Data in Training](feedback_no_customer_data_training.md) — Only synthetic data for finetuning, never customer documents
- [Teams App Must Be Fully Isolated](feedback_teams_isolation.md) — Never call main app APIs, never modify src/ for Teams — fully self-contained
- [Teams UX Rules](feedback_teams_ux.md) — messageBack buttons, plain text responses, auto-clear on upload, card-only progress updates

## Feedback (continued)
- [Base Model Approach](feedback_base_model_approach.md) — DocWain must be a base model with identity baked into weights, no system prompt dependency
- [Unified Model (No Fast/Smart Split)](feedback_unified_model.md) — One unified DocWain model; active symlink always points to best-scoring checkpoint
- [V5 Pipeline Failure — Key Lessons](feedback_v5_failure_lessons.md) — 8 hard rules from the 2026-04-20 V5 failure: gate distillation on teacher identity, validate scorers first, don't advance on failed gates
- [Engineering First, Model Training Last](feedback_engineering_first_model_last.md) — Prove intelligence changes at prompts/retrieval/reasoning layer first, capture patterns, only retrain when high-confidence
- [Domain-Agnostic & Open-Set](feedback_domain_extensibility.md) — DocWain works on any doc type; domain specializations must be plugin-shaped adapters, never a closed enum, always with a working generic default
- [No Timeouts; Use Efficiency](feedback_no_timeouts.md) — Latency varies by complexity and that's fine; never cut off a response on a wall-clock timer. Handle latency through parallelism, streaming, budgets, caps, caching
- [Adapter YAMLs in Azure Blob](feedback_adapter_yaml_blob.md) — SME domain adapter YAMLs and referenced templates live in Azure Blob (not repo); global + per-subscription override paths; hot-swappable
- [Intelligence/RAG Zero-Error Rule](feedback_intelligence_rag_zero_error.md) — 2026-04-21 workstream: every batch mechanically verifiable, pipeline-isolated, single-flag revertible; no "fix it next PR"

## Project
- [DocWain Canonical Pipeline](project_docwain_pipeline.md) — Three-stage HITL pipeline, domain-agnostic, user's canonical description 2026-04-17 (SUPERSEDES stale pipeline notes)
- [Enterprise Architecture Redesign](project_enterprise_architecture.md) — Major redesign approved 2026-03-15, new pipeline with Celery workers, plugin screening, KG-enhanced RAG
- [Auto Fine-Tune Daily Schedule](project_auto_finetune.md) — Daily feedback-driven fine-tune loop for MuthuSubramanian/DocWain model
- [DocWain V2 Unified Model](project_docwain_v2.md) — Vision-grafted Qwen3-14B with native tool-calling, 4-phase training pipeline
- [vLLM Production Integration](project_vllm_integration.md) — Code complete, systemd installed, awaiting first startup after training frees GPU. Phase 2: agentic layer
- [V2 Training Pipeline Status](project_v2_training_status.md) — 3 bugs fixed, excel_csv track iter 3 running, score 3.71/4.0 and improving. Resume instructions inside
- [Teams Standalone Service](project_teams_standalone.md) — Separate systemd service (port 8300), fully self-contained with own embedder, query handler, LLM gateway, APIM routing
- [Weekend Finetuning Apr 2026](project_finetuning_weekend_apr2026.md) — V3 training: SFT 0.127, 31K examples, 86% extraction. Next: base model conversion, V4 with element data
- [V2 Weights — HF Recovery Path](project_v2_weights_recovery.md) — 2026-04-17: HF repo `muthugsubramanian/DocWain-14B-v2` downloaded to `models/DocWain-14B-v2/`, active symlink repointed, restart now safe
- [V5 Sprint Post-Mortem](project_v5_post_mortem.md) — 2026-04-20: V5 reverted, V2 back on prod; V5 artifacts preserved on disk for possible V6 or teardown
- [preprod_v01 Is Quality Baseline](project_preprod_v01_quality.md) — 2026-04-22: preprod_v01 produces better responses than main; prod runs preprod_v01 standalone, main's 417 extra commits regressed quality
- [Post-preprod Roadmap (5 items)](project_post_preprod_roadmap.md) — 2026-04-23 directives: cherry-pick teams_app, rebuild standalone, revisit extraction, upgrade RAG, decide vLLM vs Ollama — one step at a time
- [Backend Quality Audit 2026-04-23](project_audit_2026_04_23_backend_quality.md) — Cloud 397B wins intelligence; Ollama local hallucinates/loops; same-weights engine swap does NOT close the model-scale gap. "Better preprod_v01 responses" is the Cloud model, not the code.
- [DocWain as Research Agent (not Q&A)](project_researcher_agent_vision.md) — 2026-04-23 product-direction upgrade: backend Researcher Agent runs after screening, performs domain-aware deep analysis, persists insights; extraction accuracy is the primary foundation; KG consolidated into training stage.
- [preprod_v02 Implementation Branch](project_preprod_v02_branch.md) — 2026-04-23: new long-lived branch off preprod_v01 hosts all post-audit roadmap implementations (extraction overhaul first); worktree at ~/.config/superpowers/worktrees/DocWain/preprod_v02
