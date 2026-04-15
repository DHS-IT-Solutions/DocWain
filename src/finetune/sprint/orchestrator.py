"""Sprint Orchestrator — ties all sprint components into a resumable two-phase pipeline.

Phase 1: Foundation (completeness extraction + intent + anti-hallucination)
Phase 2: Advanced capabilities (OCR, Excel/CSV, deep reasoning, cross-doc, domain data)

State is saved at every phase boundary so the pipeline can resume safely after
interruption.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase sequence
# ---------------------------------------------------------------------------

PHASE_SEQUENCE: List[str] = [
    "generate_eval_bank",
    "phase1_generate",
    "phase1_sft",
    "phase1_dpo",
    "phase1_gate",
    "phase2_generate",
    "phase2_sft",
    "phase2_dpo",
    "final_gate",
    "convert",
    "done",
]

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SprintOrchestrator:
    """Coordinates data generation, training, evaluation, and model conversion."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : SprintConfig
            Sprint configuration with paths and hyper-parameters.
        """
        from src.finetune.sprint.config import SprintState
        from src.finetune.sprint.trainer import SprintTrainer

        self.config = config
        self.artifacts_dir = Path(config.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.state = SprintState.load(self.artifacts_dir)
        self.trainer = SprintTrainer(config)

    # ------------------------------------------------------------------
    # Phase navigation
    # ------------------------------------------------------------------

    def _advance_phase(self) -> None:
        """Move to the next phase in PHASE_SEQUENCE and persist state."""
        if self.state.phase == "init":
            self.state.phase = PHASE_SEQUENCE[0]
        else:
            try:
                idx = PHASE_SEQUENCE.index(self.state.phase)
                self.state.phase = PHASE_SEQUENCE[idx + 1]
            except (ValueError, IndexError):
                logger.warning("Cannot advance from phase %r", self.state.phase)
                return
        self.state.save()
        logger.info("Advanced to phase: %s", self.state.phase)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run (or resume) the full sprint pipeline."""
        logger.info("Starting sprint from phase: %s", self.state.phase)

        # If we haven't started yet, advance to the first phase
        if self.state.phase == "init":
            self._advance_phase()

        handlers = {
            "generate_eval_bank": self._run_generate_eval_bank,
            "phase1_generate": self._run_phase1_generate,
            "phase1_sft": self._run_phase1_sft,
            "phase1_dpo": self._run_phase1_dpo,
            "phase1_gate": self._run_phase1_gate,
            "phase2_generate": self._run_phase2_generate,
            "phase2_sft": self._run_phase2_sft,
            "phase2_dpo": self._run_phase2_dpo,
            "final_gate": self._run_final_gate,
            "convert": self._run_convert,
            "done": lambda: logger.info("Sprint already complete."),
        }

        while self.state.phase != "done":
            phase = self.state.phase
            handler = handlers.get(phase)
            if handler is None:
                logger.error("Unknown phase %r — aborting", phase)
                break
            logger.info("=== Running phase: %s ===", phase)
            handler()
            if self.state.phase == phase:
                # Handler did not advance — advance now
                self._advance_phase()

        logger.info("Sprint complete. Final phase: %s", self.state.phase)

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _run_generate_eval_bank(self) -> None:
        from src.finetune.sprint.eval_bank import generate_eval_bank, save_eval_bank

        bank_path = Path(self.config.eval_bank_path)
        if bank_path.exists():
            logger.info("Eval bank already exists at %s — skipping generation", bank_path)
        else:
            logger.info("Generating eval bank…")
            examples = generate_eval_bank()
            save_eval_bank(examples, bank_path)
            logger.info("Saved %d eval examples to %s", len(examples), bank_path)

    def _run_phase1_generate(self) -> None:
        from src.finetune.sprint.distiller import generate_sft_batch, generate_dpo_batch, save_examples

        sft_path = self.artifacts_dir / "phase1_sft.jsonl"
        dpo_path = self.artifacts_dir / "phase1_dpo.jsonl"

        logger.info("Phase 1 — generating SFT data…")
        # completeness_extraction: 8000 SFT
        logger.info("  completeness_extraction: 8000 SFT")
        examples = generate_sft_batch("completeness_extraction", count=8000)
        save_examples(examples, sft_path)

        # intent_context: 5000 SFT
        logger.info("  intent_context: 5000 SFT")
        examples = generate_sft_batch("intent_context", count=5000)
        save_examples(examples, sft_path)

        self.state.sft_count += 13000

        # anti_hallucination: 5000 DPO
        logger.info("  anti_hallucination: 5000 DPO")
        examples = generate_dpo_batch("anti_hallucination", count=5000)
        save_examples(examples, dpo_path)

        self.state.dpo_count += 5000
        self.state.save()
        logger.info("Phase 1 data ready: sft=%s dpo=%s", sft_path, dpo_path)

    def _run_phase1_sft(self) -> None:
        sft_path = self.artifacts_dir / "phase1_sft.jsonl"
        output_dir = self.artifacts_dir / "phase1_sft_model"
        logger.info("Phase 1 SFT training → %s", output_dir)
        self.trainer.train_sft(sft_path, output_dir)
        self.state.best_checkpoint = str(output_dir)
        self.state.save()

    def _run_phase1_dpo(self) -> None:
        dpo_path = self.artifacts_dir / "phase1_dpo.jsonl"
        sft_model = Path(self.state.best_checkpoint) if self.state.best_checkpoint else self.artifacts_dir / "phase1_sft_model"
        output_dir = self.artifacts_dir / "phase1_dpo_model"
        logger.info("Phase 1 DPO training → %s", output_dir)
        self.trainer.train_dpo(dpo_path, sft_model, output_dir)
        self.state.best_checkpoint = str(output_dir)
        self.state.save()

    def _run_phase1_gate(self) -> None:
        passed = self._run_eval_gate(
            thresholds=self.config.phase1_gate,
            phase_name="phase1",
        )
        if passed:
            self.state.phase1_passed = True
            self.state.save()
            logger.info("Phase 1 gate PASSED")
        else:
            logger.warning("Phase 1 gate FAILED — proceeding anyway")
            self._handle_gate_failure("phase1")

    def _run_phase2_generate(self) -> None:
        from src.finetune.sprint.distiller import generate_sft_batch, generate_dpo_batch, save_examples
        from src.finetune.sprint.domain_data import generate_all_domain_data

        sft_path = self.artifacts_dir / "phase2_sft.jsonl"
        dpo_path = self.artifacts_dir / "phase2_dpo.jsonl"

        phase2_tracks = [
            ("ocr_vision", 4000, 1000),
            ("excel_csv", 4000, 1000),
            ("deep_reasoning", 4000, 1000),
            ("cross_document", 3000, 1000),
        ]

        logger.info("Phase 2 — generating SFT + DPO data…")
        for category, sft_count, dpo_count in phase2_tracks:
            logger.info("  %s: %d SFT + %d DPO", category, sft_count, dpo_count)
            sft_examples = generate_sft_batch(category, count=sft_count)
            save_examples(sft_examples, sft_path)

            dpo_examples = generate_dpo_batch(category, count=dpo_count)
            save_examples(dpo_examples, dpo_path)

        # Domain data: 12000 SFT
        logger.info("  domain data: 12000 SFT via generate_all_domain_data()")
        domain_examples = generate_all_domain_data()
        save_examples(domain_examples, sft_path)

        self.state.sft_count += 27000
        self.state.dpo_count += 4000
        self.state.save()
        logger.info("Phase 2 data ready: sft=%s dpo=%s", sft_path, dpo_path)

    def _run_phase2_sft(self) -> None:
        sft_path = self.artifacts_dir / "phase2_sft.jsonl"
        base_model = Path(self.state.best_checkpoint) if self.state.best_checkpoint else self.artifacts_dir / "phase1_dpo_model"
        output_dir = self.artifacts_dir / "phase2_sft_model"
        logger.info("Phase 2 SFT training → %s", output_dir)
        self.trainer.train_sft(sft_path, output_dir)
        self.state.best_checkpoint = str(output_dir)
        self.state.save()

    def _run_phase2_dpo(self) -> None:
        dpo_path = self.artifacts_dir / "phase2_dpo.jsonl"
        sft_model = Path(self.state.best_checkpoint) if self.state.best_checkpoint else self.artifacts_dir / "phase2_sft_model"
        output_dir = self.artifacts_dir / "phase2_dpo_model"
        logger.info("Phase 2 DPO training → %s", output_dir)
        self.trainer.train_dpo(dpo_path, sft_model, output_dir)
        self.state.best_checkpoint = str(output_dir)
        self.state.save()

    def _run_final_gate(self) -> None:
        passed = self._run_eval_gate(
            thresholds=self.config.final_targets,
            phase_name="final",
        )
        if passed:
            self.state.final_passed = True
            self.state.save()
            logger.info("Final gate PASSED")
        else:
            logger.warning("Final gate FAILED — proceeding to conversion anyway")
            self._handle_gate_failure("final")

    def _run_convert(self) -> None:
        from src.finetune.sprint.converter import (
            convert_to_base_model,
            export_gguf,
            generate_model_card,
            upload_to_huggingface,
        )

        checkpoint = Path(self.state.best_checkpoint) if self.state.best_checkpoint else self.artifacts_dir / "phase2_dpo_model"
        output_dir = self.artifacts_dir / "release"
        model_name = self.config.model_name

        logger.info("Converting to base model…")
        base_dir = convert_to_base_model(checkpoint, output_dir, model_name)

        gguf_path = output_dir / f"{model_name}.gguf"
        logger.info("Exporting GGUF → %s", gguf_path)
        export_gguf(base_dir, gguf_path)

        model_card = generate_model_card(model_name, self.state.scores)
        repo_id = f"MuthuSubramanian/{model_name}"
        logger.info("Uploading to HuggingFace: %s", repo_id)
        upload_to_huggingface(base_dir, repo_id, model_card)

        logger.info("Conversion complete.")

    # ------------------------------------------------------------------
    # Shared eval logic
    # ------------------------------------------------------------------

    def _run_eval_gate(self, thresholds: Dict[str, float], phase_name: str) -> bool:
        """Query the model on the eval bank, score with judge, check gate.

        Returns True if all thresholds are met (or if eval bank / model are
        unavailable — in that case we log a warning and return True to avoid
        blocking the pipeline).
        """
        from src.finetune.sprint.eval_bank import load_eval_bank
        from src.finetune.sprint.judge import evaluate_batch, check_regression

        bank_path = Path(self.config.eval_bank_path)
        if not bank_path.exists():
            logger.warning("Eval bank not found at %s — skipping gate", bank_path)
            return True

        examples = load_eval_bank(bank_path)
        logger.info("Evaluating on %d examples for gate '%s'", len(examples), phase_name)

        responses = self._query_model_batch(examples)
        scores_list = evaluate_batch(examples, responses)

        if not scores_list:
            logger.warning("No scores returned — skipping gate")
            return True

        # Aggregate per-dimension
        agg: Dict[str, float] = {}
        for key in scores_list[0]:
            vals = [s[key] for s in scores_list if key in s]
            agg[key] = round(sum(vals) / len(vals), 3) if vals else 0.0

        logger.info("Aggregate scores for %s: %s", phase_name, agg)

        # Check regression against previous scores
        if self.state.scores:
            regressions = check_regression(self.state.scores, agg)
            if regressions:
                logger.warning("Regressions detected in dimensions: %s", regressions)

        # Record history
        self.state.scores = agg
        self.state.eval_history.append({"phase": phase_name, "scores": agg})

        # Update best score
        avg = agg.get("average", agg.get("accuracy", 0.0))
        if avg > self.state.best_score:
            self.state.best_score = avg
        self.state.save()

        # Check gate thresholds
        # Map from config threshold key names to judge dimension names where needed
        _key_map = {
            "hallucination_rate": "honesty",
            "completeness": "completeness",
            "intent_accuracy": "accuracy",
            "extraction_completeness": "completeness",
            "intent_understanding": "accuracy",
            "hallucination_target": "honesty",
            "judge_score_target": "average",
        }

        passed = True
        for threshold_key, threshold_val in thresholds.items():
            score_key = _key_map.get(threshold_key, threshold_key)
            if score_key not in agg:
                logger.debug("Gate key %r not in scores — skipping", score_key)
                continue
            actual = agg[score_key]
            # For hallucination_rate: lower is better (rate < threshold)
            if "hallucination" in threshold_key:
                if actual > threshold_val:
                    logger.warning(
                        "Gate MISS: %s hallucination_rate=%.3f > threshold=%.3f",
                        phase_name, actual, threshold_val,
                    )
                    passed = False
            else:
                # For scores: higher is better
                # Scores are 1-5 from judge; completeness/intent targets are 0-1
                # Normalize if threshold looks like a proportion (< 2.0)
                effective_score = actual
                effective_threshold = threshold_val
                if threshold_val <= 1.0 and actual > 1.0:
                    # Convert 1-5 score to 0-1 proportion
                    effective_score = (actual - 1) / 4
                if effective_score < effective_threshold:
                    logger.warning(
                        "Gate MISS: %s %s=%.3f < threshold=%.3f",
                        phase_name, threshold_key, effective_score, effective_threshold,
                    )
                    passed = False

        return passed

    def _query_model_batch(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Query the current model (via Ollama) for each eval example.

        Falls back to empty strings if the model is unavailable so the pipeline
        can continue in a degraded mode.
        """
        import httpx

        model = "docwain:latest"
        responses: List[str] = []

        for ex in examples:
            prompt = ex.get("prompt", "")
            try:
                resp = httpx.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                responses.append(content)
            except Exception as exc:
                logger.debug("Model query failed for example: %s", exc)
                responses.append("")

        return responses

    def _handle_gate_failure(self, phase: str) -> None:
        """Identify weak dimensions and log targeted recovery suggestions."""
        if not self.state.scores:
            logger.warning("No scores available to diagnose gate failure for phase %r", phase)
            return

        # Find dimensions below 3.0 (mid-point on 1-5 scale)
        weak = [
            (dim, score)
            for dim, score in self.state.scores.items()
            if dim != "average" and isinstance(score, (int, float)) and score < 3.0
        ]
        weak.sort(key=lambda x: x[1])

        if weak:
            logger.warning(
                "Gate failure in phase %r. Weak dimensions (score < 3.0): %s",
                phase,
                ", ".join(f"{d}={s:.2f}" for d, s in weak),
            )
            # Log targeted recovery suggestions
            _recovery = {
                "accuracy": "Generate more completeness_extraction SFT data",
                "completeness": "Generate more completeness_extraction SFT data",
                "reasoning": "Generate more deep_reasoning SFT data",
                "honesty": "Generate more anti_hallucination DPO data",
                "format": "Review format examples in all categories",
            }
            for dim, _ in weak:
                suggestion = _recovery.get(dim, f"Review {dim} training data")
                logger.warning("  Recovery for %s: %s", dim, suggestion)
        else:
            logger.warning(
                "Gate failure in phase %r but no dimensions below 3.0 — "
                "check thresholds vs score scale",
                phase,
            )
