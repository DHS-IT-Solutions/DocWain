"""Phase 4 — Merge adapters into GGUF and promote to Ollama.

After all three training phases complete, this module:
1. Merges LoRA adapters back into the base model.
2. Quantises to GGUF (Q4_K_M by default).
3. Generates a V2 Modelfile with DocWain persona, vision, and tool-calling.
4. Runs regression tests against V1 baselines.
5. Promotes to Ollama as ``docwain:v2`` (and optionally ``docwain:latest``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Phase4Config:
    """Configuration for the merge-and-promote step."""

    # Input paths
    phase3_dir: Path = Path("finetune_artifacts/v2/phase3_7")
    base_model: str = "unsloth/Qwen3-14B"

    # Merged output
    merged_dir: Path = Path("finetune_artifacts/v2/merged")

    # Quantisation
    quant_method: str = "q4_k_m"
    gguf_output_dir: Path = Path("models/docwain-v2")

    # Ollama
    ollama_model_name: str = "docwain"
    ollama_tag_v2: str = "v2"
    ollama_tag_latest: str = "latest"

    # Regression
    min_regression_pass_rate: float = 0.90


# ---------------------------------------------------------------------------
# Modelfile generation
# ---------------------------------------------------------------------------


def generate_v2_modelfile(gguf_path: str) -> str:
    """Generate an Ollama Modelfile for DocWain V2.

    The Modelfile includes:
    - The GGUF model path
    - DocWain persona system prompt with vision and tool-calling instructions
    - Recommended inference parameters

    Parameters
    ----------
    gguf_path:
        Absolute or relative path to the quantised GGUF file.

    Returns
    -------
    The complete Modelfile content as a string.
    """
    return f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{- if .System }}}}
<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
{{{{- range .Messages }}}}
<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"You are DocWain, an enterprise document intelligence assistant created by MuthuSubramanian.

You have full vision capabilities — you can see and analyse document images, pages, tables, charts, diagrams, and infographics. When a user shares a document image, examine it carefully and provide accurate, grounded answers.

You have access to the following tool_call functions for document analysis:
- ocr_extract: Vision-based text extraction from document pages
- layout_extract: Structural layout detection (headings, paragraphs, tables, figures)
- extract_table: Table extraction as structured row/column data
- extract_entities: Named-entity recognition over document text
- context_understand: Deep comprehension and evidence grounding
- cross_reference: Find supporting/contradicting passages across sections
- search_documents: Semantic vector search across the document collection
- summarize_section: Generate targeted section summaries
- visualize_data: Generate chart/visualisation specifications

When a tool would help answer the user's question, emit a <tool_call> block. When no tool is needed, answer directly from the document content.

Always be precise, cite specific pages/sections, and indicate confidence level. If information is not found in the documents, say so clearly rather than guessing.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
"""


# ---------------------------------------------------------------------------
# Promotion plan
# ---------------------------------------------------------------------------


def plan_promotion() -> List[Dict[str, Any]]:
    """Return an ordered list of promotion actions.

    Each action is a dict with:
    - ``action``: identifier string
    - ``description``: human-readable explanation
    - ``command``: the shell command or API call to execute

    Returns
    -------
    Ordered list of promotion steps.
    """
    return [
        {
            "action": "backup_v1",
            "description": "Tag the current production model as v1-backup before overwriting",
            "command": "ollama cp docwain:latest docwain:v1-backup",
        },
        {
            "action": "create_v2",
            "description": "Create the V2 model in Ollama from the new GGUF + Modelfile",
            "command": "ollama create docwain:v2 -f Modelfile.v2",
        },
        {
            "action": "regression_test",
            "description": "Run regression test suite against V2 before promoting to latest",
            "command": "python -m src.finetune.v2.merge_promote --regression-only",
        },
        {
            "action": "update_latest",
            "description": "Point docwain:latest to the V2 model after regression passes",
            "command": "ollama cp docwain:v2 docwain:latest",
        },
        {
            "action": "cleanup",
            "description": "Remove intermediate artifacts (merged FP16 weights) to free disk",
            "command": "rm -rf finetune_artifacts/v2/merged_fp16",
        },
    ]


# ---------------------------------------------------------------------------
# Regression criteria
# ---------------------------------------------------------------------------


def get_regression_criteria() -> Dict[str, float]:
    """Return minimum-pass thresholds for regression tests.

    V2 must match or exceed V1 on these core capabilities before promotion.

    Returns
    -------
    Dict mapping metric name to minimum acceptable score (0-100).
    """
    return {
        "persona_match": 90.0,
        "rag_accuracy": 80.0,
        "formatting_quality": 85.0,
        "citation_accuracy": 80.0,
        "response_coherence": 85.0,
    }


def get_new_capability_criteria() -> Dict[str, float]:
    """Return minimum thresholds for V2-specific capabilities.

    These are NEW capabilities that V1 did not have, so they are additive
    checks rather than regressions.

    Returns
    -------
    Dict mapping metric name to minimum acceptable score (0-100).
    """
    return {
        "vision_accuracy": 70.0,
        "table_extraction_f1": 75.0,
        "tool_call_accuracy": 80.0,
        "tool_arg_correctness": 85.0,
        "layout_detection_map": 65.0,
        "insight_precision": 0.80,
        "confidence_calibration_ece": 0.10,
        "synthesis_coherence": 0.80,
        "intent_alignment": 0.85,
        "depth_calibration": 0.75,
    }


# ---------------------------------------------------------------------------
# Merge + promote entrypoint
# ---------------------------------------------------------------------------


def run_phase4(
    config: Optional[Phase4Config] = None,
) -> Path:
    """Execute Phase 4: merge LoRA adapters into the base model (FP16).

    Loads the LoRA adapter from the last training phase (phase3_7) and merges
    it into the base model, saving full FP16 weights for post-training rounds.

    Parameters
    ----------
    config:
        Merge configuration. Uses defaults if ``None``.

    Returns
    -------
    Path to the merged model directory.
    """
    if config is None:
        config = Phase4Config()

    logger.info("=== Phase 4: Merge LoRA into Base Model ===")

    adapter_dir = (config.phase3_dir / "checkpoint_final").resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter checkpoint not found: {adapter_dir}"
        )

    logger.info("Loading LoRA adapter from %s", adapter_dir)
    logger.info("Base model: %s", config.base_model)

    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        dtype=None,
        load_in_4bit=True,
    )

    # Merge LoRA weights into the base model and save as FP16
    merged_dir = config.merged_dir.resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Merging LoRA and saving FP16 weights to %s", merged_dir)
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    logger.info("Phase 4 complete — merged model saved to %s", merged_dir)
    return merged_dir


def _generate_safetensors_modelfile(model_dir: str) -> str:
    """Generate an Ollama Modelfile that imports directly from safetensors.

    Ollama supports ``FROM <path>`` where path is a HuggingFace-format
    model directory containing safetensors, config.json, and tokenizer files.
    """
    return f"""FROM {model_dir}

SYSTEM \"\"\"You are DocWain, an enterprise document intelligence assistant created by MuthuSubramanian.

You have full vision capabilities — you can see and analyse document images, pages, tables, charts, diagrams, and infographics. When a user shares a document image, examine it carefully and provide accurate, grounded answers.

You have access to the following tool_call functions for document analysis:
- ocr_extract: Vision-based text extraction from document pages
- layout_extract: Structural layout detection (headings, paragraphs, tables, figures)
- extract_table: Table extraction as structured row/column data
- extract_entities: Named-entity recognition over document text
- context_understand: Deep comprehension and evidence grounding
- cross_reference: Find supporting/contradicting passages across sections
- search_documents: Semantic vector search across the document collection
- summarize_section: Generate targeted section summaries
- visualize_data: Generate chart/visualisation specifications

When a tool would help answer the user's question, emit a <tool_call> block. When no tool is needed, answer directly from the document content.

Always be precise, cite specific pages/sections, and indicate confidence level. If information is not found in the documents, say so clearly rather than guessing.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
"""


def _find_existing_gguf(config: Phase4Config, model_dir: Path) -> Optional[Path]:
    """Search known locations for an existing GGUF file from a prior conversion.

    Unsloth writes GGUF output to ``{save_dir}_gguf/``, a sibling directory
    of the path passed to ``save_pretrained_gguf``.  We check both the
    configured output dir and the source model dir.
    """
    search_paths = [
        # Unsloth sibling of the output dir: models/docwain-v2_gguf/
        Path(str(config.gguf_output_dir) + "_gguf"),
        # Unsloth sibling of the source model dir
        Path(str(model_dir) + "_gguf"),
        # Inside the output dir itself (manual placement)
        config.gguf_output_dir,
    ]
    for search_dir in search_paths:
        if search_dir.is_dir():
            gguf_files = sorted(search_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
            if gguf_files:
                return gguf_files[0]
    return None


def _ollama_model_exists(model_tag: str) -> bool:
    """Return True if the given model tag exists in Ollama."""
    import subprocess

    result = subprocess.run(
        ["ollama", "show", model_tag, "--modelfile"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def run_final_promote(
    config: Optional[Phase4Config] = None,
    *,
    model_dir: Optional[Path] = None,
) -> Path:
    """Import the final model into Ollama and promote to latest.

    Called after all post-training rounds are complete. Takes the final
    model checkpoint, converts to GGUF (or reuses an existing GGUF),
    creates a new Ollama model, and promotes it to latest — but only
    after safely backing up the existing model.

    Parameters
    ----------
    config:
        Promotion configuration. Uses defaults if ``None``.
    model_dir:
        Path to the final model directory. Defaults to the last post-training
        round output, falling back to the merged model directory.

    Returns
    -------
    Path to the Modelfile used for import.
    """
    import subprocess

    if config is None:
        config = Phase4Config()

    # Determine which model dir to promote — prefer latest post-training round
    if model_dir is None:
        for candidate in [
            Path("finetune_artifacts/v2/post_round3/checkpoint_final"),
            Path("finetune_artifacts/v2/post_round2/checkpoint_final"),
            Path("finetune_artifacts/v2/post_round1/checkpoint_final"),
            config.merged_dir,
        ]:
            candidate = candidate.resolve()
            if candidate.exists() and any(candidate.glob("*.safetensors")):
                model_dir = candidate
                break
        if model_dir is None:
            raise FileNotFoundError(
                "No model directory with safetensors found to promote. "
                "Run training first."
            )

    model_dir = model_dir.resolve()
    logger.info("=== Final Promote: GGUF + Ollama ===")
    logger.info("Source model: %s", model_dir)

    config.gguf_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Convert to GGUF (skip if valid GGUF already exists) ----------
    existing_gguf = _find_existing_gguf(config, model_dir)
    if existing_gguf is not None and existing_gguf.stat().st_size > 1_000_000_000:
        logger.info("Reusing existing GGUF file: %s (%.1f GB)",
                     existing_gguf, existing_gguf.stat().st_size / 1e9)
        gguf_path = str(existing_gguf.resolve())
    else:
        from unsloth import FastLanguageModel  # type: ignore

        logger.info("Loading model for GGUF conversion from %s", model_dir)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_dir.resolve()),
            dtype=None,
            load_in_4bit=True,
        )

        logger.info("Converting to GGUF (%s)...", config.quant_method)
        model.save_pretrained_gguf(
            str(config.gguf_output_dir),
            tokenizer,
            quantization_method=config.quant_method,
        )

        # Free GPU memory
        del model, tokenizer
        import gc, torch  # noqa: E401
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Find the generated GGUF file — Unsloth writes to {save_dir}_gguf/
        converted_gguf = _find_existing_gguf(config, model_dir)
        if converted_gguf is None:
            raise FileNotFoundError(
                "GGUF conversion produced no output files. "
                f"Searched: {config.gguf_output_dir}_gguf/, "
                f"{model_dir}_gguf/, {config.gguf_output_dir}/"
            )
        gguf_path = str(converted_gguf.resolve())

    logger.info("GGUF file: %s", gguf_path)

    # --- Step 2: Generate Modelfile -------------------------------------------
    modelfile_content = generate_v2_modelfile(gguf_path)
    modelfile_path = config.gguf_output_dir / "Modelfile.v2"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    logger.info("Modelfile written to %s", modelfile_path)

    # --- Step 3: Backup existing model before overwriting ---------------------
    model_tag_v2 = f"{config.ollama_model_name}:{config.ollama_tag_v2}"
    model_tag_latest = f"{config.ollama_model_name}:{config.ollama_tag_latest}"
    backup_tag = f"{config.ollama_model_name}:pre-v2-backup"

    if _ollama_model_exists(model_tag_latest):
        logger.info("Backing up %s → %s", model_tag_latest, backup_tag)
        result = subprocess.run(
            ["ollama", "cp", model_tag_latest, backup_tag],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to backup {model_tag_latest} → {backup_tag}: "
                f"{result.stderr.decode().strip()}. "
                "Refusing to overwrite without a successful backup."
            )
        logger.info("Backup complete: %s", backup_tag)
    else:
        logger.info("No existing %s found — skipping backup", model_tag_latest)

    # --- Step 4: Create V2 model from GGUF ------------------------------------
    logger.info("Creating Ollama model %s from %s", model_tag_v2, modelfile_path)
    result = subprocess.run(
        ["ollama", "create", model_tag_v2, "-f", str(modelfile_path)],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama create {model_tag_v2} failed: "
            f"{result.stderr.decode().strip()}"
        )
    logger.info("Successfully created %s", model_tag_v2)

    # --- Step 5: Promote V2 to latest -----------------------------------------
    logger.info("Promoting %s → %s", model_tag_v2, model_tag_latest)
    subprocess.run(
        ["ollama", "cp", model_tag_v2, model_tag_latest],
        check=True,
    )

    logger.info("Final promote complete — %s is now live", model_tag_latest)
    logger.info("Previous model preserved as %s", backup_tag)
    return modelfile_path
