"""Full V2+ pipeline runner — orchestrates data generation through final promotion.

Usage::

    python -m src.finetune.v2.run_v2plus --start-from data_gen --scale 1.0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

PIPELINE_PHASES = [
    "data_gen",
    "phase1",
    "phase2",
    "phase2_5",
    "phase3",
    "phase3_5",
    "phase3_7",
    "phase4",
    "round1",
    "round2",
    "round3",
    "final_promote",
]


def run_full_pipeline(
    start_from: str = "data_gen",
    scale: float = 1.0,
    skip_data_gen: bool = False,
    data_dir: Path = Path("finetune_data/v2"),
    artifacts_dir: Path = Path("finetune_artifacts/v2"),
) -> Dict[str, object]:
    """Execute the full V2+ pipeline sequentially.

    Parameters
    ----------
    start_from:
        Phase to start from (skips earlier phases). Must be in PIPELINE_PHASES.
    scale:
        Dataset size multiplier passed to data generation.
    skip_data_gen:
        If True, skip data generation even when start_from is "data_gen".
    data_dir:
        Directory containing / to write training data.
    artifacts_dir:
        Directory for training artifacts and checkpoints.

    Returns
    -------
    Dict with per-phase timing and status.
    """
    if start_from not in PIPELINE_PHASES:
        raise ValueError(
            f"Unknown phase '{start_from}'. Must be one of: {PIPELINE_PHASES}"
        )

    start_idx = PIPELINE_PHASES.index(start_from)
    phases_to_run = PIPELINE_PHASES[start_idx:]
    results: Dict[str, object] = {}
    t_total = time.time()

    for phase in phases_to_run:
        if phase == "data_gen" and skip_data_gen:
            logger.info("Skipping data generation (--skip-data-gen)")
            results["data_gen"] = {"status": "skipped"}
            continue

        logger.info("=" * 60)
        logger.info("Starting phase: %s", phase)
        logger.info("=" * 60)
        t0 = time.time()

        try:
            _run_phase(phase, scale=scale, data_dir=data_dir, artifacts_dir=artifacts_dir)
            elapsed = round(time.time() - t0, 1)
            results[phase] = {"status": "complete", "time_s": elapsed}
            logger.info("Phase %s complete in %.1fs", phase, elapsed)
        except Exception:
            elapsed = round(time.time() - t0, 1)
            results[phase] = {"status": "failed", "time_s": elapsed}
            logger.exception("Phase %s FAILED after %.1fs", phase, elapsed)
            break

    results["total_time_s"] = round(time.time() - t_total, 1)
    logger.info("Pipeline finished. Results: %s", results)
    return results


def _run_phase(
    phase: str,
    *,
    scale: float,
    data_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Dispatch a single phase to its implementation."""
    if phase == "data_gen":
        from .generate_all_data import generate_all

        generate_all(data_dir, scale)

    elif phase == "phase1":
        from .phase1_vision_graft import run_phase1

        run_phase1()

    elif phase == "phase2":
        from .phase2_doc_sft import run_phase2

        run_phase2()

    elif phase == "phase2_5":
        from .phase2_5_dpo import run_phase2_5

        run_phase2_5()

    elif phase == "phase3":
        from .phase3_tool_sft import run_phase3

        run_phase3()

    elif phase == "phase3_5":
        from .phase3_5_insight_sft import run_phase3_5

        run_phase3_5()

    elif phase == "phase3_7":
        from .phase3_7_holistic import run_phase3_7

        run_phase3_7()

    elif phase == "phase4":
        from .merge_promote import run_phase4

        run_phase4()

    elif phase == "round1":
        from .post_round1 import run_round1

        run_round1()

    elif phase == "round2":
        from .post_round2 import run_round2

        run_round2()

    elif phase == "round3":
        from .post_round3 import run_round3

        run_round3()

    elif phase == "final_promote":
        from .merge_promote import run_phase4, Phase4Config

        config = Phase4Config()
        run_phase4(config=config, skip_regression=False)

    else:
        raise ValueError(f"Unknown phase: {phase}")


def main() -> None:
    """CLI entry point for the full V2+ pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full DocWain V2+ fine-tuning pipeline"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="data_gen",
        choices=PIPELINE_PHASES,
        help="Phase to start from (default: data_gen)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Dataset size multiplier (0.1 for dev, 1.0 for production)",
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip data generation even if starting from data_gen",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("finetune_data/v2"),
        help="Training data directory (default: finetune_data/v2)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("finetune_artifacts/v2"),
        help="Artifacts directory (default: finetune_artifacts/v2)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_full_pipeline(
        start_from=args.start_from,
        scale=args.scale,
        skip_data_gen=args.skip_data_gen,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
