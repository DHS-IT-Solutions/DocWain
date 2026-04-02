"""Master data generation script — generates ALL training and eval data for V2+.

Usage::

    python -m src.finetune.v2.generate_all_data --output-dir finetune_data/v2 --scale 1.0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def generate_all(output_dir: Path, scale: float = 1.0) -> Dict[str, object]:
    """Generate all training and eval data for V2+.

    Calls each data generator in pipeline order and collects statistics.

    Parameters
    ----------
    output_dir:
        Root directory for generated data files.
    scale:
        Multiplier for dataset sizes (0.1 for dev, 1.0 for production).

    Returns
    -------
    Dict with per-phase generation stats and timing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, object] = {}
    t_total = time.time()

    # Phase 2: Document intelligence SFT data (20K at scale=1.0)
    logger.info("Generating Phase 2 — document intelligence data...")
    t0 = time.time()
    from .data_generator.phase2_doc_intelligence import generate_phase2_data

    phase2_stats = generate_phase2_data(output_dir, scale)
    stats["phase2"] = {"counts": phase2_stats, "time_s": round(time.time() - t0, 1)}
    logger.info("Phase 2 done: %s", phase2_stats)

    # Phase 2.5: DPO preference pairs (5K at scale=1.0)
    logger.info("Generating Phase 2.5 — DPO pairs...")
    t0 = time.time()
    from .data_generator.phase2_5_dpo_pairs import generate_phase25_data

    count_25 = generate_phase25_data(output_dir, scale)
    stats["phase2_5"] = {"count": count_25, "time_s": round(time.time() - t0, 1)}
    logger.info("Phase 2.5 done: %d pairs", count_25)

    # Phase 3.5: Insight generation data (6K at scale=1.0)
    logger.info("Generating Phase 3.5 — insights data...")
    t0 = time.time()
    from .data_generator.phase3_5_insights import generate_phase35_data

    count_35 = generate_phase35_data(output_dir, scale)
    stats["phase3_5"] = {"count": count_35, "time_s": round(time.time() - t0, 1)}
    logger.info("Phase 3.5 done: %d examples", count_35)

    # Phase 3.7: Holistic reasoning data (8K at scale=1.0)
    logger.info("Generating Phase 3.7 — holistic reasoning data...")
    t0 = time.time()
    from .data_generator.phase3_7_holistic import generate_phase37_data

    count_37 = generate_phase37_data(output_dir, scale)
    stats["phase3_7"] = {"count": count_37, "time_s": round(time.time() - t0, 1)}
    logger.info("Phase 3.7 done: %d examples", count_37)

    # Post Round 1: Conversational DPO (3K at scale=1.0)
    logger.info("Generating Post Round 1 — conversational DPO data...")
    t0 = time.time()
    from .data_generator.post_conversational import generate_post_conversational_data

    count_conv = generate_post_conversational_data(output_dir, scale)
    stats["post_conversational"] = {"count": count_conv, "time_s": round(time.time() - t0, 1)}
    logger.info("Post conversational done: %d examples", count_conv)

    # Post Round 2: Confidence calibration (2K at scale=1.0)
    logger.info("Generating Post Round 2 — confidence calibration data...")
    t0 = time.time()
    from .data_generator.post_confidence import generate_post_confidence_data

    count_conf = generate_post_confidence_data(output_dir, scale)
    stats["post_confidence"] = {"count": count_conf, "time_s": round(time.time() - t0, 1)}
    logger.info("Post confidence done: %d examples", count_conf)

    # Eval suite (500 examples)
    logger.info("Generating eval suite...")
    t0 = time.time()
    from .data_generator.eval_suite import write_eval_suite

    write_eval_suite(output_dir)
    stats["eval_suite"] = {"time_s": round(time.time() - t0, 1)}
    logger.info("Eval suite done")

    stats["total_time_s"] = round(time.time() - t_total, 1)
    logger.info("All data generation complete in %.1fs", stats["total_time_s"])
    return stats


def main() -> None:
    """CLI entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate all V2+ training and eval data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("finetune_data/v2"),
        help="Root output directory (default: finetune_data/v2)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Dataset size multiplier (0.1 for dev, 1.0 for production)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = generate_all(args.output_dir, args.scale)
    logger.info("Generation stats: %s", stats)


if __name__ == "__main__":
    main()
