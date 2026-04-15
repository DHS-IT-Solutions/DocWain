#!/usr/bin/env python3
"""
DocWain Model Intelligence Sprint — Entry Point

Usage:
    python scripts/run_sprint.py                    # Run full sprint
    python scripts/run_sprint.py --resume           # Resume from last checkpoint
    python scripts/run_sprint.py --phase phase1_sft # Jump to specific phase
    python scripts/run_sprint.py --dry-run          # Generate data only, no training
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.finetune.sprint.config import SprintConfig, SprintState
from src.finetune.sprint.orchestrator import SprintOrchestrator


def main():
    parser = argparse.ArgumentParser(description="DocWain Model Intelligence Sprint")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved phase")
    parser.add_argument("--phase", type=str, help="Jump to specific phase")
    parser.add_argument("--dry-run", action="store_true", help="Generate data only, skip training")
    parser.add_argument("--artifacts-dir", type=str, default="finetune_artifacts/sprint",
                        help="Directory for sprint artifacts")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.artifacts_dir}/sprint.log"),
        ],
    )

    config = SprintConfig(artifacts_dir=args.artifacts_dir)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)

    if args.phase:
        state = SprintState.load(Path(args.artifacts_dir))
        state.phase = args.phase
        state.save()

    orchestrator = SprintOrchestrator(config)
    logging.info(f"Sprint config: {config.model_name}, LoRA r={config.lora_r}")
    logging.info(f"Targets: SFT={config.phase1_sft_target + config.phase2_sft_target}, "
                 f"DPO={config.phase1_dpo_target + config.phase2_dpo_target}")

    orchestrator.run()


if __name__ == "__main__":
    main()
