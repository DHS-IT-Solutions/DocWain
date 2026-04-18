#!/bin/bash
# Kick off the real V5-14B SFT run.
#
# Seed merge FAILED (see finetune_artifacts/v5/seed_eval_report.json —
# mean LLM-judge 2.81 vs 4.71 baseline, vision-graft incompatibility),
# so SFT starts from V3 weights directly per the documented fallback.
#
# Preconditions:
#   - Data gen has produced sft_generated.jsonl at an acceptable size
#     (aim: 8-12K rows minimum across the 8 behaviour capabilities)
#   - vLLM V3 is currently serving — will be stopped below
#   - ~24-30 h A100 time remaining
#
# Usage: bash scripts/v5_kickoff_sft.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SFT_CORPUS="finetune_artifacts/v5/sft_reused.jsonl,finetune_artifacts/v5/sft_generated.jsonl"
BASE="models/DocWain-14B-v2"  # V3 weights — seed merge failed, fallback active
OUTPUT="models/DocWain-14B-v5-sft"
LOG="finetune_artifacts/v5/sft_training.log"

echo "=== V5 SFT kickoff ==="
echo "  base:    $BASE"
echo "  corpus:  $SFT_CORPUS"
echo "  output:  $OUTPUT"
echo

# Row counts sanity
for f in finetune_artifacts/v5/sft_reused.jsonl finetune_artifacts/v5/sft_generated.jsonl; do
    if [[ -f "$f" ]]; then
        n=$(wc -l < "$f")
        echo "  $f : $n rows"
    else
        echo "  WARNING: $f missing"
    fi
done
echo

echo "=== Stopping vLLM to free GPU ==="
sudo systemctl stop docwain-vllm-fast docwain-gpu-scheduler || true
sleep 10
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo
echo "=== Launching SFT ==="
nohup python -u -m src.finetune.v5.sft_trainer \
    --base "$BASE" \
    --corpus "$SFT_CORPUS" \
    --output "$OUTPUT" \
    --lora-rank 128 --lora-alpha 32 \
    --epochs 2 \
    --batch-size 1 --grad-accum 16 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.03 \
    --checkpoint-interval 6h \
    > /tmp/v5_sft.log 2>&1 &

echo "  SFT pid: $!"
echo "  Log: /tmp/v5_sft.log"
echo "  Checkpoint dir: ${OUTPUT}/checkpoints/"
echo
echo "Monitor with:"
echo "  tail -F /tmp/v5_sft.log | grep -E 'step=|loss=|checkpoint|epoch|Error|Traceback'"
