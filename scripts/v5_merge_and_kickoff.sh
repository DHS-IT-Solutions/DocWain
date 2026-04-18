#!/bin/bash
# Wait for both parallel data_gen processes to finish, merge outputs, run
# a quick per-capability sanity check, and kick off the 14B SFT run.
#
# Assumes two parallel data_gen processes (A + B) are active with pids
# stored in V5_PID_A / V5_PID_B env vars OR passed as args.
#
# After merge, the SFT corpus will be
#   finetune_artifacts/v5/sft_reused.jsonl            (31,011 rows, V4 port)
#   finetune_artifacts/v5/sft_generated.jsonl         (schema + refusal + A-output)
#   finetune_artifacts/v5/sft_generated_partB.jsonl   (B-output)
# → all three concatenated form the SFT input.
#
# Usage:
#   bash scripts/v5_merge_and_kickoff.sh <pid_a> <pid_b>
#
# The script exits non-zero if either process died with no output, so CI
# or a watchdog can detect the failure.
set -euo pipefail

PID_A="${1:-${V5_PID_A:-}}"
PID_B="${2:-${V5_PID_B:-}}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "$PID_A" || -z "$PID_B" ]]; then
    echo "ERROR: pid_a and pid_b required" >&2
    exit 2
fi

echo "=== Waiting for data gen A (pid $PID_A) + B (pid $PID_B) ==="
while kill -0 "$PID_A" 2>/dev/null || kill -0 "$PID_B" 2>/dev/null; do
    a_alive=$(kill -0 "$PID_A" 2>/dev/null && echo yes || echo no)
    b_alive=$(kill -0 "$PID_B" 2>/dev/null && echo yes || echo no)
    a_rows=$(wc -l < finetune_artifacts/v5/sft_generated.jsonl 2>/dev/null || echo 0)
    b_rows=$(wc -l < finetune_artifacts/v5/sft_generated_partB.jsonl 2>/dev/null || echo 0)
    echo "  $(date +%H:%M:%S)  A=$a_alive ($a_rows rows)  B=$b_alive ($b_rows rows)"
    sleep 60
done
echo

echo "=== Both processes complete. Merging ==="
SFT_FINAL="finetune_artifacts/v5/sft_combined.jsonl"
DPO_FINAL="finetune_artifacts/v5/dpo_combined.jsonl"

cat finetune_artifacts/v5/sft_reused.jsonl \
    finetune_artifacts/v5/sft_generated.jsonl \
    finetune_artifacts/v5/sft_generated_partB.jsonl \
    > "$SFT_FINAL"

cat finetune_artifacts/v5/dpo_reused.jsonl \
    finetune_artifacts/v5/dpo_generated.jsonl \
    finetune_artifacts/v5/dpo_generated_partB.jsonl \
    2>/dev/null > "$DPO_FINAL" || true

echo "Corpus assembled:"
wc -l "$SFT_FINAL" "$DPO_FINAL"
echo

echo "=== Per-capability SFT breakdown ==="
python - <<'PY'
import json
from collections import Counter
c = Counter()
with open("finetune_artifacts/v5/sft_combined.jsonl") as f:
    for line in f:
        try: c[json.loads(line).get('capability', '?')] += 1
        except: pass
for k, v in c.most_common():
    print(f"  {k:28s} {v:>6}")
print(f"  {'TOTAL':28s} {sum(c.values()):>6}")
PY

echo
echo "=== Launching SFT ==="
exec bash scripts/v5_kickoff_sft.sh
