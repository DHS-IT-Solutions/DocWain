#!/usr/bin/env bash
#
# Monthly SME pattern-mining wrapper invoked by systemd.timer.
# Produces the monthly Markdown/JSON bundle and training-candidate evidence.
# No retraining is triggered — sub-project F remains a separate human-gated
# project.

set -euo pipefail

REPO_ROOT="${DOCWAIN_REPO_ROOT:-/home/ubuntu/PycharmProjects/DocWain}"
PY="${REPO_ROOT}/.venv/bin/python"
ANALYTICS_DIR="${REPO_ROOT}/analytics"
MONTH="$(date -u +%Y-%m)"

cd "${REPO_ROOT}"

echo "[sme-pattern-mining] start ${MONTH}"

"${PY}" -m scripts.mine_sme_patterns --analytics-dir "${ANALYTICS_DIR}"

"${PY}" "${REPO_ROOT}/scripts/evaluate_training_trigger.py" \
    --reports-dir "${ANALYTICS_DIR}" \
    --out "${ANALYTICS_DIR}/training_candidates_${MONTH}.json"

echo "[sme-pattern-mining] done"
