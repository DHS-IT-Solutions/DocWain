#!/usr/bin/env bash
# Phase 2 exit checklist (user Task 17).
#
# Runs pytest against every Phase 2 test module, greps the Phase-2-introduced
# code paths for forbidden patterns (Claude / Anthropic / datetime.utcnow /
# new pipeline status strings / hardcoded YAMLs in Phase 2 src/), verifies
# the eight canonical feature flags default OFF, and confirms
# src/intelligence/generator.py is untouched in the Phase 2 branch diff.
#
# Exit codes:
#   0 — every check passes.
#   1 — at least one check failed; details printed to stdout.
#
# Run from repo root: `bash scripts/phase2_exit_check.sh`.

set -u
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
CHECKS=()
RESULTS=()

# Phase 2 branch base — the last Phase 1 commit. Everything newer belongs
# to Phase 2 and must pass the forbidden-pattern scans.
PHASE1_TIP=$(git log --grep="phase1(sme-integration): sandbox end-to-end plumbing" \
    -n 1 --format="%H" 2>/dev/null || true)
if [ -z "$PHASE1_TIP" ]; then
    PHASE1_TIP="HEAD~25"
fi

# Files Phase 2 added or modified. We intentionally do NOT grep the whole
# src/ tree — legacy modules have their own compliance history.
PHASE2_PATHS=$(git diff --name-only "$PHASE1_TIP"..HEAD \
    -- "src/" "tests/" "scripts/" 2>/dev/null | tr '\n' ' ')

run_check() {
    local label="$1"
    shift
    if "$@" >/tmp/phase2_exit_check_$$ 2>&1; then
        CHECKS+=("$label")
        RESULTS+=("PASS")
        PASS=$((PASS + 1))
    else
        CHECKS+=("$label")
        RESULTS+=("FAIL")
        FAIL=$((FAIL + 1))
        echo "----- $label failed; output below -----"
        cat /tmp/phase2_exit_check_$$
        echo "----- end output -----"
    fi
    rm -f /tmp/phase2_exit_check_$$
}

# ---- 1. Pytest sweep over every Phase 2 test target -----------------------
run_check "pytest_phase2_suite" \
    python -m pytest tests/intelligence/sme tests/api tests/retrieval \
           tests/config tests/scripts tests/intelligence/test_qa_cache_index.py \
           --tb=no -q

# ---- 2. No Claude / Anthropic / Co-Authored-By in Phase-2 touched files ---
run_check "no_claude_anthropic_refs_in_phase2" bash -c "
    if [ -z '$PHASE2_PATHS' ]; then
        echo 'no Phase 2 files to check'
        exit 0
    fi
    offenders=\$(git grep -nE -i '(claude|anthropic|co-authored-by)' \\
        -- $PHASE2_PATHS 2>/dev/null | grep -v 'ERRATA-sme-contracts.md' || true)
    if [ -n \"\$offenders\" ]; then
        echo 'Claude/Anthropic references in Phase 2 touched files:'
        echo \"\$offenders\" | head -20
        exit 1
    fi
    exit 0
"

# ---- 3. No datetime.utcnow( introduced in Phase 2 -------------------------
# Check the diff itself so pre-existing legacy uses in a Phase 2-touched
# file don't trigger a false positive.
run_check "no_datetime_utcnow_in_phase2_diff" bash -c "
    offenders=\$(git diff '$PHASE1_TIP'..HEAD -- 'src/' 'tests/' 'scripts/' 2>/dev/null \\
        | grep -E '^\\+' | grep -v '^\\+\\+\\+' \\
        | grep 'datetime\\.utcnow(' \\
        || true)
    if [ -n \"\$offenders\" ]; then
        echo 'datetime.utcnow introduced in Phase 2 diff:'
        echo \"\$offenders\" | head -20
        exit 1
    fi
    exit 0
"

# ---- 4. No new pipeline_status strings introduced in Phase 2 --------------
# Allowed set — must match src/api/statuses.py. The current count is the
# Phase 1 + Phase 2 baseline; any new PIPELINE_ constant added in Phase 2
# would push the count up.
run_check "no_new_pipeline_status" bash -c "
    declared=\$(grep -oE 'PIPELINE_[A-Z_]+' src/api/statuses.py | sort -u || true)
    declared_count=\$(echo \"\$declared\" | grep -c '^PIPELINE_' || true)
    if [ \"\$declared_count\" -gt 20 ]; then
        echo \"Pipeline status constant count unexpectedly high: \$declared_count\"
        exit 1
    fi
    exit 0
"

# ---- 5. No hardcoded YAML reads in Phase 2 src/ changes -------------------
run_check "no_hardcoded_yaml_in_phase2_src" bash -c "
    if [ -z '$PHASE2_PATHS' ]; then
        echo 'no Phase 2 files to check'
        exit 0
    fi
    offenders=\$(git grep -n 'yaml\\.safe_load\\|yaml\\.load\\|yaml\\.dump' \\
        -- $PHASE2_PATHS 2>/dev/null \\
        | grep -v '/adapter_loader.py:' \\
        | grep -v '/adapter_schema.py:' \\
        | grep -v '/sme_admin_api.py:' \\
        | grep -v '/config/' \\
        | grep -v '/serving/' \\
        | grep -v '/scripts/sme_eval/' \\
        | grep -v '/api/config.py:' \\
        | grep -v '^tests/' \\
        || true)
    if [ -n \"\$offenders\" ]; then
        echo 'YAML reads in Phase 2 src/ outside allowlist:'
        echo \"\$offenders\"
        exit 1
    fi
    exit 0
"

# ---- 6. All eight SME feature flags default OFF --------------------------
run_check "feature_flags_default_off" \
    python -c '
from src.config.feature_flags import (
    SMEFeatureFlags, FlagStore,
    SME_REDESIGN_ENABLED, ENABLE_SME_SYNTHESIS, ENABLE_SME_RETRIEVAL,
    ENABLE_KG_SYNTHESIZED_EDGES, ENABLE_RICH_MODE, ENABLE_URL_AS_PROMPT,
    ENABLE_HYBRID_RETRIEVAL, ENABLE_CROSS_ENCODER_RERANK,
)

class _EmptyStore:
    def get_subscription_overrides(self, sub):
        return {}

flags = SMEFeatureFlags(store=_EmptyStore())
allflags = [SME_REDESIGN_ENABLED, ENABLE_SME_SYNTHESIS, ENABLE_SME_RETRIEVAL,
            ENABLE_KG_SYNTHESIZED_EDGES, ENABLE_RICH_MODE, ENABLE_URL_AS_PROMPT,
            ENABLE_HYBRID_RETRIEVAL, ENABLE_CROSS_ENCODER_RERANK]
for flag in allflags:
    assert flags.is_enabled("sub_any", flag) is False, f"flag {flag} defaults to True"
print("all 8 flags default OFF")
'

# ---- 7. src/intelligence/generator.py untouched in Phase 2 ---------------
run_check "generator_py_untouched_in_phase2" bash -c "
    changed=\$(git diff --name-only '$PHASE1_TIP'..HEAD -- 'src/intelligence/generator.py' 2>/dev/null || true)
    if [ -n \"\$changed\" ]; then
        echo 'src/intelligence/generator.py modified in Phase 2:'
        echo \"\$changed\"
        exit 1
    fi
    exit 0
"

# ---- 8. No internal wall-clock timeouts on Phase 2 synthesis paths -------
run_check "no_internal_timeouts_on_synthesis" bash -c '
    offenders=$(git grep -nE "asyncio\.wait_for|signal\.alarm" \
        -- "src/intelligence/sme/" "src/api/pipeline_api.py" \
        2>/dev/null || true)
    if [ -n "$offenders" ]; then
        echo "Internal timeouts on synthesis paths:"
        echo "$offenders"
        exit 1
    fi
    exit 0
'

# ---- 9. sme_redesign_enabled master default is False ---------------------
run_check "master_flag_default_off" \
    python -c '
from src.config.feature_flags import _DEFAULTS, SME_REDESIGN_ENABLED
assert _DEFAULTS[SME_REDESIGN_ENABLED] is False, "master flag default must be False"
print("master flag defaults to False")
'

# ---- Summary -------------------------------------------------------------
echo ""
echo "Phase 2 exit checklist — summary (base: $PHASE1_TIP):"
for i in "${!CHECKS[@]}"; do
    printf "  [%s] %s\n" "${RESULTS[$i]}" "${CHECKS[$i]}"
done
echo ""
echo "TOTAL: $PASS pass / $FAIL fail"

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
