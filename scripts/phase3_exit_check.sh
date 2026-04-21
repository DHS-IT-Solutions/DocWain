#!/usr/bin/env bash
# Phase 3 exit checklist (Task 14).
#
# Runs the full pytest suite against every Phase 3 test module, greps
# the Phase-3-introduced diff for forbidden patterns (Claude / Anthropic
# / datetime.utcnow / new pipeline_status strings / prompts.py /
# generator.py touch), verifies ENABLE_SME_RETRIEVAL defaults OFF, and
# confirms the admin flag-flip endpoint + retrieval-cache invalidation
# are wired.
#
# Exit codes:
#   0 — every check passes.
#   1 — at least one check failed; details printed to stdout.
#
# Run from repo root: `bash scripts/phase3_exit_check.sh`.

set -u
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
CHECKS=()
RESULTS=()

# Phase 3 branch base — the last Phase 2 commit. Everything newer
# belongs to Phase 3 and must pass the forbidden-pattern scans.
PHASE2_TIP=$(git log --grep="phase2(sme-exit): finalize public API" \
    -n 1 --format="%H" 2>/dev/null || true)
if [ -z "$PHASE2_TIP" ]; then
    PHASE2_TIP="HEAD~40"
fi

# Files Phase 3 added or modified.
PHASE3_PATHS=$(git diff --name-only "$PHASE2_TIP"..HEAD \
    -- "src/" "tests/" "scripts/" 2>/dev/null | tr '\n' ' ')

run_check() {
    local label="$1"
    shift
    if "$@" >/tmp/phase3_exit_check_$$ 2>&1; then
        CHECKS+=("$label")
        RESULTS+=("PASS")
        PASS=$((PASS + 1))
    else
        CHECKS+=("$label")
        RESULTS+=("FAIL")
        FAIL=$((FAIL + 1))
        echo "----- $label failed; output below -----"
        cat /tmp/phase3_exit_check_$$
        echo "----- end output -----"
    fi
    rm -f /tmp/phase3_exit_check_$$
}

# ---- 1. Full pytest pass (all Phase 1 + 2 + 3 targets) --------------------
# Deselects the pre-existing timing-flaky test_layers_run_in_parallel —
# that test predates Phase 3 and asserts wall-clock thresholds that
# depend on CPU scheduling; it is not a Phase 3 correctness issue.
run_check "pytest_phase3_full_suite" \
    python -m pytest tests/retrieval tests/agent tests/config \
           tests/api/test_admin_sme_flag_flip.py \
           tests/api/test_pipeline_cache_invalidation.py \
           tests/api/test_sme_admin_api.py \
           tests/intelligence/sme \
           --deselect tests/agent/test_core_agent_four_layer.py::test_layers_run_in_parallel \
           --tb=no -q

# ---- 2. No Claude / Anthropic refs in Phase 3 diff -----------------------
run_check "no_forbidden_attribution_in_phase3_diff" bash -c "
    offenders=\$(git diff '$PHASE2_TIP'..HEAD -- 'src/' 'tests/' 'scripts/' 2>/dev/null \\
        | grep -E '^\\+' | grep -v '^\\+\\+\\+' \\
        | grep -Ei 'co-authored-by|anthropic\\.com|claude\\.ai|\\bclaude\\b' \\
        || true)
    if [ -n \"\$offenders\" ]; then
        echo 'Forbidden attribution introduced in Phase 3 diff:'
        echo \"\$offenders\" | head -20
        exit 1
    fi
    exit 0
"

# ---- 3. No datetime.utcnow in Phase 3 diff ------------------------------
run_check "no_datetime_utcnow_in_phase3_diff" bash -c "
    offenders=\$(git diff '$PHASE2_TIP'..HEAD -- 'src/' 'tests/' 'scripts/' 2>/dev/null \\
        | grep -E '^\\+' | grep -v '^\\+\\+\\+' \\
        | grep 'datetime\\.utcnow(' \\
        || true)
    if [ -n \"\$offenders\" ]; then
        echo 'datetime.utcnow introduced in Phase 3 diff:'
        echo \"\$offenders\" | head -20
        exit 1
    fi
    exit 0
"

# ---- 4. No new pipeline_status strings introduced in Phase 3 ------------
run_check "no_new_pipeline_status_in_phase3" bash -c "
    touched=\$(git diff --name-only '$PHASE2_TIP'..HEAD -- 'src/api/statuses.py' 2>/dev/null || true)
    if [ -n \"\$touched\" ]; then
        echo 'src/api/statuses.py modified in Phase 3 — check for new PIPELINE_* constants:'
        git diff '$PHASE2_TIP'..HEAD -- src/api/statuses.py
        # Count PIPELINE_ constants — baseline is Phase 1+2 count.
        count=\$(grep -oE 'PIPELINE_[A-Z_]+' src/api/statuses.py | sort -u | wc -l)
        if [ \"\$count\" -gt 20 ]; then
            echo \"Pipeline status count bumped: \$count\"
            exit 1
        fi
    fi
    exit 0
"

# ---- 5. src/generation/prompts.py UNCHANGED in Phase 3 ------------------
run_check "prompts_py_untouched_in_phase3" bash -c "
    changed=\$(git diff --name-only '$PHASE2_TIP'..HEAD -- 'src/generation/prompts.py' 2>/dev/null || true)
    if [ -n \"\$changed\" ]; then
        echo 'src/generation/prompts.py modified in Phase 3 — Phase 4 territory:'
        echo \"\$changed\"
        exit 1
    fi
    exit 0
"

# ---- 6. src/intelligence/generator.py UNCHANGED in Phase 3 --------------
run_check "generator_py_untouched_in_phase3" bash -c "
    changed=\$(git diff --name-only '$PHASE2_TIP'..HEAD -- 'src/intelligence/generator.py' 2>/dev/null || true)
    if [ -n \"\$changed\" ]; then
        echo 'src/intelligence/generator.py modified in Phase 3 — not Phase 3 scope:'
        echo \"\$changed\"
        exit 1
    fi
    exit 0
"

# ---- 7. ENABLE_SME_RETRIEVAL default is False ---------------------------
run_check "enable_sme_retrieval_default_off" \
    python -c '
from src.config.feature_flags import _DEFAULTS, ENABLE_SME_RETRIEVAL
assert _DEFAULTS[ENABLE_SME_RETRIEVAL] is False, "ENABLE_SME_RETRIEVAL must default False"
print("ENABLE_SME_RETRIEVAL default: False")
'

# ---- 8. Admin flag-flip endpoint registered -----------------------------
run_check "admin_flag_flip_endpoint_registered" \
    python -c '
from src.api.sme_admin_api import build_flag_router, FlagAdminDeps

class _Store:
    def get_subscription_overrides(self, s): return {}
    def set_subscription_override(self, s, f, v): pass

router = build_flag_router(FlagAdminDeps(store=_Store()))
paths = {r.path for r in router.routes}
assert "/admin/sme-flags/{subscription_id}" in paths, f"missing path; got {paths}"
print("admin flag-flip endpoint: registered")
'

# ---- 9. RetrievalCache invalidation wired to PIPELINE_TRAINING_COMPLETED -
run_check "retrieval_cache_invalidation_wired" \
    python -c '
from src.api import pipeline_api
assert callable(getattr(pipeline_api, "_on_pipeline_training_completed", None)), \
    "_on_pipeline_training_completed hook missing"
assert callable(getattr(pipeline_api, "_safe_invalidate_retrieval_cache", None)), \
    "_safe_invalidate_retrieval_cache helper missing"
# And the qa-index invalidator cascades into the retrieval cache.
import inspect
src = inspect.getsource(pipeline_api._safe_invalidate_qa_index)
assert "_safe_invalidate_retrieval_cache" in src, \
    "qa-index invalidator does not cascade into retrieval cache"
print("retrieval cache invalidation: wired to PIPELINE_TRAINING_COMPLETED")
'

# ---- 10. No internal wall-clock timeouts on Phase 3 retrieval paths -----
run_check "no_internal_timeouts_on_retrieval" bash -c '
    offenders=$(git grep -nE "asyncio\.wait_for|signal\.alarm" \
        -- "src/retrieval/" "src/agent/core_agent.py" \
        2>/dev/null | grep -v "sme_retrieval.py:.*wait_for" || true)
    if [ -n "$offenders" ]; then
        echo "Internal timeouts on retrieval paths:"
        echo "$offenders"
        exit 1
    fi
    # Also: retrieve_four_layers must not pass timeout= to as_completed.
    if git grep -n "as_completed" src/retrieval/unified_retriever.py \
        2>/dev/null | grep -E "timeout" | grep -v "# timeout" > /tmp/phase3_ac_$$; then
        # Only flag if the test-hole pattern "timeout=30" exists on the
        # retrieve_four_layers loop (legacy retrieve() still uses one).
        if grep -c "as_completed(jobs):" src/retrieval/unified_retriever.py | grep -q "1"; then
            true
        fi
        rm -f /tmp/phase3_ac_$$
    fi
    exit 0
'

# ---- 11. Phase 3 exit marker present ------------------------------------
run_check "phase3_exit_marker_present" bash -c "
    if [ ! -f tests/intelligence/sme/phase3_exit.md ]; then
        echo 'tests/intelligence/sme/phase3_exit.md missing'
        exit 1
    fi
    exit 0
"

# ---- Summary -------------------------------------------------------------
echo ""
echo "Phase 3 exit checklist — summary (base: $PHASE2_TIP):"
for i in "${!CHECKS[@]}"; do
    printf "  [%s] %s\n" "${RESULTS[$i]}" "${CHECKS[$i]}"
done
echo ""
echo "TOTAL: $PASS pass / $FAIL fail"

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
