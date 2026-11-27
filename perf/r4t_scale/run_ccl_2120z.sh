#!/usr/bin/env bash
# Detached runner for the booked 2026-11-27 21:20-21:35Z anx-glx whole-box CCL window:
#   1. 1x4 (4 chips), 2x4 (8 chips), 4x4 (16 chips) exact fp32 2D Ring CCL (1 clean process/width)
#   2. 8x4 (32 chips) DeviceCCLReducer dict-bench on all 558 real RFD4 parameter shapes (178.4M fp32)
#   3. Clean glx_reset + DONE marker before 21:35Z.
set -uo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

REPO="$HOME/fleet/r4t/ccl/repo"
OUT="$HOME/fleet/r4t/ccl/out_2120z"
PY="$HOME/fleet/env/bin/python"
SCRIPT="$REPO/perf/r4t_scale/ccl_allreduce_2d.py"
GO_EARLY="$HOME/fleet/r4t/ccl/GO_EARLY"
mkdir -p "$OUT"
LOG="$OUT/run.log"
exec >>"$LOG" 2>&1

START_EPOCH=$(date -u -d '2026-11-27 21:20:00 UTC' +%s)
GIVEUP_EPOCH=$(date -u -d '2026-11-27 21:30:00 UTC' +%s)
DEADLINE_EPOCH=$(date -u -d '2026-11-27 21:33:00 UTC' +%s)
SHA=$(git -C "$REPO" rev-parse --short=9 HEAD 2>/dev/null || echo "unknown")

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "=== [$(ts)] run_ccl_2120z.sh started pid=$$ sha=$SHA ==="

# 1. Wait until 21:20:00Z (or GO_EARLY flag after rlx clears) AND fuser=0.
while true; do
    if [ -f "$HOME/fleet/HANDBACK" ]; then
        echo "[$(ts)] HANDBACK present, aborting without opening devices"
        echo "HANDBACK" >"$OUT/DONE"
        exit 0
    fi
    NOW=$(date -u +%s)
    if [ "$NOW" -ge "$GIVEUP_EPOCH" ]; then
        echo "[$(ts)] GIVEUP reached without quiet box, exiting cleanly"
        echo "GIVEUP" >"$OUT/DONE"
        exit 0
    fi
    if [ "$NOW" -lt "$START_EPOCH" ] && [ ! -f "$GO_EARLY" ]; then
        sleep 5
        continue
    fi
    if fuser /dev/tenstorrent/* >/dev/null 2>&1; then
        sleep 5
        continue
    fi
    # Confirm 2 consecutive quiet checks (5s apart) so we never race a finishing process.
    sleep 5
    if ! fuser /dev/tenstorrent/* >/dev/null 2>&1; then
        break
    fi
done

echo "[$(ts)] BOX QUIET (uptime: $(uptime)); starting 1-process-per-width CCL suite"

RAN_ANY=0
cleanup() {
    echo "[$(ts)] cleanup: ensuring fuser=0 and running glx_reset"
    if [ "$RAN_ANY" -eq 1 ]; then
        sleep 2
        if fuser /dev/tenstorrent/* >/dev/null 2>&1; then
            echo "[$(ts)] WARNING: devices still open at cleanup:"
            fuser -v /dev/tenstorrent/* 2>&1 || true
        else
            /usr/local/bin/tt-smi -glx_reset >/dev/null 2>&1
            echo "[$(ts)] glx_reset rc=$?"
        fi
    fi
    echo "DONE $(ts) sha=$SHA" >"$OUT/DONE"
    echo "=== [$(ts)] run_ccl_2120z.sh finished ==="
}
trap cleanup EXIT

run_step() {
    local label="$1"
    local tmo="$2"
    shift 2
    local now
    now=$(date -u +%s)
    if [ -f "$HOME/fleet/HANDBACK" ]; then
        echo "[$(ts)] HANDBACK before $label, stopping"
        return 1
    fi
    if [ "$now" -ge "$DEADLINE_EPOCH" ]; then
        echo "[$(ts)] DEADLINE reached before $label, stopping"
        return 1
    fi
    local rem=$(( DEADLINE_EPOCH - now ))
    if [ "$rem" -lt "$tmo" ]; then
        tmo="$rem"
    fi
    if [ "$tmo" -lt 30 ]; then
        echo "[$(ts)] <30s left before deadline, skipping $label"
        return 1
    fi
    RAN_ANY=1
    echo "--- [$(ts)] START $label (timeout=${tmo}s) ---"
    PYTHONPATH="$REPO" TORCH_THREADS=32 timeout --signal=TERM --kill-after=15 "$tmo" \
        "$PY" -u "$SCRIPT" "$OUT" --sha "$SHA" "$@"
    local rc=$?
    echo "--- [$(ts)] END $label rc=$rc ---"
    if [ "$rc" -ne 0 ]; then
        echo "[$(ts)] $label failed rc=$rc; running glx_reset before next step"
        sleep 2
        /usr/local/bin/tt-smi -glx_reset >/dev/null 2>&1
        echo "[$(ts)] recovery glx_reset rc=$?"
    fi
    return "$rc"
}

run_step "width_4 (1x4)" 180 --widths 1x4 --reps 5 --num-links 2 || true
run_step "width_8 (2x4)" 180 --widths 2x4 --reps 5 --num-links 2 || true
if ! run_step "width_16 (4x4) nl=2" 200 --widths 4x4 --reps 5 --num-links 2; then
    echo "[$(ts)] retrying 4x4 with --num-links 1 --ag-linear"
    run_step "width_16 (4x4) nl=1 ag-lin" 200 --widths 4x4 --reps 5 --num-links 1 --ag-linear || true
fi
run_step "dict558_32 (8x4)" 240 --widths 8x4 --dict-bench --reps 5 --num-links 2 || true
