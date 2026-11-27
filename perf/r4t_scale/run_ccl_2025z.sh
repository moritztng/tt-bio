#!/usr/bin/env bash
# Detached runner for the 2026-11-27 20:25-20:40Z whole-box 32-chip exact 2D Ring CCL window on anx-glx.
# Hard-stops all device work by 20:38:00Z (and runs glx_reset only on failure) so vrt-fluid gets
# 32/32 clean chips at 20:41Z.
set -u

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
PY="${1:-/home/ttuser/fleet/env/bin/python}"
REPO="${2:-/home/ttuser/fleet/r4t/ccl/repo}"
OUT="${3:-/home/ttuser/fleet/r4t/ccl/out_2025z}"
START_EPOCH="${4:-1795811100}"     # 2026-11-27T20:25:00Z
GIVEUP_EPOCH="${5:-1795811520}"    # 2026-11-27T20:32:00Z
DEADLINE_EPOCH="${6:-1795811850}"  # 2026-11-27T20:37:30Z
HARD_STOP_EPOCH="${7:-1795811880}" # 2026-11-27T20:38:00Z

mkdir -p "$OUT"
LOG="$OUT/run.log"
exec >>"$LOG" 2>&1

ts() { date -u +%FT%TZ; }
fuser_count() { fuser /dev/tenstorrent/* 2>/dev/null | wc -w; }
do_reset() {
    echo "[$(ts)] running tt-smi -glx_reset..."
    timeout 120 /usr/local/bin/tt-smi -glx_reset >"$OUT/reset_$(date -u +%H%M%S).log" 2>&1
    echo "[$(ts)] glx_reset rc=$? fuser=$(fuser_count)"
}

echo "[$(ts)] ARM run_ccl_2025z repo=$REPO out=$OUT start=$START_EPOCH deadline=$DEADLINE_EPOCH hard_stop=$HARD_STOP_EPOCH"
echo "[$(ts)] git_sha=$(git -C "$REPO" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)"

# 1. Wait for START_EPOCH and zero device holders
while true; do
    now=$(date -u +%s)
    if [ "$now" -ge "$GIVEUP_EPOCH" ]; then
        echo "[$(ts)] GAVE-UP: box still busy at giveup_epoch (fuser=$(fuser_count))"
        echo "DONE_2025Z rc=2 $(ts)"
        exit 2
    fi
    if [ "$now" -lt "$START_EPOCH" ]; then
        sleep 5
        continue
    fi
    cnt=$(fuser_count)
    if [ "$cnt" -eq 0 ]; then
        echo "[$(ts)] BOX FREE (fuser=0); starting probe ladder"
        break
    fi
    echo "[$(ts)] waiting for fuser=0 (currently $cnt)"
    sleep 5
done

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TORCH_THREADS=32
SHA=$(git -C "$REPO" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)
SCRIPT="$REPO/perf/r4t_scale/ccl_allreduce_2d.py"

# 2. Fast probe ladder on 8x4 (tries best config first, falls back with clean glx_reset if needed)
MODE=""
PROBES=(
    "--num-links 2"
    "--num-links 2 --ag-linear"
    "--num-links 1 --ag-linear"
    "--num-links 1 --ag-linear --force-linear"
)
for idx in "${!PROBES[@]}"; do
    now=$(date -u +%s)
    if [ "$now" -ge "$((DEADLINE_EPOCH - 90))" ]; then
        echo "[$(ts)] ABORT probe ladder: too close to deadline"
        break
    fi
    flags="${PROBES[$idx]}"
    echo "[$(ts)] PROBE[$idx]: $flags"
    timeout --signal=TERM --kill-after=10 90 \
        "$PY" "$SCRIPT" "$OUT" --widths 8x4 --max-payload 8192 --probe-only --sha "$SHA" $flags \
        >"$OUT/probe_${idx}.log" 2>&1
    prc=$?
    tail -n 6 "$OUT/probe_${idx}.log"
    echo "[$(ts)] PROBE[$idx] rc=$prc fuser=$(fuser_count)"
    if [ "$prc" -eq 0 ]; then
        MODE="$flags"
        break
    fi
    # Kill any stray descendant and reset before next probe
    pkill -9 -f "[c]cl_allreduce_2d.py" 2>/dev/null || true
    sleep 1
    do_reset
done

if [ -z "$MODE" ]; then
    echo "[$(ts)] ALL PROBES FAILED"
    echo "DONE_2025Z rc=1 $(ts)"
    exit 1
fi

# 3. Full 4-width sweep (8x4 -> 4x4 -> 2x4 -> 1x4) inside hard wall
now=$(date -u +%s)
rem=$((HARD_STOP_EPOCH - now))
if [ "$rem" -lt 60 ]; then
    echo "[$(ts)] ABORT sweep: only ${rem}s before hard stop"
    echo "DONE_2025Z rc=3 $(ts)"
    exit 3
fi

echo "[$(ts)] SWEEP start: MODE='$MODE' rem=${rem}s"
timeout --signal=TERM --kill-after=15 "$rem" \
    "$PY" "$SCRIPT" "$OUT" --widths 8x4,4x4,2x4,1x4 --max-payload 8192 --reps 5 \
    --deadline "$DEADLINE_EPOCH" --sha "$SHA" $MODE \
    >"$OUT/sweep.log" 2>&1
src=$?
tail -n 30 "$OUT/sweep.log"
echo "[$(ts)] SWEEP exit rc=$src fuser=$(fuser_count)"

if [ "$src" -ne 0 ] || [ "$(fuser_count)" -ne 0 ]; then
    pkill -9 -f "[c]cl_allreduce_2d.py" 2>/dev/null || true
    sleep 1
    do_reset
fi

echo "[$(ts)] FINAL fuser=$(fuser_count) json_files=$(ls -1 "$OUT"/ring2d_*.json 2>/dev/null | tr '\n' ' ')"
echo "DONE_2025Z rc=$src $(ts)"
exit "$src"
