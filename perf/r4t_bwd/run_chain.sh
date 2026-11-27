#!/bin/bash
# r4t-bwd full device verification chain on qb2 card 2 (waits for fleet-r4t-bwd booking + flock):
#   1. screen_mm.py (B2 dW + B3 dX + B4a/b/c micro-screen + float64 VJP + broken arms)
#   2. gradcheck_rms.py (B4b RMSNorm fused kernel float64 FD-validated VJP + --break FAIL)
#   3. l4grade.py (B3/B2 MMVJP float64 VJP + TT_BIO_MMVJP_BROKEN=1 FAIL)
#   4. block_gradcheck.py PairUpdate bf16 (D39 floor) + --break FAIL + fp32 arm with B2+B3+B4 ON
#   5. grade_b1.py (B1 D37 9-batch loss parity + broken arm FAIL + inference A/A byte-identical)
#   6. Paired ABBA PERF benchmark on trunk_step (crop 256, k=4, depth 2, attn_fp32=on, 7 steps = 2 warm + 5 timed)
set -uo pipefail
C=${1:-2}
WT=${2:-/home/ttuser/.coworker/wt/r4t-bwd}
O=${3:-/home/ttuser/r4t/bwd/run1}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
mkdir -p "$O"
cd "$WT" || exit 1
echo "WAIT $(date -u +%FT%TZ) card=$C sha=$(git rev-parse --short=9 HEAD)" > "$O/run.log"

GIVE=$(date -u -d "2026-11-27 23:30" +%s)
while true; do
  [ "$(date +%s)" -gt "$GIVE" ] && { echo "GAVE-UP $(date -u +%FT%TZ)" >> "$O/run.log"; exit 4; }
  act=$(python3 ~/spd/feed/book.py active "$C" 2>/dev/null || true)
  other=$(python3 ~/spd/feed/book.py active "$C" --except fleet-r4t-bwd 2>/dev/null || true)
  if echo "$act" | grep -qw fleet-r4t-bwd && [ -z "$other" ] && ! fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; then
    break
  fi
  sleep 15
done

exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { echo "FLOCK-TIMEOUT $(date -u +%FT%TZ)" >> "$O/run.log"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 5; done

( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 2; done ) > "$O/aiclk.log" 2>&1 &
SMP=$!
cleanup() {
  kill "$SMP" 2>/dev/null || true
}
trap cleanup EXIT

export PYTHONPATH="$WT" TT_VISIBLE_DEVICES="$C" TT_BIO_LEASE_CARDS="$C" TT_BIO_LEASE_HOLDER="worker:r4t-bwd"
export TT_BIO_SHA="$(git rev-parse --short=9 HEAD)" OMP_NUM_THREADS=2 R4T_REBUILD_INLINE=1 TT_BIO_SDPA_OWN_FORWARD=1
unset TT_BIO_LEVERS TT_BIO_TAPED_KERNELS
echo "START $(date -u +%FT%TZ) sha=$TT_BIO_SHA card=$C load=$(cut -d' ' -f1 /proc/loadavg)" >> "$O/run.log"

# 1. B2 DW_2D + split-K float64 grade (normal PASS + BREAK=1 FAIL) & micro-screen B2/B3/B4
timeout 600 "$PY" "$WT/perf/r4t_bwd/dw2d_grade.py" --out "$O/dw2d_gc.json" > "$O/dw2d_gc.log" 2>&1
echo "RC_dw2d_gc $? $(date -u +%T)" >> "$O/run.log"
BREAK=1 timeout 600 "$PY" "$WT/perf/r4t_bwd/dw2d_grade.py" --out "$O/dw2d_brk.json" > "$O/dw2d_brk.log" 2>&1
echo "RC_dw2d_brk $? $(date -u +%T)" >> "$O/run.log"

timeout 900 "$PY" "$WT/perf/r4t_bwd/screen_mm.py" --out "$O/screen_mm.json" --reps 5 > "$O/screen_mm.log" 2>&1
echo "RC_screen_mm $? $(date -u +%T)" >> "$O/run.log"

# 2. B4b RMSNorm gradcheck (FD-validated float64 + --break negative control)
TT_BIO_RMSBW_FUSED=1 timeout 600 "$PY" "$WT/perf/r4t_tape/gradcheck_rms.py" --out "$O/rms_gc.json" > "$O/rms_gc.log" 2>&1
echo "RC_rms_gc $? $(date -u +%T)" >> "$O/run.log"
TT_BIO_RMSBW_FUSED=1 timeout 600 "$PY" "$WT/perf/r4t_tape/gradcheck_rms.py" --break --out "$O/rms_brk.json" > "$O/rms_brk.log" 2>&1
echo "RC_rms_brk $? $(date -u +%T)" >> "$O/run.log"

# 3. B2/B3 L4 gradcheck + broken arm
TT_BIO_MMVJP_MIN=1 timeout 600 "$PY" "$WT/perf/r4t_perf/l4grade.py" --out "$O/l4_gc.json" > "$O/l4_gc.log" 2>&1
echo "RC_l4_gc $? $(date -u +%T)" >> "$O/run.log"
TT_BIO_MMVJP_MIN=1 TT_BIO_MMVJP_BROKEN=1 timeout 600 "$PY" "$WT/perf/r4t_perf/l4grade.py" --out "$O/l4_brk.json" > "$O/l4_brk.log" 2>&1
echo "RC_l4_brk $? $(date -u +%T)" >> "$O/run.log"

# 4. PairUpdate block_gradcheck (bf16 D39 floor + --break + TrunkBlock) with B2+B3+B4 ON
env TT_BIO_LNBW_WEIGHTS=1 TT_BIO_RMSBW_FUSED=1 TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_WIDEN_ADD=1 TT_BIO_FANIN_CAST_FUSED=1 \
  timeout 900 "$PY" "$WT/perf/r4t_tape/block_gradcheck.py" --module pair_update --n 64 --real 48 --probe 8 --seed 7 --arm bf16 \
  --floor "$WT/perf/r4t_tape/floor/pu_n64r48_s7.json" --out "$O/pu_on.json" > "$O/pu_on.log" 2>&1
echo "RC_pu_on $? $(date -u +%T)" >> "$O/run.log"

env TT_BIO_LNBW_WEIGHTS=1 TT_BIO_RMSBW_FUSED=1 TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_WIDEN_ADD=1 TT_BIO_FANIN_CAST_FUSED=1 \
  timeout 900 "$PY" "$WT/perf/r4t_tape/block_gradcheck.py" --module pair_update --n 64 --real 48 --probe 8 --seed 7 --arm bf16 --break \
  --floor "$WT/perf/r4t_tape/floor/pu_n64r48_s7.json" --out "$O/pu_brk.json" > "$O/pu_brk.log" 2>&1
echo "RC_pu_brk $? $(date -u +%T)" >> "$O/run.log"

env TT_BIO_LNBW_WEIGHTS=1 TT_BIO_RMSBW_FUSED=1 TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_WIDEN_ADD=1 TT_BIO_FANIN_CAST_FUSED=1 \
  timeout 900 "$PY" "$WT/perf/r4t_tape/block_gradcheck.py" --module trunk_block --n 64 --real 48 --probe 8 --seed 7 --arm bf16 --attn-fp32 \
  --out "$O/tb_on.json" > "$O/tb_on.log" 2>&1
echo "RC_tb_on $? $(date -u +%T)" >> "$O/run.log"

# 5. B1 9-batch D37 loss-parity grade + broken arm + inference A/A
B9=$(ls "$HOME"/r4t/step/batch0?.npz | paste -sd,)
timeout 1200 "$PY" "$WT/perf/r4t_bwd/grade_b1.py" --batches "$B9" --out "$O/grade_b1.json" > "$O/grade_b1.log" 2>&1
echo "RC_grade_b1 $? $(date -u +%T)" >> "$O/run.log"

# Pick best passing B1 arm automatically from grade_b1.json
B1_FLAGS=$(python3 -c '
import json
d = json.load(open("'"$O"'/grade_b1.json"))["arms"]
if d["b1_reuse_serve"]["passed"]:
    print("--cand-levers serve --win-pre reuse")
elif d["b1_reuse_train"]["passed"]:
    print("--cand-levers serve_train --win-pre reuse")
else:
    print("--cand-levers serve --win-pre suspend")
' 2>/dev/null || echo "--cand-levers serve --win-pre suspend")
echo "SELECTED_B1 $B1_FLAGS" >> "$O/run.log"

# 6. Paired ABBA PERF benchmark (crop 256, k=4, depth 2, attn_fp32=on, 7 steps = 2 warmup + 5 warm timed)
B0="$HOME/r4t/step/batch00.npz"
run_step() {
  local name=$1; shift
  local env_args=$1; shift
  env $env_args timeout 900 "$PY" "$WT/perf/r4t_step/trunk_step.py" \
    --n 256 --k 4 --depth 2 --steps 7 --batch "$B0" --attn-fp32 on --phase-timing \
    --exempt "$WT/perf/r4t_step/g2_exempt.tsv" --out "$O/ck_$name" "$@" > "$O/ab_$name.log" 2>&1
  echo "RC_ab_$name $? $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)" >> "$O/run.log"
  rm -rf "$O/ck_$name"
}

BASE_ENV="TT_BIO_DW_2D=0 TT_BIO_MMVJP_MIN=0 TT_BIO_SPLIT_K_OUT_TILES=32 TT_BIO_SUM_LEADING_SPLIT=0 TT_BIO_RMSBW_FUSED=0 TT_BIO_WIDEN_ADD=0 TT_BIO_FANIN_CAST_FUSED=0"
B123_ENV="TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_SPLIT_K_OUT_TILES=144 TT_BIO_SUM_LEADING_SPLIT=0 TT_BIO_RMSBW_FUSED=0 TT_BIO_WIDEN_ADD=0 TT_BIO_FANIN_CAST_FUSED=0"
ALL_ENV="TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_SPLIT_K_OUT_TILES=144 TT_BIO_SUM_LEADING_SPLIT=1 TT_BIO_RMSBW_FUSED=1 TT_BIO_WIDEN_ADD=1 TT_BIO_FANIN_CAST_FUSED=1"

run_step base_1 "$BASE_ENV" --cand-levers serve --win-pre tape
run_step b1_1   "$BASE_ENV" $B1_FLAGS
run_step b123_1 "$B123_ENV" $B1_FLAGS
run_step all_1  "$ALL_ENV"  $B1_FLAGS
run_step all_2  "$ALL_ENV"  $B1_FLAGS
run_step b123_2 "$B123_ENV" $B1_FLAGS
run_step b1_2   "$BASE_ENV" $B1_FLAGS
run_step base_2 "$BASE_ENV" --cand-levers serve --win-pre tape

echo "END $(date -u +%FT%TZ)" >> "$O/run.log"
