#!/bin/bash
# r4t-bwd 4-chip parallel verification runner on anx-glx (BH Galaxy 120c, chips C0..C3 inside 0..15).
# ONLY run AFTER 20:40Z and AFTER verifying state/anx-glx/HOLDER.windows has no conflicting window!
set -uo pipefail
C0=${1:-0}; C1=${2:-1}; C2=${3:-2}; C3=${4:-3}
WT=${5:-/home/ttuser/r4t/bwd/wt}
O=${6:-/home/ttuser/r4t/bwd/anx_run1}
PY=/home/ttuser/fleet/env/bin/python
BATCH_DIR=/home/ttuser/r4t/bwd/batches
mkdir -p "$O"
cd "$WT" || exit 1

if [ -f "$HOME/fleet/HANDBACK" ]; then
  echo "ABORT HANDBACK present" | tee "$O/run.log"; exit 2
fi
for c in "$C0" "$C1" "$C2" "$C3"; do
  if fuser "/dev/tenstorrent/$c" >/dev/null 2>&1; then
    echo "ABORT chip $c busy" | tee "$O/run.log"; exit 3
  fi
done

export PYTHONPATH="$WT" OMP_NUM_THREADS=2 R4T_REBUILD_INLINE=1 TT_BIO_SDPA_OWN_FORWARD=1
export TT_BIO_SHA="$(git rev-parse --short=9 HEAD)"
unset TT_BIO_LEVERS TT_BIO_TAPED_KERNELS
echo "START $(date -u +%FT%TZ) host=$(hostname) sha=$TT_BIO_SHA chips=$C0,$C1,$C2,$C3 load=$(cut -d' ' -f1 /proc/loadavg)" > "$O/run.log"

# Lane 0 (chip C0): B2 DW_2D + split-K grade (normal + BREAK) & B2/B3/B4 micro-screen
(
  export TT_VISIBLE_DEVICES="$C0" TT_BIO_LEASE_CARDS="$C0" TT_BIO_LEASE_HOLDER="worker:r4t-bwd"
  timeout 600 "$PY" "$WT/perf/r4t_bwd/dw2d_grade.py" --out "$O/dw2d_gc.json" > "$O/dw2d_gc.log" 2>&1
  echo "L0 RC_dw2d_gc $? $(date -u +%T)" >> "$O/run.log"
  BREAK=1 timeout 600 "$PY" "$WT/perf/r4t_bwd/dw2d_grade.py" --out "$O/dw2d_brk.json" > "$O/dw2d_brk.log" 2>&1
  echo "L0 RC_dw2d_brk $? $(date -u +%T)" >> "$O/run.log"
  timeout 900 "$PY" "$WT/perf/r4t_bwd/screen_mm.py" --out "$O/screen_mm.json" --reps 5 > "$O/screen_mm.log" 2>&1
  echo "L0 RC_screen_mm $? $(date -u +%T)" >> "$O/run.log"
) &
P0=$!

# Lane 1 (chip C1): RMSNorm + L4 + PairUpdate + TrunkBlock VJP gradchecks (+ broken arms)
(
  export TT_VISIBLE_DEVICES="$C1" TT_BIO_LEASE_CARDS="$C1" TT_BIO_LEASE_HOLDER="worker:r4t-bwd"
  TT_BIO_RMSBW_FUSED=1 timeout 600 "$PY" "$WT/perf/r4t_tape/gradcheck_rms.py" --out "$O/rms_gc.json" > "$O/rms_gc.log" 2>&1
  echo "L1 RC_rms_gc $? $(date -u +%T)" >> "$O/run.log"
  TT_BIO_RMSBW_FUSED=1 timeout 600 "$PY" "$WT/perf/r4t_tape/gradcheck_rms.py" --break --out "$O/rms_brk.json" > "$O/rms_brk.log" 2>&1
  echo "L1 RC_rms_brk $? $(date -u +%T)" >> "$O/run.log"
  TT_BIO_MMVJP_MIN=1 timeout 600 "$PY" "$WT/perf/r4t_perf/l4grade.py" --out "$O/l4_gc.json" > "$O/l4_gc.log" 2>&1
  echo "L1 RC_l4_gc $? $(date -u +%T)" >> "$O/run.log"
  TT_BIO_MMVJP_MIN=1 TT_BIO_MMVJP_BROKEN=1 timeout 600 "$PY" "$WT/perf/r4t_perf/l4grade.py" --out "$O/l4_brk.json" > "$O/l4_brk.log" 2>&1
  echo "L1 RC_l4_brk $? $(date -u +%T)" >> "$O/run.log"
  env TT_BIO_LNBW_WEIGHTS=1 TT_BIO_RMSBW_FUSED=1 TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_WIDEN_ADD=1 TT_BIO_FANIN_CAST_FUSED=1 \
    timeout 900 "$PY" "$WT/perf/r4t_tape/block_gradcheck.py" --module pair_update --n 64 --real 48 --probe 8 --seed 7 --arm bf16 \
    --floor "$WT/perf/r4t_tape/floor/pu_n64r48_s7.json" --out "$O/pu_on.json" > "$O/pu_on.log" 2>&1
  echo "L1 RC_pu_on $? $(date -u +%T)" >> "$O/run.log"
  env TT_BIO_LNBW_WEIGHTS=1 TT_BIO_RMSBW_FUSED=1 TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_WIDEN_ADD=1 TT_BIO_FANIN_CAST_FUSED=1 \
    timeout 900 "$PY" "$WT/perf/r4t_tape/block_gradcheck.py" --module pair_update --n 64 --real 48 --probe 8 --seed 7 --arm bf16 --break \
    --floor "$WT/perf/r4t_tape/floor/pu_n64r48_s7.json" --out "$O/pu_brk.json" > "$O/pu_brk.log" 2>&1
  echo "L1 RC_pu_brk $? $(date -u +%T)" >> "$O/run.log"
  env TT_BIO_LNBW_WEIGHTS=1 TT_BIO_RMSBW_FUSED=1 TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_WIDEN_ADD=1 TT_BIO_FANIN_CAST_FUSED=1 \
    timeout 900 "$PY" "$WT/perf/r4t_tape/block_gradcheck.py" --module trunk_block --n 64 --real 48 --probe 8 --seed 7 --arm bf16 --attn-fp32 \
    --out "$O/tb_on.json" > "$O/tb_on.log" 2>&1
  echo "L1 RC_tb_on $? $(date -u +%T)" >> "$O/run.log"
) &
P1=$!

# Lane 2 (chip C2): B1 9-batch D37 loss-parity + final-cycle inputs + broken arm + inference A/A
(
  export TT_VISIBLE_DEVICES="$C2" TT_BIO_LEASE_CARDS="$C2" TT_BIO_LEASE_HOLDER="worker:r4t-bwd"
  B9=$(ls "$BATCH_DIR"/batch0?.npz | paste -sd,)
  timeout 1200 "$PY" "$WT/perf/r4t_bwd/grade_b1.py" --batches "$B9" --out "$O/grade_b1.json" > "$O/grade_b1.log" 2>&1
  echo "L2 RC_grade_b1 $? $(date -u +%T)" >> "$O/run.log"
) &
P2=$!

# Lane 3 (chip C3): Paired ABBA PERF benchmark (crop 256, k=4, depth 2, attn_fp32=on, 7 steps = 2 warm + 5 timed)
(
  export TT_VISIBLE_DEVICES="$C3" TT_BIO_LEASE_CARDS="$C3" TT_BIO_LEASE_HOLDER="worker:r4t-bwd"
  B0="$BATCH_DIR/batch00.npz"
  run_step() {
    local name=$1; shift
    local env_args=$1; shift
    env $env_args timeout 900 "$PY" "$WT/perf/r4t_step/trunk_step.py" \
      --n 256 --k 4 --depth 2 --steps 7 --batch "$B0" --attn-fp32 on --phase-timing \
      --exempt "$WT/perf/r4t_step/g2_exempt.tsv" --out "$O/ck_$name" "$@" > "$O/ab_$name.log" 2>&1
    echo "L3 RC_ab_$name $? $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)" >> "$O/run.log"
    rm -rf "$O/ck_$name"
  }
  BASE_ENV="TT_BIO_DW_2D=0 TT_BIO_MMVJP_MIN=0 TT_BIO_SPLIT_K_OUT_TILES=32 TT_BIO_SUM_LEADING_SPLIT=0 TT_BIO_RMSBW_FUSED=0 TT_BIO_WIDEN_ADD=0 TT_BIO_FANIN_CAST_FUSED=0"
  B123_ENV="TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_SPLIT_K_OUT_TILES=144 TT_BIO_SUM_LEADING_SPLIT=0 TT_BIO_RMSBW_FUSED=0 TT_BIO_WIDEN_ADD=0 TT_BIO_FANIN_CAST_FUSED=0"
  ALL_ENV="TT_BIO_DW_2D=1 TT_BIO_MMVJP_MIN=1 TT_BIO_SPLIT_K_OUT_TILES=144 TT_BIO_SUM_LEADING_SPLIT=1 TT_BIO_RMSBW_FUSED=1 TT_BIO_WIDEN_ADD=1 TT_BIO_FANIN_CAST_FUSED=1"

  run_step base_1  "$BASE_ENV" --cand-levers serve --win-pre tape
  run_step susp_1  "$BASE_ENV" --cand-levers serve --win-pre suspend
  run_step reuse_1 "$BASE_ENV" --cand-levers serve --win-pre reuse
  run_step b123_1  "$B123_ENV" --cand-levers serve --win-pre reuse
  run_step all_1   "$ALL_ENV"  --cand-levers serve --win-pre reuse
  run_step all_2   "$ALL_ENV"  --cand-levers serve --win-pre reuse
  run_step b123_2  "$B123_ENV" --cand-levers serve --win-pre reuse
  run_step reuse_2 "$BASE_ENV" --cand-levers serve --win-pre reuse
  run_step susp_2  "$BASE_ENV" --cand-levers serve --win-pre suspend
  run_step base_2  "$BASE_ENV" --cand-levers serve --win-pre tape
) &
P3=$!

wait "$P0" "$P1" "$P2" "$P3"
echo "END $(date -u +%FT%TZ) load=$(cut -d' ' -f1 /proc/loadavg)" >> "$O/run.log"
