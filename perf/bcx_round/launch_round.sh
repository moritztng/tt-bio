#!/bin/bash
# 14 consecutive gradient rounds of BindCraft 2's own loop, timed from the loop's side.
# Same arm and same environment as the acceptance run post_seed100 (bucket 32, seed 100,
# 6 OMP threads, the same XLA flag), on card 0 of qb2.
set -u
WT=/home/ttuser/.coworker/wt/bcx-round
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=${CARD:-3} TT_BIO_LEASE_CARDS=0,${CARD:-3} TT_BIO_LEASE_HOLDER=worker:bcx-round
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
OUT=$WT/perf/bcx_round/runs/round_seed100_card${CARD:-3}
mkdir -p "$OUT"
exec /home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_round/run_round.py" \
  --rounds "${1:-14}" --seed 100 --bucket 0 --out "$OUT"
