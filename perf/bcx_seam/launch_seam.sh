#!/bin/bash
# bcx-seam rounds on qb2 card 3: same arm and environment as perf/bcx_round/launch_round.sh.
#   launch_seam.sh <out-name> <run_seam.py args...>
# DUMP=1 adds the optimised-HLO dump hostmap.py needs.
set -u
WT=/home/ttuser/.coworker/wt/bcx-seam
cd "$WT" || exit 1
NAME=$1; shift
OUT=$WT/perf/bcx_seam/runs/$NAME
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-seam
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
if [ "${DUMP:-0}" = 1 ]; then
  XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$OUT/hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*sequence_design_loss.*"
fi
exec /home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_seam/run_seam.py" --out "$OUT" "$@"
