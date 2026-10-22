#!/bin/bash
# bc4-bet-mm2: run one perf/bc4_mm2 script on a qb2 chip, detached.
# usage: CHIP=2 perf/bc4_mm2/run.sh <out-name> <script.py> [args...]   marker <out>/done
set -u
NAME=$1; SCRIPT=$2; shift 2
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT" || exit 1
OUT=$WT/_mm2/$NAME; mkdir -p "$OUT"
C=${CHIP:?}
git rev-parse HEAD > "$OUT/sha"; git diff > "$OUT/diff.patch"; cat /proc/loadavg > "$OUT/loadavg.start"
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bc4-bet-mm2
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params TT_BIO_LEASE_TIMEOUT=7200
export JAPANFOLD_BC2_MPNN_WEIGHTS=$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral
export PYTHONPATH=$WT
unset LD_LIBRARY_PATH
flock -w 60 "$HOME/spd_qb2_chip$C.lock" timeout "${TMO:-2400}" "$HOME/bcx_e2e_venv/bin/python3" \
    "$SCRIPT" --out "$OUT/out.json" "$@" > "$OUT/log" 2>&1
echo "rc=$? $(date -u +%FT%TZ) $(cat /proc/loadavg)" > "$OUT/done"
