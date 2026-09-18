#!/usr/bin/env bash
# The fold A/B for the fitted in0_block_w on the DiT's four square token projections.
# Interleaved rep by rep with base at BOTH ends of every rep (--palindrome), so the session
# carries its own A/A floor; benchlocked; clock forced and sampled DURING each fold.
set -u
WT=/home/ttuser/.coworker/wt/c14-matmul-ceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c14_matmul_ceiling
NODE="${NODE:-3}"
TAG="${1:-f1}"; shift || true
cd "$WT" || exit 1
python3 perf/c12_orchestrator/pair_guard/pair_idle.py --card "$NODE" || { echo "PAIR GUARD REFUSED"; exit 75; }
exec /home/ttuser/.coworker/scripts/benchlock.sh c14-matmul-ceiling -- env \
  TT_VISIBLE_DEVICES=$NODE TT_BIO_LEASE_CARDS=$NODE TT_BIO_LEASE_HOLDER=worker:c14-matmul-ceiling \
  "$PY" perf/c12_compose/fold_compose.py \
    --out "$OUT/fold_ab_$TAG.json" --cifs "$OUT/fold_ab_${TAG}_cifs" \
    --arms base,bw,base --reps 4 --size 512 --mhz 1350 --palindrome "$@"
