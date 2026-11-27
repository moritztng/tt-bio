#!/usr/bin/env bash
# Grade the assembled score network against upstream's own, on several complexes at several
# sigmas (orchestrator D-dfd-orch-8). For each complex: dump upstream's score-network inputs
# and outputs in the reference env, then replay the port on them and record PCC per output.
# Sequential on purpose -- pc is shared and each arm loads ESM-2 650M.
set -u
WT=/home/moritz/.coworker/wt/dfd-parity
UP=/home/moritz/scratch/dfd/upstream
REF=/home/moritz/scratch/dfd/venv-ref/bin/python
PORT=/home/moritz/tt-bio/env/bin/python3.10
STEPS=${STEPS:-0,5,10,15,18}   # upstream makes 19 forward calls, 0..18
OUT=${OUT:-/tmp/dfd_assembled}
mkdir -p "$OUT"
for cid in "$@"; do
  echo "=== $cid $(date -Is)"
  if [ ! -s "$OUT/$cid.npz" ]; then
    (cd "$UP" && PYTHONPATH=. nice -n 5 "$REF" \
      "$WT/scripts/diffdock_port/score_net_parity.py" \
      --id "$cid" --steps "$STEPS" --out "$OUT/$cid.npz") \
      > "$OUT/$cid.dump.log" 2>&1
    echo "  dump rc=$? size=$(stat -c %s "$OUT/$cid.npz" 2>/dev/null || echo 0)"
  else
    echo "  dump already present, reusing"
  fi
  [ -s "$OUT/$cid.npz" ] || { echo "  NO DUMP, skipping grade"; continue; }
  (cd "$WT" && TT_VISIBLE_DEVICES= PYTHONPATH="$WT" nice -n 5 "$PORT" \
    scripts/diffdock_port/score_net_grade.py --npz "$OUT/$cid.npz" --out "$OUT/$cid.grade.json") \
    > "$OUT/$cid.grade.log" 2>&1
  echo "  grade rc=$?"
  grep -E "^step" "$OUT/$cid.grade.log" || tail -3 "$OUT/$cid.grade.log"
done
echo "=== ALL DONE $(date -Is)"
