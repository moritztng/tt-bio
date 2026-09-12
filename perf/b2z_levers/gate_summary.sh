#!/usr/bin/env bash
# Report the gate's own verdict for every leg this tree has already folded. Each one resumes
# from its cached <leg>.json, so this prints the authoritative table without re-folding.
set -u
cd /home/ttuser/.coworker/wt/b2z-levers-default-on
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export ESM_ROOT=/home/ttuser/esm
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:b2z-levers-default-on
args=()
for f in perf/b2z_levers/gate_work/*.json; do
  b=$(basename "$f" .json)
  [ "$b" = GATE_CODE ] && continue
  [ "$b" = report ] && continue
  args+=(--leg "$b")
done
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py \
  --workdir perf/b2z_levers/gate_work \
  --out perf/b2z_levers/gate_summary.json \
  --workers localhost:0 "${args[@]}"
