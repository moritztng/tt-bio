#!/usr/bin/env bash
# run_gate.sh <tree> <tag>  — four committed affinity legs, envelope gate, margin 0.5
set -u
TREE="$1"; TAG="$2"
OUT=/home/ttuser/.coworker/wt/boltz2-affinity-device-only/perf/boltz2-affinity-device-only
cd "$TREE"
export PYTHONPATH="$TREE"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:boltz2-affinity-device-only
/home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py --workers localhost:0 \
  --leg boltz2-affinity-fkbp12-nomsa --leg boltz2-affinity-fkbp12-msa \
  --leg boltz2-affinity-dhfr-nomsa  --leg boltz2-affinity-dhfr-msa \
  --margin 0.5 --workdir /tmp/gate_$TAG --out $OUT/gate_$TAG.json >> $OUT/gate_$TAG.log 2>&1
echo "EXIT=$? tag=$TAG $(date -u +%FT%TZ)" >> $OUT/gate_$TAG.log
