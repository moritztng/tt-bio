#!/bin/bash
# tri512 device probe for a kernel that may hang: one short tail leg under a hard limit; on a hang the process is
# interrupted, killed if it must be, and the chip reset with tt-smi so the next leg starts clean. rc 0 = it ran.
#   bash perf/rfd4/tri512/probe.sh <chip> <out.jsonl> <legs>      (run from a snapshot tree)
set -u
N=$1; OUT=$2; LEGS=$3; WT=$(cd "$(dirname "$0")/../../.." && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri512 PYTHONPATH=$WT:$WT/perf
$HOME/tt-bio-dev/env/bin/python perf/rfd4/g512/tail.py --ns 384,512 --legs "$LEGS" --rounds 1 --reps 5 --out $OUT &
P=$!
for _ in $(seq 60); do kill -0 $P 2>/dev/null || break; sleep 3; done
if kill -0 $P 2>/dev/null; then
  kill -INT $P; sleep 10; kill -0 $P 2>/dev/null && kill -9 $P; sleep 2
  timeout 120 $HOME/.local/bin/tt-smi -r $N > /dev/null 2>&1
  echo "HUNG, chip $N reset"; exit 2
fi
wait $P
