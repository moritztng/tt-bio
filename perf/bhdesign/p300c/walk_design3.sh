#!/usr/bin/env bash
# Round 3, card 3: push both design models past the fixture top that made them PARTIAL. The
# largest single deposited chain on hand is 1008 residues / 8095 atoms; big_7324.cif stacks
# laczc chain A and gpb chain A four times each, every chain keeping its deposited geometry and
# translated clear along +x. That reaches 7324 residues / 59144 atoms without inventing a
# coordinate, so the walk can stop at a failure instead of at the end of the fixture shelf.
set -u
cd /home/ttuser/.coworker/wt/p300c-1536-design-embed
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
T=perf/bhdesign/targets/big_7324.cif
run() {
  TT_BIO_LEASE_TIMEOUT=1800 BH_LEASE_CARDS=2,3 $PY perf/bhdesign/ladder.py \
    --model "$1" --sizes "$2" --card 3 --board p300c --target "$T" \
    --holder worker:p300c-1536-design-embed --stop-on-fail --timeout 1200 \
    --out "$D/$1.jsonl" --work "$D/work"
}
run boltzgen 1831,3662,7324
run pxdesign 3662,7324
echo "WALK_DESIGN3_DONE"
