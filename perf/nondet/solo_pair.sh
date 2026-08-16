#!/bin/bash
# solo_pair.sh <size> <rep> — one fresh-process protenix-v2 solo fold of cdk2_<size> at seed 0.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-nondeterminism-rootcause
PY=/home/ttuser/tt-bio-dev/env/bin/python3.10
SIZE="${1:?usage: solo_pair.sh <size> <rep>}"
REP="${2:?usage: solo_pair.sh <size> <rep>}"
O=$WT/perf/nondet/out/solo_${SIZE}_${REP}
mkdir -p "$O"
cd "$WT" || exit 1
TT_VISIBLE_DEVICES=3 TT_METAL_LOGGER_LEVEL=FATAL \
TT_BIO_LEASE_HOLDER=worker:protenix-v2-nondeterminism-rootcause \
PYTHONPATH=. "$PY" -c 'from tt_bio.main import cli; cli()' predict \
    "$WT/perf/nondet/targets/cdk2_${SIZE}.yaml" \
    --model protenix-v2 --single_sequence --seed 0 \
    --out_dir "$O" > "$O/run.log" 2>&1
echo "EXIT=$?" >> "$O/run.log"
ls "$O"
