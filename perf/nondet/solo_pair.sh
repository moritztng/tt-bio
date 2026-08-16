#!/bin/bash
# solo_pair.sh <size> <rep> — one fresh-process protenix-v2 solo fold of cdk2_<size> at seed 0.
# Host-agnostic: WT derives from script location; override python with TT_BIO_PY,
# device with TT_VISIBLE_DEVICES (default 3), dump dir with TT_PROTENIX_DUMP.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIZE="${1:?usage: solo_pair.sh <size> <rep>}"
REP="${2:?usage: solo_pair.sh <size> <rep>}"
PY="${TT_BIO_PY:-}"
if [ -z "$PY" ]; then
  for c in /home/ttuser/tt-bio-dev/env/bin/python3.10 /home/moritz/tt-bio/env/bin/python3; do
    [ -x "$c" ] && PY="$c" && break
  done
fi
O=$WT/perf/nondet/out/solo_${SIZE}_${REP}
mkdir -p "$O"
cd "$WT" || exit 1
TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-3}" TT_METAL_LOGGER_LEVEL=FATAL \
TT_BIO_LEASE_HOLDER=worker:protenix-v2-nondeterminism-rootcause \
PYTHONPATH=. "$PY" -c 'from tt_bio.main import cli; cli()' predict \
    "$WT/perf/nondet/targets/cdk2_${SIZE}.yaml" \
    --model protenix-v2 --single_sequence --seed 0 \
    --out_dir "$O" > "$O/run.log" 2>&1
echo "EXIT=$? PY=$PY DEV=${TT_VISIBLE_DEVICES:-3}" >> "$O/run.log"
tail -2 "$O/run.log"
