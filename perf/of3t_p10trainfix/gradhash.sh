#!/bin/bash
# gradhash.sh <tag>... : one-step gradient-hash runs on qb2 card ${CARD:-1}, sequential.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/of3t-p10trainfix
O=$W/perf/of3t_p10trainfix/out
C=/home/ttuser/of3t_p10trainout/corpus
cd "$W" || exit 1
mkdir -p "$L" "$O"
for tag in "$@"; do
  echo "=== gh_$tag $(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!${CARD:-1}/tt_aiclk)" >> "$L/gradhash.log"
  env TT_VISIBLE_DEVICES=${CARD:-1} TT_BIO_LEASE_CARDS=${CARD:-1} TT_BIO_LEASE_HOLDER=worker:of3t-p10trainfix \
    timeout 900 "$PY" perf/of3t_p10trainfix/${PROBE:-gradhash}.py --hash-out "$O/${PROBE:-gh}_$tag.json" \
    --corpus $C/train12 --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt --exact off --steps 1 \
    --checkpoint-every 9 --displacement-band 0.01,100 --seed 0 \
    --out-dir /home/ttuser/of3t_p10trainfix/runs/gh_$tag --out "$O/gharm_$tag.json" > "$L/gh_$tag.log" 2>&1
  echo "=== gh_$tag rc=$? $(date -u +%FT%TZ)" >> "$L/gradhash.log"
done
