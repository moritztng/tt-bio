#!/bin/bash
# One training arm on qb2, card ${CARD:-3}. Rooted in this row's own worktree.
#   [CARD=n] armrun.sh <tag> <on|off> <steps> <corpus> [extra args...]
set -uo pipefail
# The worktree this script is IN. Pinned to one path, an arm launched from another worktree
# measures that worktree's tt_bio and stamps this one's sha.
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=/home/ttuser/tt-bio-dev/env/bin/python
ROW=${ROW:-of3t-p10trainout}
L=/tmp/of3t/$ROW
O=$W/perf/of3t_p10trainout/out
S=/home/ttuser/of3t_p10trainout
cd "$W" || exit 1
mkdir -p "$L" "$O" "$S/runs"
tag=$1; exact=$2; steps=$3; corpus=$4; shift 4
echo "=== $tag start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg) sha $(git rev-parse --short HEAD) card ${CARD:-3}" >> "$L/arms.log"
env TT_VISIBLE_DEVICES=${CARD:-3} TT_BIO_LEASE_CARDS=${CARD:-3} TT_BIO_LEASE_HOLDER=worker:$ROW \
    "$PY" perf/of3t_p10trainout/trainarm.py \
    --corpus "$corpus" --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --exact "$exact" --steps "$steps" \
    --out-dir "$S/runs/$tag" --curve "$S/runs/$tag.jsonl" \
    --out "$O/arm_$tag.json" "$@" > "$L/$tag.log" 2>&1
echo "=== $tag done  $(date -u +%FT%TZ) rc=$?" >> "$L/arms.log"
