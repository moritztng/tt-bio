#!/bin/bash
# One training arm on card 3 of qb2. Rooted in this row's own worktree.
#   armrun.sh <tag> <on|off> <steps> <corpus> [extra args...]
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-p10trainout
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/p10trainout
O=$W/perf/of3t_p10trainout/out
S=/home/ttuser/of3t_p10trainout
cd "$W" || exit 1
mkdir -p "$L" "$O" "$S/runs"
tag=$1; exact=$2; steps=$3; corpus=$4; shift 4
echo "=== $tag start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg) sha $(git rev-parse --short HEAD)" >> "$L/arms.log"
env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-p10trainout \
    "$PY" perf/of3t_p10trainout/trainarm.py \
    --corpus "$corpus" --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --exact "$exact" --steps "$steps" \
    --out-dir "$S/runs/$tag" --curve "$S/runs/$tag.jsonl" \
    --out "$O/arm_$tag.json" "$@" > "$L/$tag.log" 2>&1
echo "=== $tag done  $(date -u +%FT%TZ) rc=$?" >> "$L/arms.log"
