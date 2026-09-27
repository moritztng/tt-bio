#!/bin/bash
# One training arm on qb2, card ${CARD:-3}, rooted in THIS row own worktree.
#   [CARD=n] armrun.sh <tag> <steps> [extra args...]
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-p10leak
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/p10leak
O=$W/perf/of3t_p10leak/out
S=/home/ttuser/of3t_p10leak
cd "$W" || exit 1
mkdir -p "$L" "$O" "$S/runs"
tag=$1; steps=$2; shift 2
echo "=== $tag start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) card ${CARD:-3}" >> "$L/arms.log"
env TT_VISIBLE_DEVICES=${CARD:-3} TT_BIO_LEASE_CARDS=${CARD:-3} TT_BIO_LEASE_HOLDER=worker:of3t-p10leak \
    "$PY" perf/of3t_p10trainout/trainarm.py \
    --corpus /home/ttuser/of3t_p10leak/corpus --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --exact off --steps "$steps" --checkpoint-every 100000 --displacement-band 0.01,100 \
    --out-dir "$S/runs/$tag" --curve "$S/runs/$tag.jsonl" \
    --out "$O/arm_$tag.json" "$@" > "$L/$tag.log" 2>&1
echo "=== $tag done  $(date -u +%FT%TZ) rc=$?" >> "$L/arms.log"
