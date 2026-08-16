#!/bin/bash
# Fold-level A/B on a fixed card (Blackhole side: this task owns qb1 card 1, so there is nothing
# to pick). Generalises bh_k4_ab.sh, which was the same shape hardcoded to one env var.
#
# A discarded warm-up arm first, then A,B,A,B interleaved, then an A/A pair. The warm-up exists
# because the Wormhole 640 aa round measured its first arm 3.5 s slow with a 5.6 s internal spread
# and a 148.6 s cold fold against 89-97 s everywhere else: that arm was still paying the ttnn
# kernel cache, and discarding it removes the argument rather than having it afterwards.
#
# Env: PY TREE OUT CARD SIZE REP, plus OFF and ON as the two arms' env assignments.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}" "${CARD:?}" "${OFF:?}" "${ON:?}"
SIZE=${SIZE:-640}
REP=${REP:-2}
mkdir -p "$OUT"
cd "$TREE" || exit 1

arm() {  # label env_assignment repeat
  local label=$1 envset=$2 rep=$3
  echo "=== $label $envset rep=$rep start $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL \
      TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 $envset \
    "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" --size "$SIZE" \
      --repeat "$rep" --label "$label" --out "$OUT/$label.json" > "$OUT/$label.log" 2>&1
  echo "EXIT $label = $?"
  grep -hE "median|cold " "$OUT/$label.log" | tail -2
}

arm "w_$SIZE" "$OFF" 1
for r in 1 2; do
  arm "A${r}_$SIZE" "$OFF" $REP
  arm "B${r}_$SIZE" "$ON"  $REP
done
arm "AA1_$SIZE" "$OFF" $REP
echo "BH AB DONE $(date -u +%H:%M:%S)"
