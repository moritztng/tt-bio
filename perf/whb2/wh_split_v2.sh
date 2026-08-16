#!/bin/bash
# Boltz-2 stage split on the Wormhole Galaxy, by ablation at fold level. Deliverable spine.
#
# Four arms per size. base - R0 = the recycles, base - S20 = 180 diffusion steps, R0S20 = the
# residue (embedder + one trunk pass + 20 steps + confidence + host). Nothing sits between the
# clock and the work: the two loop counts are tt_baseline module globals and xmodel_ab.py's
# --recycles/--steps set them after _resolve_*, so an unset flag reproduces the shipped value.
#
# baseAA is a second identical base arm: the A/A floor every delta here is judged against.
#
# v2 picks a card PER ARM with retry instead of one card for the block. v1 took one card at the
# front of the lock queue and starved; on a box with four other workers a chain cannot assume it
# still owns the card it picked. The arms can therefore straddle chips, so each arm records the
# card it ran on. That is acceptable here and it is measured, not assumed: the 576 aa round put
# four control arms on three different cards and they spanned 0.882 s, 1.27 %, while the smallest
# delta this split resolves (the R0S20 residue) is tens of seconds.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}"
SIZE=${SIZE:-512}
REP=${REP:-2}
RETRIES=${RETRIES:-60}
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh

arm() {  # label recycles steps repeat
  local label=$1 rec=$2 st=$3 rep=$4 try C
  for try in $(seq 1 "$RETRIES"); do
    C=$(pick_card) || { sleep 30; continue; }
    echo "=== $label rec=$rec steps=$st rep=$rep try=$try card=$C $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg)"
    env TT_VISIBLE_DEVICES=$C TT_METAL_LOGGER_LEVEL=FATAL \
        TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 \
      "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" --size "$SIZE" \
        --recycles "$rec" --steps "$st" --repeat "$rep" --label "$label" \
        --out "$OUT/$label.json" > "$OUT/$label.log" 2>&1
    if [ $? -eq 0 ]; then
      echo "EXIT $label = 0 (card $C)"
      grep -hE "median|cold " "$OUT/$label.log" | tail -2
      return 0
    fi
    grep -q DeviceInUseError "$OUT/$label.log" || { echo "EXIT $label = FAILED"; tail -3 "$OUT/$label.log"; return 1; }
    sleep 15
  done
  echo "EXIT $label = GAVE UP"
  return 1
}

arm "split_base_$SIZE"   3 200 $REP
arm "split_R0_$SIZE"     0 200 $REP
arm "split_S20_$SIZE"    3  20 $REP
arm "split_R0S20_$SIZE"  0  20 $REP
arm "split_baseAA_$SIZE" 3 200 $REP
echo "SPLIT $SIZE DONE $(date -u +%H:%M:%S)"
