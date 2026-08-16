#!/bin/bash
# WH size sweep, second attempt. The first picked one card at launch and lost it: another fleet
# worker grabbed UMD 28 between arms and three of five arms at 384 aa died on the device lease
# (correctly -- the lease refused rather than colliding at the fd level). The card has to be picked
# per ARM, not per chain, and an arm that loses the race has to retry rather than be dropped.
#
# Sizes are ordered by predicted effect, largest first, so a truncated run still carries the result
# that matters: 576 (predicted 3.79-11.86 s), then 448 (0.89-2.78 s), then K4 at 384 (0.37-1.16 s).
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}"
REP=${REP:-2}
RETRIES=${RETRIES:-4}
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh

arm() {  # label size env_assignment repeat
  local label=$1 size=$2 envset=$3 rep=$4 try C
  for try in $(seq 1 "$RETRIES"); do
    C=$(pick_card) || { echo "$label: no free card, try $try"; sleep 30; continue; }
    echo "=== $label size=$size $envset rep=$rep try=$try card=$C $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg)"
    env TT_VISIBLE_DEVICES=$C TT_METAL_LOGGER_LEVEL=FATAL \
        TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 $envset \
      "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" --size "$size" \
        --repeat "$rep" --label "$label" --out "$OUT/$label.json" > "$OUT/$label.log" 2>&1
    if [ $? -eq 0 ]; then
      echo "EXIT $label = 0 (card $C)"
      grep -hE "median|cold " "$OUT/$label.log" | tail -2
      return 0
    fi
    if grep -q DeviceInUseError "$OUT/$label.log"; then
      echo "$label: lost card $C to another worker, retrying"
      sleep 15
    else
      echo "EXIT $label = FAILED, not a lease error"
      tail -3 "$OUT/$label.log"
      return 1
    fi
  done
  echo "EXIT $label = GAVE UP after $RETRIES tries"
  return 1
}

k3() {  # size
  local S=$1
  arm "w_k3_$S" $S "TT_BIO_SDPA_DIV_K=0 TT_BIO_SDPA_BAND_DIV_K=0" 1
  for r in 1 2; do
    arm "k3_A${r}_$S" $S "TT_BIO_SDPA_DIV_K=0 TT_BIO_SDPA_BAND_DIV_K=0" $REP
    arm "k3_B${r}_$S" $S "TT_BIO_SDPA_DIV_K=1 TT_BIO_SDPA_BAND_DIV_K=0" $REP
  done
  echo "SIZE $S DONE $(date -u +%H:%M:%S)"
}

k4() {  # size
  local S=$1
  for r in 1 2; do
    arm "k4_A${r}_$S" $S "TT_BIO_SDPA_BAND_DIV_K=0" $REP
    arm "k4_B${r}_$S" $S "TT_BIO_SDPA_BAND_DIV_K=1" $REP
  done
  echo "SIZE $S K4 DONE $(date -u +%H:%M:%S)"
}

k3 576
k3 448
k4 384
echo "SWEEP DONE $(date -u +%H:%M:%S)"
