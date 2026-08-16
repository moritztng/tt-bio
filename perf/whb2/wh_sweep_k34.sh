#!/bin/bash
# WH size sweep: K3 at every size it actually changes and has a fixture for, K4 at its band.
#
# K3 changes padded 448, 576, 640, 896, 960. Fixtures exist at 448, 576 and 640 only, and 640 is
# already measured (state doc 13.4), so 448 and 576 are the whole remaining reachable set.
# K4 changes padded 320 and 384; 384 is the larger and cheaper of the two to fold.
#
# Each size runs a throwaway warm-up arm first. The 640 round measured its first arm 3.5 s slow with
# a 5.6 s internal spread and a 148.6 s cold fold against 89-97 s everywhere else -- that arm was
# still paying the ttnn kernel cache. A discarded warm-up arm per size removes that from the result
# instead of arguing about it afterwards. Then A,B,A,B interleaved, so each size carries its own
# two-control spread as its floor.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}" "${CARD:?}"
REP=${REP:-2}
mkdir -p "$OUT"
cd "$TREE" || exit 1

arm() {  # label size env_assignment repeat
  local label=$1 size=$2 envset=$3 rep=$4
  echo "=== $label size=$size $envset rep=$rep start $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL \
      TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 $envset \
    "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" --size "$size" \
      --repeat "$rep" --label "$label" --out "$OUT/$label.json" > "$OUT/$label.log" 2>&1
  echo "EXIT $label = $?"
  grep -hE "median|cold " "$OUT/$label.log" | tail -2
}

# K4 at 384: the band's k, 64 against the dividing 192. K3 is irrelevant here (the band returns
# before _dividing_sdpa_chunk_size is consulted), so it stays at its default.
sweep_k4() {
  local S=384
  arm "w_k4_$S"  $S "TT_BIO_SDPA_BAND_DIV_K=0" 1
  for r in 1 2; do
    arm "k4_A${r}_$S" $S "TT_BIO_SDPA_BAND_DIV_K=0" $REP
    arm "k4_B${r}_$S" $S "TT_BIO_SDPA_BAND_DIV_K=1" $REP
  done
}

# K3 at the two remaining sizes it changes. K4 off throughout so only one lever moves.
sweep_k3() {
  local S=$1
  arm "w_k3_$S"  $S "TT_BIO_SDPA_DIV_K=0 TT_BIO_SDPA_BAND_DIV_K=0" 1
  for r in 1 2; do
    arm "k3_A${r}_$S" $S "TT_BIO_SDPA_DIV_K=0 TT_BIO_SDPA_BAND_DIV_K=0" $REP
    arm "k3_B${r}_$S" $S "TT_BIO_SDPA_DIV_K=1 TT_BIO_SDPA_BAND_DIV_K=0" $REP
  done
}

sweep_k4
sweep_k3 448
sweep_k3 576
echo "SWEEP DONE $(date -u +%H:%M:%S)"
