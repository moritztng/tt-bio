#!/bin/bash
# Fold-level A/B for K3, the dividing k_chunk, at a size the change actually touches.
#
# 640 aa: padded 640, today k_chunk 256 (does not divide, fused kernel declines), new k_chunk 160.
# It is also the first size past Wormhole's SEQ_LEN_MORE_CHUNKING = 608, so it exercises the chunked
# path this arch takes and Blackhole does not.
#
# Arms alternate A,B,A,B and the round ends with an A/A pair, because all-A-then-all-B measured
# +13.3 % against +5.2 % interleaved on a hot card. Each process discards its own cold fold; the
# harness reports the warm median. K3 is NOT bit-exact (k_chunk sets the softmax reduction order),
# so the accuracy arm is pLDDT, which xmodel_ab.py now records for Boltz-2.
#
# Env: PY (interpreter), TREE, OUT, CARD, SIZE. Everything else is derived.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}" "${CARD:?}"
SIZE=${SIZE:-640}
REP=${REP:-2}
mkdir -p "$OUT"
cd "$TREE" || exit 1

arm() {  # label divk_value
  local label=$1 divk=$2
  echo "=== $label (TT_BIO_SDPA_DIV_K=$divk) start $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg)"
  TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 TT_BIO_SDPA_DIV_K=$divk \
  HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface} \
    "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" --size "$SIZE" \
      --repeat "$REP" --label "$label" --out "$OUT/$label.json" > "$OUT/$label.log" 2>&1
  echo "EXIT $label = $?"
  grep -h "median\|cold " "$OUT/$label.log" | tail -2
}

for r in 1 2; do
  arm "divk_A${r}_$SIZE" 0     # control: today's pick
  arm "divk_B${r}_$SIZE" 1     # K3
done
arm "divk_AA1_$SIZE" 0         # A/A floor, same config both legs
arm "divk_AA2_$SIZE" 0
echo "AB DONE $(date -u +%H:%M:%S)"
