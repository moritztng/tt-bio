#!/bin/bash
# Lever D: what does diffusion-sample multiplicity cost on a 12 GB Wormhole part?
#
# Section 4D. JapanFold offers max_diffusion_samples 5 and every wall on this document is B=1. The
# multiplicity batching worth 3.56x at B=4 on Blackhole has never been run on a chip with a third of
# the DRAM. This is throughput, not latency: it changes what the service can offer, not what the
# single-fold wall says.
#
# Two passes, because they cannot share a run. Timing has TT_BIO_DRAM_PEAK unset (the probe drains
# the pipeline and measures itself). Capacity sets it, at B=5 only, which is the worst case the
# service can be asked for, and its walls are discarded.
#
# The number that matters is per-sample cost, wall/B, against B=1. If it does not fall, batching
# buys the service nothing here whatever it buys on Blackhole. If a size OOMs, say which B and stop
# climbing at that size rather than reporting a partial ladder as if it were a ceiling.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}"
REP=${REP:-1}
RETRIES=${RETRIES:-60}
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh
. "$TREE/perf/whb2/wh_arm.sh"

for S in 298 512; do
  for B in 1 2 4 5; do
    ARM_ENV="" run_arm "d_B${B}_$S" "$OUT" "$RETRIES" -- \
      --model boltz2 --size "$S" --samples "$B" --repeat "$REP"
    if [ $? -ne 0 ]; then
      echo "LEVER D: $S aa stops at B=$B (see log); not climbing further at this size"
      break
    fi
  done
  echo "LEVER D SIZE $S DONE $(date -u +%H:%M:%S)"
done

# Capacity at the worst case the service can ask for. Walls here are meaningless by construction.
for S in 298 512; do
  ARM_ENV="TT_BIO_DRAM_PEAK=$OUT/dcap_B5_$S.dram" run_arm "dcap_B5_$S" "$OUT" "$RETRIES" -- \
    --model boltz2 --size "$S" --samples 5 --recycles 0 --steps 20 --repeat 1
  if [ -s "$OUT/dcap_B5_$S.dram" ]; then
    echo "PEAK_MAX_GIB dcap_B5_$S: $(grep -oE '[0-9]+\.[0-9]+ GiB used' "$OUT/dcap_B5_$S.dram" | sort -g -r | head -1)"
  fi
done
echo "LEVER D DONE $(date -u +%H:%M:%S)"
