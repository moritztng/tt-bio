#!/bin/bash
# Lever C's capacity arm: what does the unchunked path at 1024 aa actually cost in device DRAM,
# and how much headroom is left on a ~12 GB Galaxy Wormhole chip?
#
# Separate from the timing arm on purpose. tenstorrent.dram_peak's own docstring says never A/B perf
# with TT_BIO_DRAM_PEAK set -- get_memory_view drains the pipeline and measured 12.0 s against 28.8 s
# on one 117-aa fold. So this run's walls are meaningless and only its footprint is read.
#
# --recycles 0 --steps 20 keeps it short. The peak this gate guards is the trunk's, set by the
# [1,L,L,c_z] pair tensors (268 MB each at L=1024, c_z=128), and R0S20 still runs one full trunk
# pass, so the trunk peak is unchanged while the wall drops to about a third. Stated as an
# assumption rather than hidden: if the two arms' peaks come out equal to the byte, suspect the
# probe rather than the model.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}"
SIZE=${SIZE:-1024}
RETRIES=${RETRIES:-60}
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh
. "$TREE/perf/whb2/wh_arm.sh"

cap() {  # label threshold_env
  local label=$1 thr=$2
  ARM_ENV="$thr TT_BIO_DRAM_PEAK=$OUT/$label.dram" \
    run_arm "$label" "$OUT" "$RETRIES" -- --model boltz2 --size "$SIZE" \
      --recycles 0 --steps 20 --repeat 1
  if [ -s "$OUT/$label.dram" ]; then
    echo "PEAK $label: $(sort -t: -k2 -g -r "$OUT/$label.dram" | head -1)"
    echo "PEAK_MAX_GIB $label: $(grep -oE '[0-9]+\.[0-9]+ GiB used' "$OUT/$label.dram" | sort -g -r | head -1)"
  else
    echo "PEAK $label: no samples (probe wrote nothing)"
  fi
}

cap "cap_unchunked_$SIZE" "TT_BIO_SEQ_LEN_MORE_CHUNKING=1536"
cap "cap_chunked_$SIZE"   "TT_BIO_SEQ_LEN_MORE_CHUNKING=608"
echo "CAP DONE $(date -u +%H:%M:%S)"
