#!/bin/bash
# Lever C's screen: is the chunked path this part takes above SEQ_LEN_MORE_CHUNKING = 608 actually
# faster than the unchunked one, in the 640-1024 aa range JapanFold serves?
#
# 1024 aa runs FIRST because it carries the kill gate. State doc 4C: a single OOM at 1024 aa ends
# this lever, and three consecutive clean folds are required to continue. REP=2 is one cold plus two
# warm, so a clean arm IS three consecutive clean folds.
#
# The unchunked arm runs before its control for the same reason: if it cannot fit, the lever is dead
# and the control is not worth a card-hour.
#
# Peak DRAM is deliberately NOT collected here. tenstorrent.dram_peak's own docstring says never A/B
# perf with TT_BIO_DRAM_PEAK set -- it drains the pipeline and measured 12.0 s against 28.8 s on one
# 117-aa fold. Capacity and timing are separate arms; this one is timing, and the capacity verdict it
# yields is the binary one the kill gate actually asks for: does it fit, or does it OOM.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}"
REP=${REP:-2}
RETRIES=${RETRIES:-60}
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh
. "$TREE/perf/whb2/wh_arm.sh"

# 1536 is Blackhole's value, i.e. "never chunk in this range". Unset is this part's scaled 608.
for S in 1024 640; do
  ARM_ENV="TT_BIO_SEQ_LEN_MORE_CHUNKING=1536" \
    run_arm "cC_unchunked_$S" "$OUT" "$RETRIES" -- --model boltz2 --size "$S" --repeat "$REP"
  if [ $? -ne 0 ]; then
    echo "KILL GATE: unchunked arm failed at $S aa -- see log for OOM"
    [ "$S" = 1024 ] && { echo "LEVER C DEAD at 1024 aa"; break; }
  fi
  run_arm "cC_chunked_$S" "$OUT" "$RETRIES" -- --model boltz2 --size "$S" --repeat "$REP"
  echo "SIZE $S DONE $(date -u +%H:%M:%S)"
done
echo "LEVER C DONE $(date -u +%H:%M:%S)"
