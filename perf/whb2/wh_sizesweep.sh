#!/bin/bash
# The brief's mandated size sweep, on the winning configuration.
#
# Arm A reproduces pre-campaign behaviour: K3 off and the chunking threshold back at this part's
# old 608. Arm B is the shipped default of this branch: K3 on, threshold re-fit to 1088.
#
# Of the nine sizes the brief lists, only three can move. K3 changes padded 448/576/640/896/960 and
# of those only 640 is on the list; lever C changes everything between the old 608 and the new 1088,
# which on the list is 640, 768 and 1024. The other six pad to 128/256/320/320/384/512, where both
# levers return what they returned before by construction. So this sweep is a neutrality check at six
# sizes and a win table at three, and both halves are the point.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}"
REP=${REP:-1}
RETRIES=${RETRIES:-60}
SIZES=${SIZES:-"128 256 298 320 384 512 640 768 1024"}
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh
. "$TREE/perf/whb2/wh_arm.sh"

PRE="TT_BIO_SDPA_DIV_K=0 TT_BIO_SEQ_LEN_MORE_CHUNKING=608"

for S in $SIZES; do
  ARM_ENV="$PRE" run_arm "sw_pre_$S"  "$OUT" "$RETRIES" -- --model boltz2 --size "$S" --repeat "$REP"
  ARM_ENV=""    run_arm "sw_post_$S" "$OUT" "$RETRIES" -- --model boltz2 --size "$S" --repeat "$REP"
  echo "SWEEP SIZE $S DONE $(date -u +%H:%M:%S)"
done
echo "SIZESWEEP DONE $(date -u +%H:%M:%S)"
