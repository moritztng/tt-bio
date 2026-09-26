#!/bin/bash
# Does the composed stack accept and reject for the reasons all-off does?
#
# A full trajectory is 125 gradient rounds plus 15 mutate steps, so a paired run to a decision is
# about two hours of card time. The stage budget is settings-driven, so shrinking every stage
# gives both arms the WHOLE path -- gradient stages, mutate, MPNN, validation, filters, the
# accept/reject decision and the ranked CSV -- at 9 gradient rounds instead of 125. The design
# that comes out is not a good one; the question is whether the two arms decide the same way
# about the same fixture, which a short trajectory answers as well as a long one.
#
# Both arms on card 0, serially, never one per card: a decision that differed would then be
# confounded with which card ran it.
set -euo pipefail
cd /home/ttuser/.coworker/wt/bcx-p10-stack
log=perf/bcx_p10_stack/out/accept.log
SHORT="--set screen_steps=4 --set refine_steps=2 --set anneal_steps=2 --set harden_steps=1 --set mutate_steps=1"
for spec in dec_off:0:0:0 dec_on:1:1:1; do
    IFS=: read -r name e t r <<< "$spec"
    echo "=== $(date -u +%FT%TZ) $name ===" >> "$log"
    # --rounds 999 so the round meter never trips StopAfterRounds and the campaign finishes
    bash perf/bcx_p10_stack/arm.sh "$name" 999 "$e" "$t" "$r" $SHORT \
        > "perf/bcx_p10_stack/out/$name.log" 2>&1 || echo "ARM $name exited $?" >> "$log"
    grep -aE 'accepted|rejected|Ranked|summary' "perf/bcx_p10_stack/out/$name.log" | tail -5 >> "$log" || true
done
echo "=== $(date -u +%FT%TZ) accept done ===" >> "$log"
