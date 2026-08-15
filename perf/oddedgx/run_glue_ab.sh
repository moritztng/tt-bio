#!/usr/bin/env bash
# Run 3': lever B at the fold. Lever A is dead on parity (state E3.3), so the A/B that decides
# this task is the host glue against the shipped fp32 path, arms interleaved in one process on
# one device context so nothing but the two host flags moves.
# Acceptance, all three: digest 357c67003bb738ac and plDDT 0.754110 on EVERY fold in BOTH arms;
# host_glue.relp_served = 1 in the glue arm and relp_legacy = 1 in the noglue arm, or the A/B
# measured nothing; wall delta quoted against the run-0 anchor 82.843 with the session A/A floor.
set -u
source /home/ttuser/.coworker/wt/opendde-beat-dgx-h200/perf/oddedgx/env.sh
cd $WT || exit 1
echo "=== opendde 512 aa, noglue/glue x2, card 2 $(date -Is) ==="
$BL opendde-beat-dgx-h200 -- $PY -u perf/size512/fold_ab512.py --model opendde --sizes 512 \
    --arms noglue,glue,noglue,glue --out $O/glue_ab_512_c2.json
echo "RC=$? $(date -Is)"
