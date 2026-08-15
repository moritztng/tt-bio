#!/usr/bin/env bash
# Run 3, lever A gate 2: F1-at-kt=12 against the shipped path, arms interleaved in one process
# on one device context, glue ON in both arms so F1 is the only thing moving.
# fold_ab512.py is F1's own release-gate harness: it emits trimul_tail_f1 {served, declined,
# rejects} per arm alongside fold_s / plddt / cif_sha256, which is the counter that separates a
# real A/B from an A/A with a flag that never fired.
# Acceptance, all three: digest 357c67003bb738ac and plDDT 0.754110 on EVERY fold in BOTH arms;
# served=1056 declined=160 in the f1 arm and 0/1216 in the nof1 arm; wall delta >= 0.45 s.
set -u
source /home/ttuser/.coworker/wt/opendde-beat-dgx-h200/perf/oddedgx/env.sh
cd $WT || exit 1
echo "=== opendde 512 aa, nof1/f1 x2, card 2 $(date -Is) ==="
$BL opendde-beat-dgx-h200 -- $PY -u perf/size512/fold_ab512.py --model opendde --sizes 512 \
    --arms nof1,f1,nof1,f1 --out $O/f1_ab_512_c2.json
echo "RC=$? $(date -Is)"
