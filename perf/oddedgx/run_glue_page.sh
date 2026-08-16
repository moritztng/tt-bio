#!/usr/bin/env bash
# Run 3b: the page number. Same harness, fixture, card and acceptance as run 0's anchor, with the
# host glue ON (both flags default True in the source), so this median is directly comparable to
# 82.843 rather than to the instrumented fold_ab512 wall.
# Acceptance: digest 357c67003bb738ac, plDDT 0.754110, spread < 0.20 s.
set -u
source /home/ttuser/.coworker/wt/opendde-beat-dgx-h200/perf/oddedgx/env.sh
cd $WT || exit 1
echo "=== glue ON, 512 aa, card 2 $(date -Is) ==="
$BL opendde-beat-dgx-h200 -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 3 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
    --label "512 aa cdk2x2, host glue ON, card 2" --msa-dir $WT/.msa_oddedgx \
    --keep-cif $O/cif_glue_512 --out $O/glue_512_c2.json
echo "RC=$? $(date -Is)"
