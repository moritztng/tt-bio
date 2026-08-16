#!/usr/bin/env bash
# Run 0: re-anchor OpenDDE at 512 aa on qb2 card 2. Everything else in this task is a delta
# against this median, so it runs before any lever is measured.
# Acceptance: CIF 357c67003bb738ac, plDDT 0.754110, spread < 0.20 s. Expect 82.9-83.1 s.
set -u
source /home/ttuser/.coworker/wt/opendde-beat-dgx-h200/perf/oddedgx/env.sh
cd $WT || exit 1
echo "=== anchor, 512 aa, card 2 $(date -Is) ==="
$BL opendde-beat-dgx-h200 -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 3 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
    --label "512 aa cdk2x2, main anchor, card 2" --msa-dir $WT/.msa_oddedgx \
    --keep-cif $O/cif_anchor_512 --out $O/anchor_512_c2.json
echo "RC=$? $(date -Is)"
