#!/usr/bin/env bash
# Run 4, lever C: the diffusion trace at 512 aa, post-8880800a4, and the default arm re-taken
# right after it as the A/A control. Predicted -0.03 to +0.10 s (state doc S4): the fix
# invalidates on conditioning identity, so every fold now recaptures (+~0.18 s) against a
# measured 0.149 s replay win. Gate: digest 357c67003bb738ac, plDDT 0.754110, delta >= 0.25 s.
# Re-pointed at THIS worktree -- the merged perf/odde4pd/run_trace.sh still names a concluded
# slug's worktree, which fleet hygiene deletes under a running job.
set -u
source /home/ttuser/.coworker/wt/opendde-beat-dgx-h200/perf/oddedgx/env.sh
cd $WT || exit 1
echo "=== T1 --trace at 512 aa $(date -Is) ==="
$BL opendde-beat-dgx-h200 -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 3 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m --trace \
    --label "512 aa cdk2x2, --trace" --msa-dir $WT/.msa_oddedgx \
    --keep-cif $O/cif_trace_512 --out $O/trace_512_c2.json
echo "T1 RC=$?"; sleep 15
echo "=== T2 default arm, A/A control $(date -Is) ==="
$BL opendde-beat-dgx-h200 -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 3 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
    --label "512 aa cdk2x2, default A/A control" --msa-dir $WT/.msa_oddedgx \
    --keep-cif $O/cif_aa_512 --out $O/aa_512_c2.json
echo "T2 RC=$? $(date -Is)"
