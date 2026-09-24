#!/bin/bash
# Score the two whglx GroEL upstream CPU draws, once each has actually finished refolding.
#
# Written 2026-09-24 18:0xZ while both draws were computing, so the harvest pass does not
# re-derive it. Same completion test as ref/harvest_upstream_1536.sh and for the same reason:
# `intermediate_designs_inverse_folded/refold_design_cif/` is written by step 4 and is the only
# thing scrmsd.py reads, so its presence is the completion test and the exit code is not. Both
# runs end rc=1 at the analysis step BY CONSTRUCTION (results/upstream_512_valid.txt).
#
# Unlike the qb2 pair this needs no rsync: the draws run on whglx, where this row's device
# cells already live. Run it ON whglx from the worktree there.
#
#   ssh whglx 'cd ~/wt-mgx-design-accuracy && bash perf/mgxaccuracy/ref/harvest_groel_upstream.sh'
#
# The identity of these draws -- package pin, HF revision, weight hashes, byte-identical
# fixtures -- is verified in results/upstream_groel_whglx.txt. Nothing here re-checks it.
set -u
WORK=${1:-$HOME/bgref-work}

for size in 512 1536; do
    d="$WORK/outgroel${size}"
    rf="$d/intermediate_designs_inverse_folded/refold_design_cif"
    n=$(ls "$rf"/*.cif 2>/dev/null | wc -l)
    if [ "${n:-0}" -lt 1 ]; then
        echo "groel${size}: NOT READY (refold_design_cif holds ${n:-0} cif) -- leave it alone"
        continue
    fi
    echo "groel${size}: $n refold cif, scoring"
    python3 perf/mgxaccuracy/scrmsd.py "$d" --json \
        --model boltzgen --target perf/mgxaccuracy/targets/groel_ring4_2096.cif \
        --target-res "$size" --side upstream-design-upstream-refold
done

cat <<'NOTE'

Read against the DEVICE cells on the same fixture (results/groel_2x2_result.txt, offset 0):

    rung    device n=8 median    device range        upstream m=1
     512            0.852        0.585 -  9.694      <- this harvest
    1536           15.606       11.313 - 17.064      <- this harvest

m=1 against n=8 cannot reject at 0.05 either way: the smallest one-sided p is 1/C(9,1) = 0.111
(results/upstream_draw_count.txt). So neither value ORDERS the two sides. What a draw landing
inside the device's own range does is remove a direction, which is exactly how the banked
11.375 A at 512 on 1GPB was read. Do not upgrade "inside the range" into "no difference".

This arm is corroboration. plans/closing_rule.txt §4 is pre-registered on the qb2 1GPB pair and
stays the deciding test; GroEL does not move the verdict mapping.
NOTE
