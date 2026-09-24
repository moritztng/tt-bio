#!/bin/bash
# Pull the two upstream 1536 draws off qb2 and score them, in one command.
#
# Written 2026-09-24 12:5xZ while both draws were still computing, so the harvest pass does
# not have to re-derive any of it. It refuses a draw that has not finished refolding rather
# than scoring a partial directory: `refold_design_cif/` is written by step 4 and is the only
# thing `scrmsd.py` reads, so its presence is the completion test, not the exit code. Both
# runs will end rc=1 at the analysis step BY CONSTRUCTION (see results/upstream_512_valid.txt).
#
# Inputs are already cleared: results/upstream_1536_fixture_precleared.txt hashed qb2's
# fx_valid/bgt1536.cif against all three whglx 1536 offset-0 fixtures (ffa258bb...421e9caf)
# and diffed the two specs. Nothing here needs to re-hash them.
set -u
DEST=${1:-$HOME/mgxacc-upstream1536}
REMOTE=qb2:bgref-work
mkdir -p "$DEST"

for d in out1536_valid out1536_valid_b; do
    n=$(timeout 60 ssh -o BatchMode=yes qb2 \
        "ls ~/bgref-work/$d/intermediate_designs_inverse_folded/refold_design_cif/*.cif 2>/dev/null | wc -l")
    if [ "${n:-0}" -lt 1 ]; then
        echo "$d: NOT READY (refold_design_cif holds ${n:-0} cif) -- leave it alone"
        continue
    fi
    echo "$d: $n refold cif, pulling"
    rsync -a --info=stats0 \
        --include='intermediate_designs/***' \
        --include='intermediate_designs_inverse_folded/***' \
        --include='config/***' --include='steps.yaml' \
        --exclude='*' \
        "$REMOTE/$d/" "$DEST/$d/"
    python3 "$(dirname "$0")/../scrmsd.py" "$DEST/$d" --json \
        --model boltzgen --target perf/mgxaccuracy/targets/gpb_dimer_1646.cif \
        --target-res 1536 --side upstream-design-upstream-refold
done

echo
echo "Then read the pair with:  python3 perf/mgxaccuracy/ref/read_upstream_pair.py A B"
