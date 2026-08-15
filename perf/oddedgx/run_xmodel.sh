#!/usr/bin/env bash
# Run 5b: cross-model. `_generate_relp` lives in protenix.py and protenix.py:1924 calls it, so the
# scatter is shared with every Protenix-derived model, not just opendde. It is torch.equal against
# the shipped function, so the digest must not move -- and a shared default flipped for one model
# is exactly the class of change `tt-bio-shared-diffusion-global-env-default-regression` records.
# Acceptance: protenix-v2 CIF digest and plDDT identical in both arms.
set -u
source /home/ttuser/.coworker/wt/opendde-beat-dgx-h200/perf/oddedgx/env.sh
cd $WT || exit 1
echo "=== protenix-v2 512 aa, noglue vs glue, card 2 $(date -Is) ==="
$BL opendde-beat-dgx-h200 -- $PY -u perf/size512/fold_ab512.py --model protenix-v2 \
    --sizes 512 --arms noglue,glue --out $O/xmodel_px_c2.json
echo "RC=$? $(date -Is)"
