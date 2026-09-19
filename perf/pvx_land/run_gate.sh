#!/bin/bash
# Full release gate at the wk/pvx-land tip, with TT_BIO_SDPA_WIDE_K at its NEW default (on).
# PYTHONPATH pins the BRANCH tree: release_gate.py otherwise resolves tt_bio through the editable
# install of /home/ttuser/tt-bio-dev and scores the shared checkout
# (parity-gate-scores-installed-package-not-checkout, and c14 pass 43 lost a merge decision to it).
set -u
WT=/home/ttuser/.coworker/wt/pvx-land
cd "$WT" || exit 1
export PYTHONPATH="$WT"
export TT_BIO_AICLK=1350
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:pvx-land
echo "=== gate start $(date -u +%FT%TZ) tree $(git rev-parse --short HEAD) ==="
"$WT"/../../../tt-bio-dev/env/bin/python3 -c "import tt_bio, sys; sys.stderr.write('resolved tt_bio: '+tt_bio.__file__+chr(10))" 2>&1 | grep resolved
/home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --keep 2>&1
rc=$?
echo "=== gate end $(date -u +%FT%TZ) rc=$rc ==="
exit $rc
