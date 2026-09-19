#!/bin/bash
# The l1-budget arm, the one leg of the release gate that qb2's 16:21 reboot took with it.
#
# It runs on qb1 card 0 rather than on a p300c, and that is the stronger place for it, not a
# compromise: run_l1_budget_fold forces every grid in L1_BUDGET_PARTS that FITS INSIDE the
# physical grid, so a 130-core p150a exercises the 110-core p300c budget path as well as its own
# while a 110-core part cannot force the 130-core one. The arm gates a part and an md5, not a
# number, so host load cannot move its verdict.
#
# PYTHONPATH pins the BRANCH tree: release_gate.py otherwise resolves tt_bio through the editable
# install of /home/ttuser/tt-bio-dev and scores the shared checkout
# (parity-gate-scores-installed-package-not-checkout).
set -u
WT=/home/ttuser/.coworker/wt/pvx-land
cd "$WT" || exit 1
export PYTHONPATH="$WT"
export TT_BIO_AICLK=1350
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:pvx-land
echo "=== l1-budget start $(date -u +%FT%TZ) tree $(git rev-parse --short HEAD) ==="
/home/ttuser/tt-bio-dev/env/bin/python3 -c "import tt_bio,sys;sys.stderr.write('resolved tt_bio: '+tt_bio.__file__+chr(10))" 2>&1 | grep resolved
BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-7200} BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-3600} \
  ~/.coworker/scripts/benchlock.sh pvx-land -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --model l1-budget --keep 2>&1
rc=$?
echo "=== l1-budget end $(date -u +%FT%TZ) rc=$rc ==="
exit $rc
