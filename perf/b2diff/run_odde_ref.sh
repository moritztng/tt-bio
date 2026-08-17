#!/bin/bash
# OpenDDE is the model a shared diffusion default regressed 60x once, and the only committed
# 512 aa on-arm digest for it comes from qb2 (p300, different grid), so it is not comparable to a
# qb1 p150a fold. This runs the SAME fold from a read-only reference worktree of origin/main on the
# SAME card, so the two digests differ only by this branch's edits.
REF=/tmp/b2diff-main-ref
cd /home/ttuser/.coworker/wt/boltz2-diffusion-perf || exit 1
git worktree add --detach -f $REF origin/main >/dev/null 2>&1 || echo "worktree add rc=$?"
cd $REF || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf PYTHONPATH=$REF
export BENCHLOCK_WAIT_S=5400 BENCHLOCK_LOAD_WAIT_S=900
# main's fold_ab_multi has the pinned ppc signature, so call _pair_proj_config's own default path:
# copy in only the harness fix, never any tt_bio change.
cp /home/ttuser/.coworker/wt/boltz2-diffusion-perf/perf/other512/fold_ab_multi.py perf/other512/
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-diffusion-perf -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py --model opendde \
    --sizes 512 --arms on --out /tmp/neutral_opendde_MAINREF.json
echo "RC=$?"
cp /tmp/neutral_opendde_MAINREF.json /home/ttuser/.coworker/wt/boltz2-diffusion-perf/perf/b2diff/ 2>/dev/null
echo DONEREF
