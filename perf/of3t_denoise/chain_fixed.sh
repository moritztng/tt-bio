#!/usr/bin/env bash
# of3t-denoise qb2 card 2: guard control on the pre-fix tree, then the fixed tree at 64 and twice
# at 384 (the second is the A/A). The adapter default is now denoise on; --denoise is passed too
# because trainfwd_run.py sets fwd.denoise from its own flag.
cd /home/ttuser/.coworker/wt/of3t-denoise
S=/home/ttuser/of3t_denoise
source /home/ttuser/tt-bio-dev/env/bin/activate
echo "=== guard start $(date -u +%FT%TZ)" >> $S/chain_fixed.log
TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:of3t-denoise PYTHONPATH=$PWD \
  timeout 1800 python3 perf/of3t_denoise/guard_control.py --batch /home/ttuser/of3t_fullstep64/batch_step003_t64.pt \
  --out perf/of3t_denoise/GUARD_CONTROL.json > $S/guard.log 2>&1
echo "=== guard exit $? $(date -u +%FT%TZ)" >> $S/chain_fixed.log
DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN64 on /home/ttuser/of3t_fullstep64/batch_step003_t64.pt > $S/devarm_DN64.log 2>&1
echo "=== DN64 exit $? $(date -u +%FT%TZ)" >> $S/chain_fixed.log
DEVSTEP_EXTRA="--denoise --weights-out $S/weights_walked.pt" bash perf/of3t_denoise/devarm.sh DN384 on > $S/devarm_DN384.log 2>&1
echo "=== DN384 exit $? $(date -u +%FT%TZ)" >> $S/chain_fixed.log
DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN384B on > $S/devarm_DN384B.log 2>&1
echo "=== DN384B exit $? $(date -u +%FT%TZ)" >> $S/chain_fixed.log
echo "=== chain done $(date -u +%FT%TZ)" >> $S/chain_fixed.log
