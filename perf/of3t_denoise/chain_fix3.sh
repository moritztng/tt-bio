#!/usr/bin/env bash
# of3t-denoise after the conditioning-mask fix (b22e9a6bf). Card 2: DN384G. Card 3 (free at
# launch, widened grant): DN64G, then DN384GB. DN384G vs DN384GB is the A/A across two cards.
cd /home/ttuser/.coworker/wt/of3t-denoise
S=/home/ttuser/of3t_denoise
L=$S/chain_fix3.log
(
  echo "=== c2 DN384G start $(date -u +%FT%TZ)" >> $L
  CARD=2 DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN384G on > $S/devarm_DN384G.log 2>&1
  echo "=== c2 DN384G exit $? $(date -u +%FT%TZ)" >> $L
) &
(
  echo "=== c3 DN64G start $(date -u +%FT%TZ)" >> $L
  CARD=3 TT_BIO_LEASE_CARDS=3 DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN64G on /home/ttuser/of3t_fullstep64/batch_step003_t64.pt > $S/devarm_DN64G.log 2>&1
  echo "=== c3 DN64G exit $? $(date -u +%FT%TZ)" >> $L
  CARD=3 DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN384GB on > $S/devarm_DN384GB.log 2>&1
  echo "=== c3 DN384GB exit $? $(date -u +%FT%TZ)" >> $L
) &
wait
echo "=== chain done $(date -u +%FT%TZ)" >> $L
