#!/usr/bin/env bash
# of3t-denoise after the si/zij dtype fix (3f074e163). Card 0 (free, widened grant): DN64F then
# DN384F. Card 2: DN384FB once the pre-fix DN384 (devarm pid $1) has exited. DN384F vs DN384FB is
# the A/A, across two cards, so it also tests that the output does not depend on the card.
cd /home/ttuser/.coworker/wt/of3t-denoise
S=/home/ttuser/of3t_denoise
L=$S/chain_fix2.log
(
  echo "=== c0 DN64F start $(date -u +%FT%TZ)" >> $L
  CARD=0 DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN64F on /home/ttuser/of3t_fullstep64/batch_step003_t64.pt > $S/devarm_DN64F.log 2>&1
  echo "=== c0 DN64F exit $? $(date -u +%FT%TZ)" >> $L
  CARD=0 DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN384F on > $S/devarm_DN384F.log 2>&1
  echo "=== c0 DN384F exit $? $(date -u +%FT%TZ)" >> $L
) &
(
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
  echo "=== c2 pre-fix DN384 gone $(date -u +%FT%TZ)" >> $L
  CARD=2 DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN384FB on > $S/devarm_DN384FB.log 2>&1
  echo "=== c2 DN384FB exit $? $(date -u +%FT%TZ)" >> $L
) &
wait
echo "=== chain done $(date -u +%FT%TZ)" >> $L
