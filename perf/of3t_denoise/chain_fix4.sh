#!/usr/bin/env bash
# of3t-denoise after the diffusion ref-embedder fix (training path runs the encoder's
# RefAtomFeatureEmbedder on the card). Card 1 (free at launch, no row dispatched to it): DN384R.
# Card 3, once chain_fix3's DN384GB has released it: DN384RB. DN384R vs DN384RB = A/A across cards.
cd /home/ttuser/.coworker/wt/of3t-denoise
S=/home/ttuser/of3t_denoise
L=$S/chain_fix4.log
(
  echo "=== c1 DN384R start $(date -u +%FT%TZ)" >> $L
  CARD=1 DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN384R on > $S/devarm_DN384R.log 2>&1
  echo "=== c1 DN384R exit $? $(date -u +%FT%TZ)" >> $L
) &
(
  while pgrep -f "grad_DN384GB.pt" > /dev/null; do sleep 30; done
  echo "=== c3 DN384RB start $(date -u +%FT%TZ)" >> $L
  CARD=3 DEVSTEP_EXTRA=--denoise bash perf/of3t_denoise/devarm.sh DN384RB on > $S/devarm_DN384RB.log 2>&1
  echo "=== c3 DN384RB exit $? $(date -u +%FT%TZ)" >> $L
) &
wait
echo "=== chain done $(date -u +%FT%TZ)" >> $L
