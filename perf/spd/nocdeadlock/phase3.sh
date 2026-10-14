#!/bin/bash
# Phase 3 (2026-10-14, spd-nocdeadlock2): paircond_repro.py arms on qb2 chips 2+3, both at once, one board reset
# before each arm. Arm = tag:args. Index phase3.log; per-leg logs rp_<tag>_c<N>.log; hang dumps hang_rp_<tag>_c<N>_*.
cd ~/spd/spd-nocdeadlock
exec 8>~/spd_qb2_chip2.lock; flock 8; exec 9>~/spd_qb2_chip3.lock; flock 9
tree=~/spd/spd-nocdeadlock/fix2d
for arm in "$@"; do tag=${arm%%:*}; args=${arm#*:}
  timeout 300 ~/.local/bin/tt-smi -r 2,3 > /dev/null 2>&1; echo "reset rc=$? $(date -u +%FT%TZ)"; sleep 15
  echo "== $tag ($args) start $(date -u +%FT%TZ)"
  for c in 2 3; do ./repro.sh $c $tag $tree $args & done; wait
  for c in 2 3; do echo "$tag c$c: $(grep -c '^it ' rp_${tag}_c$c.log) prints, last: $(grep '^it ' rp_${tag}_c$c.log | tail -1), done=$(grep -c REPRO-DONE rp_${tag}_c$c.log), timeout=$(grep -c 'TIMEOUT: device timeout' rp_${tag}_c$c.log)"; done
done
echo "P3-DONE $(date -u +%FT%TZ)"
