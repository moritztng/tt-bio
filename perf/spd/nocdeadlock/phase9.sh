#!/bin/bash
# Phase 9 (2026-10-14, spd-nocdeadlock2): one chip of the (2,3) board at a time, sibling idle. Arm = chip:tag:args,
# board reset before each arm. Index phase9.log.
cd ~/spd/spd-nocdeadlock
exec 8>~/spd_qb2_chip2.lock; flock 8; exec 9>~/spd_qb2_chip3.lock; flock 9
tree=~/spd/spd-nocdeadlock/fixm
for arm in "$@"; do c=${arm%%:*}; r=${arm#*:}; tag=${r%%:*}; args=${r#*:}
  timeout 300 ~/.local/bin/tt-smi -r 2,3 > /dev/null 2>&1; echo "reset rc=$? $(date -u +%FT%TZ)"; sleep 15
  echo "== $tag c$c alone ($args) start $(date -u +%FT%TZ)"
  ./repro.sh $c $tag $tree $args
  echo "$tag c$c: $(grep -c "^it " rp_${tag}_c$c.log) prints, last: $(grep "^it " rp_${tag}_c$c.log | tail -1), done=$(grep -c REPRO-DONE rp_${tag}_c$c.log), timeout=$(grep -c "TIMEOUT: device timeout" rp_${tag}_c$c.log)"
done
timeout 300 ~/.local/bin/tt-smi -r 2,3 > /dev/null 2>&1; echo "P9-DONE $(date -u +%FT%TZ)"
