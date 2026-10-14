#!/bin/bash
# Phase 11 (2026-10-14, spd-nocdeadlock2): AICLK margin sweep on chip 3 alone (chip 2 idle). Per clock: pcG x1000 x2 +
# atom x3000, forced via aiclk.py; 1350 forced = control that forcing itself does not hide the hang. Index phase11.log.
cd ~/spd/spd-nocdeadlock
exec 8>~/spd_qb2_chip2.lock; flock 8; exec 9>~/spd_qb2_chip3.lock; flock 9
tree=~/spd/spd-nocdeadlock/fixm; py=~/.tenstorrent-venv/bin/python
for f in 1350 1250 1200 1150; do
for arm in s${f}a:--iters\ 1000\ --every\ 50 s${f}b:--iters\ 1000\ --every\ 50 s${f}atom:--atom\ --iters\ 3000\ --every\ 100; do
  tag=${arm%%:*}; args=${arm#*:}
  timeout 300 ~/.local/bin/tt-smi -r 2,3 > /dev/null 2>&1; echo "reset rc=$? $(date -u +%FT%TZ)"; sleep 15
  timeout 60 $py aiclk.py 3 $f
  echo "== $tag c3 forced ${f} MHz ($args) start $(date -u +%FT%TZ)"
  ./repro.sh 3 $tag $tree $args
  echo "$tag c3: $(grep -c "^it " rp_${tag}_c3.log) prints, last: $(grep "^it " rp_${tag}_c3.log | tail -1), done=$(grep -c REPRO-DONE rp_${tag}_c3.log), timeout=$(grep -c "TIMEOUT: device timeout" rp_${tag}_c3.log)"
done; done
timeout 60 $py aiclk.py 3 0; timeout 300 ~/.local/bin/tt-smi -r 2,3 > /dev/null 2>&1; echo "P11-DONE reset rc=$? $(date -u +%FT%TZ)"
