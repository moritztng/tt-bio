#!/bin/bash
# Phase 10 (2026-10-14, spd-nocdeadlock2): timing-margin test. chip 3 alone with FORCE_AICLK <MHz>, pcG x1000 x2 +
# atom x3000 (at 1350 pcG hangs at it 1). Then release + board reset. Index phase10.log.
cd ~/spd/spd-nocdeadlock; f=${1:-1000}
exec 8>~/spd_qb2_chip2.lock; flock 8; exec 9>~/spd_qb2_chip3.lock; flock 9
tree=~/spd/spd-nocdeadlock/fixm; py=~/.tenstorrent-venv/bin/python
for arm in clk${f}a:--iters\ 1000\ --every\ 50 clk${f}b:--iters\ 1000\ --every\ 50 clk${f}atom:--atom\ --iters\ 3000\ --every\ 100; do
  tag=${arm%%:*}; args=${arm#*:}
  timeout 300 ~/.local/bin/tt-smi -r 2,3 > /dev/null 2>&1; echo "reset rc=$? $(date -u +%FT%TZ)"; sleep 15
  timeout 60 $py aiclk.py 3 $f
  echo "== $tag c3 forced ${f} MHz ($args) start $(date -u +%FT%TZ)"
  ./repro.sh 3 $tag $tree $args
  echo "$tag c3: $(grep -c "^it " rp_${tag}_c3.log) prints, last: $(grep "^it " rp_${tag}_c3.log | tail -1), done=$(grep -c REPRO-DONE rp_${tag}_c3.log), timeout=$(grep -c "TIMEOUT: device timeout" rp_${tag}_c3.log)"
done
timeout 60 $py aiclk.py 3 0; timeout 300 ~/.local/bin/tt-smi -r 2,3 > /dev/null 2>&1; echo "P10-DONE reset rc=$? $(date -u +%FT%TZ)"
