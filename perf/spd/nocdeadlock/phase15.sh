#!/bin/bash
# Phase 15 (2026-10-14, spd-nocdeadlock2): AICLK margin sweep on qb2 chip 0 (board 410D, ASIC B9017B91B319B0DE), which
# hung pcG at it 1 in both phase-13 tries at 1350 while chips 1-3 ran clean. Highest clock where pcG x1000 x2 + atom
# x3000 all run clean, down from 1250; appends that cap to ~/japanfold/AICLK_CAP (restoring the phase-13 hold first).
# Caller holds no chip-0/1 flock; this takes both (pair reset). Index phase15.log, result line P15-CAP <MHz>.
cd ~/spd/spd-nocdeadlock
exec 8>~/spd_qb2_chip0.lock; flock 8; exec 9>~/spd_qb2_chip1.lock; flock 9
tree=~/spd/spd-nocdeadlock/fixm; py=~/.tenstorrent-venv/bin/python; cap=
$py tel.py p15_tel.csv 0.2 & tp=$!
for f in 1250 1200 1150 1100 1000; do clean=1
for arm in c0s${f}a:--iters\ 1000\ --every\ 50 c0s${f}b:--iters\ 1000\ --every\ 50 c0s${f}atom:--atom\ --iters\ 3000\ --every\ 100; do
  tag=${arm%%:*}; args=${arm#*:}
  timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "reset 0,1 rc=$? $(date -u +%FT%TZ) iommu_faults=$(dmesg 2>/dev/null | grep -c IO_PAGE_FAULT)"; sleep 15
  timeout 60 $py aiclk.py 0 $f
  echo "== $tag c0 forced ${f} MHz ($args) start $(date -u +%FT%TZ)"
  ./repro.sh 0 $tag $tree $args
  d=$(grep -c REPRO-DONE rp_${tag}_c0.log)
  echo "$tag c0: $(grep -c "^it " rp_${tag}_c0.log) prints, last: $(grep "^it " rp_${tag}_c0.log | tail -1), done=$d, timeout=$(grep -c "TIMEOUT: device timeout" rp_${tag}_c0.log) end $(date -u +%FT%TZ)"
  [ "$d" = 1 ] || { clean=0; break; }
done
[ $clean = 1 ] && { cap=$f; break; }; done
timeout 60 $py aiclk.py 0 0; timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "reset 0,1 rc=$? $(date -u +%FT%TZ)"
kill -TERM $tp
H=~/japanfold/AICLK_CAP.held-slot-test-20261014; [ -f $H ] && [ ! -f ~/japanfold/AICLK_CAP ] && mv $H ~/japanfold/AICLK_CAP
[ -n "$cap" ] && echo "B9017B91B319B0DE $cap qb2 chip 0 (p300c 000004613193410D): PV2 matmul deadlocks at 1350 MHz, clean at $cap (spd-nocdeadlock2 phase 15)" >> ~/japanfold/AICLK_CAP
echo "P15-CAP ${cap:-none} $(date -u +%FT%TZ)"; cat ~/japanfold/AICLK_CAP
