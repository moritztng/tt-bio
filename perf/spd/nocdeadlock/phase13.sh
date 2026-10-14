#!/bin/bash
# Phase 13 (2026-10-14, spd-nocdeadlock2): slot test after Moritz's swap (board 410D -> slot 1, 4103 -> slot 2, 1005 out).
# Inventory (sysfs tt_* per node + lspci), AICLK_CAP set aside (the listed chip left qb2), then each chip ALONE at its
# default clock: pcG x1000 x2 (fixm forced grid, hung new chip 3 at it 1), then 20-fold exact c730 fixm soak on all four
# in parallel. tel.py samples clock/voltage/current/temp of all chips throughout. Index phase13.log.
cd ~/spd/spd-nocdeadlock
for c in 0 1 2 3; do eval "exec $((8+c))>~/spd_qb2_chip$c.lock"; flock $((8+c)); done
echo "locks held $(date -u +%FT%TZ)"
for d in /sys/class/tenstorrent/tenstorrent!*; do echo "== $d $(readlink -f $d/device | xargs basename)"
  for a in $d/tt_*; do printf '  %s=%s\n' ${a##*/} "$(head -c 200 $a 2>/dev/null | tr '\n' ' ')"; done; done > p13_inventory.txt
lspci -d 1e52: >> p13_inventory.txt; cat p13_inventory.txt
[ -f ~/japanfold/AICLK_CAP ] && { echo "AICLK_CAP set aside: $(cat ~/japanfold/AICLK_CAP)"; mv ~/japanfold/AICLK_CAP ~/japanfold/AICLK_CAP.held-slot-test-20261014; }
py=~/.tenstorrent-venv/bin/python; tree=~/spd/spd-nocdeadlock/fixm
$py tel.py p13_tel.csv 0.2 & tp=$!
for c in 0 1 2 3; do pair=0,1; [ $c -ge 2 ] && pair=2,3
  for r in a b; do tag=slot_${c}$r
    timeout 300 ~/.local/bin/tt-smi -r $pair > /dev/null 2>&1; echo "reset $pair rc=$? $(date -u +%FT%TZ) iommu_faults=$(dmesg 2>/dev/null | grep -c IO_PAGE_FAULT)"; sleep 15
    echo "== $tag c$c alone pcG x1000 start $(date -u +%FT%TZ)"
    ./repro.sh $c $tag $tree --iters 1000 --every 50
    echo "$tag c$c: $(grep -c "^it " rp_${tag}_c$c.log) prints, last: $(grep "^it " rp_${tag}_c$c.log | tail -1), done=$(grep -c REPRO-DONE rp_${tag}_c$c.log), timeout=$(grep -c "TIMEOUT: device timeout" rp_${tag}_c$c.log) end $(date -u +%FT%TZ)"
  done; done
timeout 300 ~/.local/bin/tt-smi -r 0,1,2,3 > /dev/null 2>&1; echo "reset all rc=$? $(date -u +%FT%TZ)"; sleep 15
for c in 0 1 2 3; do flock -u $((8+c)); done   # soak.sh takes the four flocks itself
./soak.sh s7 none 0=fixm:exact 1=fixm:exact 2=fixm:exact 3=fixm:exact
kill -TERM $tp; timeout 300 ~/.local/bin/tt-smi -r 0,1,2,3 > /dev/null 2>&1; echo "P13-DONE reset rc=$? $(date -u +%FT%TZ)"
