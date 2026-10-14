#!/bin/bash
# phase16.sh: Tensix-level hang capture on qb2 chip 0 (ASIC B9017B91B319B0DE) at its default 1350 MHz, no clock cap.
# Waits for unforce_c0.sh (releases the rejected cap's FORCE_AICLK), takes the chip 0+1 flocks (a hang needs the
# board-pair reset), runs the stock fixm pcG repro twice; hangcap.sh adds tensix_diff.txt (GPRs + debug bus,
# stuck vs healthy core). tel.py logs clock/V/I/T at 5 Hz. Index phase16.log.
cd ~/spd/spd-nocdeadlock
until grep -q UNFORCE-DONE unforce_c0.log 2>/dev/null; do sleep 30; done
for c in 0 1; do eval "exec $((8+c))>~/spd_qb2_chip$c.lock"; flock $((8+c)); done
echo "locks 0,1 held $(date -u +%FT%TZ)"
py=~/.tenstorrent-venv/bin/python; tree=~/spd/spd-nocdeadlock/fixm
$py tel.py p16_tel.csv 0.2 & tp=$!
for r in a b; do tag=p16_$r
  timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "reset 0,1 rc=$? $(date -u +%FT%TZ) iommu_faults=$(dmesg 2>/dev/null | grep -c IO_PAGE_FAULT)"; sleep 15
  echo "== $tag c0 pcG x1000 start $(date -u +%FT%TZ) aiclk=$(cat /sys/class/tenstorrent/tenstorrent!0/tt_aiclk)"
  ./repro.sh 0 $tag $tree --iters 1000 --every 50
  echo "$tag c0: last: $(grep "^it " rp_${tag}_c0.log | tail -1), done=$(grep -c REPRO-DONE rp_${tag}_c0.log), timeout=$(grep -c "TIMEOUT: device timeout" rp_${tag}_c0.log) end $(date -u +%FT%TZ)"
  ls -t hang_rp_${tag}_c0_*.tensix_diff.txt 2>/dev/null | head -1
done
kill -TERM $tp; timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "P16-DONE reset rc=$? $(date -u +%FT%TZ)"
