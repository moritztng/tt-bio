#!/bin/bash
# phase19.sh: is the qb2 chip 0 (ASIC B9017B91B319B0DE) FPU clock-gate hang voltage-sensitive? Phase 18: FPU CG
# (CG_CTRL_EN bit 1) alone hangs pcG x1000 at 1350 MHz, all other blocks gated run clean, hysteresis does not help.
# Here: stock fw state after reset (all blocks gated, hyst 2), 1350 MHz, VDD forced to a fixed value per leg.
# Clean at +V and hang at nominal = timing margin of the gated FPU clock path on this die (silicon), not a logic bug.
# Diagnostic only, never a fix. Index: "p19_NN verdict=HANG|CLEAN vdd=<mV>".
cd ~/spd/spd-nocdeadlock
for c in 0 1; do eval "exec $((8+c))>~/spd_qb2_chip$c.lock"; flock $((8+c)); done
echo "locks 0,1 held $(date -u +%FT%TZ)"
py=~/.tenstorrent-venv/bin/python; tree=~/spd/spd-nocdeadlock/fixm
$py tel.py p19_tel.csv 0.2 & tp=$!
n=0
leg() { n=$((n+1)); tag=p19_$(printf %02d $n)
  timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "reset 0,1 rc=$? $(date -u +%FT%TZ)"; sleep 15
  echo "cg: $(TT_VISIBLE_DEVICES=0 timeout -s TERM 300 ~/triage_venv/bin/python cgctl.py read 2>&1 | tail -1)"
  [ "$1" != nominal ] && echo "vdd $1: $(timeout -s TERM 60 $py vdd.py 0 $1 2>&1 | tr "\n" " ")"
  echo "== $tag vdd=$1 start $(date -u +%FT%TZ)"
  ./repro.sh 0 $tag $tree --iters 1000 --every 50
  d=$(grep -c REPRO-DONE rp_${tag}_c0.log); V=CLEAN; [ "$d" = 1 ] || V=HANG
  echo "$tag verdict=$V vdd=$1 last: $(grep "^it " rp_${tag}_c0.log | tail -1) end $(date -u +%FT%TZ)"
}
leg 860; leg 860; leg 840; leg 880; leg nominal
timeout -s TERM 60 $py vdd.py 0 0
kill -TERM $tp; timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "P19-DONE reset rc=$? $(date -u +%FT%TZ)"
