#!/bin/bash
# phase20.sh: qb2 chip 0 (ASIC B9017B91B319B0DE) on fw 19.15 (CG on, stock) at 1350 MHz with VDD forced to 832 mV, the
# value the pre-19.12 legacy p300c V/F curve gives at 1350 (807.6 + fw-table voltage_margin 25). The process-based curve
# that 19.12 enabled for p300c (tt-system-firmware 3738f7b2) asks 809 mV for this die. Diagnostic only, never a fix.
# Chip 0 alone, no reset (chip 1 is busy): stop at the first HANG. Index: "p20_NN verdict=HANG|CLEAN vdd=<mV>".
cd ~/spd/spd-nocdeadlock
exec 8>~/spd_qb2_chip0.lock; flock 8; echo "lock 0 held $(date -u +%FT%TZ)"
py=~/.tenstorrent-venv/bin/python; tree=~/spd/spd-nocdeadlock/fixm
$py tel.py p20_tel.csv 0.2 & tp=$!
echo "cg: $(TT_VISIBLE_DEVICES=0 timeout -s TERM 300 ~/triage_venv/bin/python cgctl.py read 2>&1 | tail -1)"
echo "vdd 832: $(timeout -s TERM 60 $py vdd.py 0 832 2>&1 | tr "\n" " ")"
for n in 1 2 3 4 5; do tag=p20_0$n
  echo "== $tag vdd=832 start $(date -u +%FT%TZ)"
  ./repro.sh 0 $tag $tree --iters 1000 --every 50
  d=$(grep -c REPRO-DONE rp_${tag}_c0.log); V=CLEAN; [ "$d" = 1 ] || V=HANG
  echo "$tag verdict=$V vdd=832 last: $(grep "^it " rp_${tag}_c0.log | tail -1) end $(date -u +%FT%TZ)"
  [ $V = HANG ] && break
done
[ $V = CLEAN ] && timeout -s TERM 60 $py vdd.py 0 0
kill -TERM $tp; echo "P20-DONE last=$V $(date -u +%FT%TZ)"
