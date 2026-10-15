#!/bin/bash
# phase22.sh (after P21-DONE, board 410D back on 19.15): (1) margin of the healthy sibling chip 1 at 1350 MHz on fw 19.15
# (CG on): pcG x1000 at forced VDD 800/790/780/770, stop at the first HANG; chip 0 needs >=830 (phase 19b/20).
# (2) if fw 19.11 ran 5/5 clean on chip 0 in phase21: flash board 410D to 19.11 and soak s10 = main tree $1, chips 0+1,
# exact+fast x20 each, uncapped. Index: "p22 margin chip=1 vdd=<mV> verdict=", "p22 s10 ...", P22-DONE.
cd ~/spd/spd-nocdeadlock; tree=${1:-main_469ed1356}
until grep -q P21-DONE phase21.log; do sleep 15; done
for c in 0 1; do eval "exec $((8+c))>~/spd_qb2_chip$c.lock"; flock $((8+c)); done
echo "locks 0,1 held $(date -u +%FT%TZ)"
py=~/.tenstorrent-venv/bin/python
reset() { timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "reset 0,1 rc=$? $(date -u +%FT%TZ)"; sleep 15; }
$py tel.py p22_tel.csv 0.2 & tp=$!
reset
for v in 800 790 780 770; do tag=p22_m1_$v
  echo "vdd $v: $(timeout -s TERM 60 $py vdd.py 1 $v 2>&1 | tr "\n" " ")"
  ./repro.sh 1 $tag fixm --iters 1000 --every 50
  d=$(grep -c REPRO-DONE rp_${tag}_c1.log); V=CLEAN; [ "$d" = 1 ] || V=HANG
  echo "p22 margin chip=1 vdd=$v verdict=$V last: $(grep "^it " rp_${tag}_c1.log | tail -1) end $(date -u +%FT%TZ)"
  [ $V = HANG ] && break
done
timeout -s TERM 60 $py vdd.py 1 0 > /dev/null 2>&1; kill -TERM $tp; reset
n=$(grep -c "p21 fw=19.11.0 leg=.* verdict=CLEAN" phase21.log)
if [ "$n" = 5 ]; then
  echo "== flash 19.11.0 for s10 $(date -u +%FT%TZ)"
  (cd fw && timeout 1800 /usr/bin/python3 flash410d.py ~/fw_pack-19.11.0.fwbundle --force > flash_p22.log 2>&1); echo "flash rc=$?"
  reset
  echo "fw: $(timeout 120 $py vfq.py 0 1 2>&1 | tr "\n" " ")"
  flock -u 8; flock -u 9
  ./soak.sh s10 none 0=$tree:exact,$tree:fast 1=$tree:exact,$tree:fast > soak_s10.idx 2>&1
  wait; echo "p22 s10 $(cat soak_s10_c0.out soak_s10_c1.out | grep -E "rc=|HANG" | tr "\n" " ")"
else echo "p22 19.11 not 5/5 clean (n=$n): no s10"; fi
echo "P22-DONE $(date -u +%FT%TZ)"
