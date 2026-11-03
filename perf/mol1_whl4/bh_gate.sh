#!/bin/bash
# mol1-whl4 BH merge-gate legs on qb2 card 3 (booked in ~/spd/feed/windows.tsv as worker:mol1-whl4): protenix-v2, opendde,
# mol1 at 512, base 974d7c9ee (main merge-base) vs fix bfb187307, one chip, one process per arm, order base, fix, base2;
# cold + 2 warm (base2 cold + 1). AICLK of every node every 0.5 s in OUT/<model>/aiclk.tsv, the opened node in nodes.txt,
# so perf/mol1_whl4/gate_read.py reads it as it reads the WH cells.
#   setsid nohup bash bh_gate.sh CARD > /dev/null 2>&1 &
set -u
C=${1:-3} R=~/spd/mol1-whl4 PY=/home/ttuser/tt-bio-dev/env/bin/python
B=$R/tree-974d7c9ee F=$R/tree-bfb187307 O=$R/out/bh-gate
mkdir -p $O
say() { echo "$(date -u +%FT%TZ) $*" >> $O/run.log; }
exec 8> ~/spd_qb2_card$C.lock 9> ~/spd_qb2_chip$C.lock
say "waiting for card $C flocks"
flock -w 36000 8 && flock -w 36000 9 || { say "card $C flock busy"; exit 3; }
say "card $C taken"
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_HOLDER=mol1-whl4
unset TT_BIO_LEVERS TT_BIO_DRAM_PEAK
for m in protenix-v2 opendde mol1; do
  o=$O/$m; mkdir -p $o
  ( while kill -0 $$ 2>/dev/null; do l=$(cut -d' ' -f1 /proc/loadavg); for n in /sys/class/tenstorrent/tenstorrent\!*; do
      printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done; sleep 0.5; done ) 8>&- 9>&- >> $o/aiclk.tsv 2>/dev/null &
  S=$!
  for arm in base:$B:2 fix:$F:2 base2:$B:1; do
    IFS=: read a t n <<< "$arm"; mkdir -p $o/$a
    ( cd $t && PYTHONPATH=$t timeout -s INT 3000 $PY $F/perf/of3_4xpd/xmodel_ab.py --model $m --tree $t --size 512 \
        --repeat $n --label $a --out $o/$a/result.json ) > $o/$a/harness.log 2>&1 < /dev/null 8>&- 9>&- &
    H=$!
    while kill -0 $H 2>/dev/null; do sleep 5
      [ -s $o/nodes.txt ] || for p in $(pstree -p $H | grep -o "([0-9]*)" | tr -d "()"); do
        ls -l /proc/$p/fd 2>/dev/null | grep -o 'tenstorrent/[0-9]*'; done | sort -u > $o/nodes.txt
    done
    wait $H; say "$m $a rc=$?"
  done
  kill $S
done
say "done"
