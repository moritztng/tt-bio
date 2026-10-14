#!/bin/bash
# ctl_qb1.sh <card>: hardware discriminator on a qb1 p150a: fixm forced-grid repro (pcG x1000 x3, atom x3000) under the
# card flock (3 h cap). Op timeout 120 s ends a hung leg; a hang resets that card. Index ctl_qb1_c<card>.log.
c=$1; cd ~/spd/spd-nocdeadlock2; tree=~/spd/spd-nocdeadlock2/fixm; source ~/tt-bio-dev/env/bin/activate
exec 8>~/spd_qb1_card$c.lock; flock -w 10800 8 || { echo "NO-FLOCK $(date -u +%FT%TZ)"; exit 1; }; echo "got card$c $(date -u +%FT%TZ)"
for arm in q1:--iters\ 1000\ --every\ 50 q2:--iters\ 1000\ --every\ 50 q3:--iters\ 1000\ --every\ 50 qa:--atom\ --iters\ 3000\ --every\ 100; do
  tag=${arm%%:*}; args=${arm#*:}; l=~/spd/spd-nocdeadlock2/rp_${tag}_c$c.log; echo "== $tag ($args) start $(date -u +%FT%TZ)"
  (cd $tree && env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$c TT_METAL_OPERATION_TIMEOUT_SECONDS=120 timeout -s TERM 1800 python ~/spd/spd-nocdeadlock2/paircond_repro.py $args > $l 2>&1)
  to=$(grep -c "TIMEOUT: device timeout" $l)
  echo "$tag c$c rc=$?: $(grep -c "^it " $l) prints, last: $(grep "^it " $l | tail -1), done=$(grep -c REPRO-DONE $l), timeout=$to"
  if [ "$to" != 0 ] || ! grep -q REPRO-DONE $l; then timeout 300 ~/.local/bin/tt-smi -r $c >/dev/null 2>&1; echo "HANG/FAIL; reset rc=$? $(date -u +%FT%TZ)"; break; fi
done
echo "CONTROL-DONE c$c $(date -u +%FT%TZ)"
