#!/bin/bash
# control.sh <chip> <sibling> (2026-10-14, spd-nocdeadlock2 hardware discriminator): the same fixm forced-grid repro
# arms that hang chips 2/3, on a chip of the OTHER qb2 board, under that chip only flock (no reset before; a hang
# takes the sibling flock too and resets the pair). Index control_c<chip>.log.
c=$1; s=$2; cd ~/spd/spd-nocdeadlock; tree=~/spd/spd-nocdeadlock/fixm
exec 8>~/spd_qb2_chip$c.lock; flock 8; echo "got chip$c $(date -u +%FT%TZ)"
for arm in cpcG1:--iters\ 1000\ --every\ 50 cpcG2:--iters\ 1000\ --every\ 50 cpcG3:--iters\ 1000\ --every\ 50 catomG:--atom\ --iters\ 3000\ --every\ 100; do
  tag=${arm%%:*}; args=${arm#*:}; echo "== $tag ($args) start $(date -u +%FT%TZ)"
  ./repro.sh $c $tag $tree $args
  l=rp_${tag}_c$c.log; to=$(grep -c "TIMEOUT: device timeout" $l)
  echo "$tag c$c: $(grep -c "^it " $l) prints, last: $(grep "^it " $l | tail -1), done=$(grep -c REPRO-DONE $l), timeout=$to"
  if [ "$to" != 0 ]; then exec 9>~/spd_qb2_chip$s.lock; flock 9; timeout 300 ~/.local/bin/tt-smi -r $(( c<s ? c : s )),$(( c<s ? s : c )) >/dev/null 2>&1; echo "HANG; reset rc=$? $(date -u +%FT%TZ)"; break; fi
done
echo "CONTROL-DONE c$c $(date -u +%FT%TZ)"
