#!/bin/bash
# spd-trans2 BH job: take the first free card among CARDS, run JOB's arms on it. Gives up after 8 h of waiting.
#   bh_trans2.sh HOST JOB "CARDS"
set -u
H=$1 JOB=$2 CARDS=$3
B=~/spd/spd-trans2; O=$B/bh/$JOB; mkdir -p $O; T=$B/t2; S=$B/s19
say(){ echo "$(date -u +%FT%TZ) $*" >> $O/run.log; }
lockf(){ if [ $H = qb1 ]; then echo ~/spd_qb1_card$1.lock; else echo ~/spd_qb2_chip$1.lock; fi; }
say "waiting for a card in [$CARDS]"
end=$(( $(date +%s) + 28800 ))
while [ $(date +%s) -lt $end ]; do
  for c in $CARDS; do
    exec 9> "$(lockf $c)"
    if flock -n 9; then
      say "card $c taken"
      export TT_BIO_LEASE_HOLDER=spd-trans2 WAIT=5 LOCK=$O/inner.lock PY=/home/ttuser/tt-bio-dev/env/bin/python
      case $JOB in
        ab)  for a in "$S s19n" "$T t2n" "$S s19f:fast" "$T t2f:fast" "$S s19n" "$T t2n"; do
               set -- $a; bash $1/perf/spd/run_chip.sh $c $O/c730 "$2" c730 2400 --seed 101 --warm 3 < /dev/null; done ;;
        lad) bash $T/perf/spd/run_chip.sh $c $O/n t2n l512,l1536,l2048 7200 --seed 101 --warm 0 < /dev/null
             bash $T/perf/spd/run_chip.sh $c $O/f t2f:fast l512,l1536,l2048 7200 --seed 101 --warm 0 < /dev/null ;;
        gn)  for cx in 28VJ 9DBP 9HL2 9LLG 9LV4 9PCQ 9TH6 9TY2 9W3L 9W89 9W8A; do
               bash $T/perf/spd/run_chip.sh $c $O/$cx t2n $cx 2400 --seed 101 --warm 3 < /dev/null; done ;;
      esac
      say "done card $c"; exit 0
    fi
    exec 9>&-
  done
  sleep 60
done
say "gave up"
