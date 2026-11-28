#!/bin/bash
# r4t-perf L4: site grade (a), broken arm (d), then block A/B off1/on/off2 (L2 weights ON in every arm, so only L4 moves).
cd ~/.coworker/wt/r4t-perf || exit 1
O=${L4OUT:-$HOME/r4t/perf/l4}; mkdir -p $O; PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
E="PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_LNBW_WEIGHTS=1"
free() { python3 ~/spd/feed/book.py active 2 2>/dev/null | grep -qw fleet-r4t-perf && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; }
go() { for i in $(seq 1 45); do free && break; sleep 20; done; free || { echo "NO WINDOW $1 $(date -u +%FT%TZ)" >> $O/run.log; exit 4; }; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $O/run.log
go a; env $E TT_BIO_MMVJP_MIN=1 timeout 300 $PY perf/r4t_perf/l4grade.py --out $O/a.json > $O/a.log 2>&1; echo "a RC $? $(tail -1 $O/a.log)" >> $O/run.log
go d; env $E TT_BIO_MMVJP_MIN=1 TT_BIO_MMVJP_BROKEN=1 timeout 300 $PY perf/r4t_perf/l4grade.py --out $O/d.json > $O/d.log 2>&1; echo "d RC $? $(tail -1 $O/d.log)" >> $O/run.log
for arm in off1 on off2; do m=0; [ $arm = on ] && m=1
  go $arm; env $E TT_BIO_MMVJP_MIN=$m timeout 400 $PY perf/r4t_perf/blockprof.py --n 256 --reps 5 --out $O/b_$arm.json > $O/b_$arm.log 2>&1
  echo "$arm RC $? $(date -u +%FT%TZ)" >> $O/run.log
done
echo L4-DONE >> $O/run.log
