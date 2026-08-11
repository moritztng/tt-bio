#!/usr/bin/env bash
# Clean re-run of the E6 fold A/B: wait until no other fold is on the box, then benchlock.
set -u
WT=/home/ttuser/.coworker/wt/trimul-fused-kernel-final
cd "$WT" || exit 70
MINE=$$
others() { pgrep -f 'python3? .*(fold_ab512|tt_concurrency)\.py' | grep -v "^$MINE\$" | tr '\n' ' '; }
echo "start $(date -Is) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
i=0
while [ $i -lt 60 ]; do
  o=$(others); [ -z "$o" ] && break
  echo "waiting on folds: $o"
  sleep 15; i=$((i+1))
done
o=$(others)
if [ -n "$o" ]; then echo "ABORT: folds still running: $o"; exit 75; fi
echo "box clear of folds at $(date -Is) after $((i*15))s"

BENCHLOCK_MAXLOAD=0.8 BENCHLOCK_LOAD_WAIT_S=420 \
/home/ttuser/.coworker/scripts/benchlock.sh trimul-fused-kernel-final -- \
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:trimul-fused-kernel-final \
      PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/size512/fold_ab512.py \
    --sizes 512 --arms on,e6,on,e6 \
    --out perf/trimul_f2/fold_e6_512_qb2c0_quiet.json
rc=$?
echo "done $(date -Is) rc=$rc loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
exit $rc
