#!/usr/bin/env bash
# Three-arm teardown-abort ladder. Each arm is its own process (one device context per process),
# serial on card 1. Arm order is by decisiveness, so a truncated run still answers the most.
OUT=/home/ttuser/.coworker/wt/land-standing/perf/land_standing/out/gate_land_merge
PY=/home/ttuser/bcx_e2e_venv/bin/python3
BC2=/home/ttuser/bcx_e2e/bc2
run () {  # run <label> <tree>
  local label=$1 tree=$2 log=$OUT/ladder_$1.log
  echo "### $label  tree=$tree  start=$(date -u +%H:%M:%SZ)" >> $OUT/ladder_SUMMARY.txt
  ( cd $tree && TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:land-standing \
    PYTHONPATH=$tree:$BC2 $PY -m pytest tests/test_bindcraft2_hw.py -q -rs -s ) > $log 2>&1
  local rc=$?
  local aborts=$(grep -cE "pthread_mutex_unlock failed|terminate called" $log)
  local res=$(grep -vE "leaked|nanobind|skipped remainder" $log | grep -E "passed|failed" | tail -1)
  local step=$(grep -o "step [0-9.]* s" $log | tail -1)
  echo "$label rc=$rc aborts=$aborts | $res | $step  end=$(date -u +%H:%M:%SZ)" >> $OUT/ladder_SUMMARY.txt
}
: > $OUT/ladder_SUMMARY.txt
echo "chain pid $$ started $(date -u +%H:%M:%SZ)" >> $OUT/ladder_SUMMARY.txt
run fixonly_1d1856090  /tmp/fixonly_ls
run merged_198e373a5   /home/ttuser/.coworker/wt/land-standing
run prefix_3997446c6   /tmp/prefix_ls
echo "CHAIN DONE $(date -u +%H:%M:%SZ)" >> $OUT/ladder_SUMMARY.txt
