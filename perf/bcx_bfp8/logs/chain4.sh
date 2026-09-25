#!/bin/bash
# bcx-bfp8: does bfp8 NaN the design loop on its own, or only under trace replay?
#
# The four bfp8 arms that died at round 1 were all --arm trace. `zeros` is the same program
# without the replay and with DEVICE_ZEROS on, which is what bcx-devzeros settled the flag to,
# so zeros-vs-trace isolates the capture and nothing else. `eager` additionally puts
# DEVICE_ZEROS back off (round_ab.py:144), which is the shipped program.
cd /home/ttuser/.coworker/wt/bcx-bfp8 || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-bfp8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_bfp8/logs; S=$L/chain4.status; : > $S

# wait out the float64 VJP leg that already owns card 3
while kill -0 225243 2>/dev/null; do sleep 15; done
echo "card free $(date -u +%FT%TZ)" >> $S

run() {  # run <name> <arm> <b8flag>
  local name=$1 arm=$2 flag=$3
  timeout 900 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --arm "$arm" $flag --finite \
    --extra-msa --rounds 6 --seed 100 --card 3 --project perf/bcx_bfp8/runs/nan_$name \
    --out ../bcx_bfp8/nan_${name}_seed100.json > $L/nan_$name.log 2>&1 < /dev/null
  echo "nan_$name exit $? $(date -u +%FT%TZ)" >> $S
}
run zeros_b8   zeros --b8
run zeros_bf16 zeros ""
run eager_b8   eager --b8
run eager_bf16 eager ""

# the bf16 half of the float64 VJP floor, last: it is the slow leg and not the decisive one
BFP8_B8=0 timeout 1700 $PY perf/bcx_bfp8/grade.py --card 3 --n 128 --blocks 0,1,26 \
  --seed 0 --out grade_b80.json > $L/grade_b80.log 2>&1 < /dev/null
echo "grade_b80 exit $? $(date -u +%FT%TZ)" >> $S
echo "chain4 done $(date -u +%FT%TZ)" >> $S
