#!/bin/bash
# bcx-bfp8: the bf16 control for the float64 VJP floor, then the rest of the NaN attribution.
#
# The b8 leg at 04:55Z graded extra0, extra1 and evo22 and put dz at Infinity, Infinity and NaN
# while every forward stayed at cos 0.99999. grade_b80 runs the same three blocks in bf16 so the
# pairing is this row_s and not a transcription.
cd /home/ttuser/.coworker/wt/bcx-bfp8 || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-bfp8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_bfp8/logs; S=$L/chain5.status; : > $S

while pgrep -f "round_ab.py --arm zeros --b8" > /dev/null; do sleep 10; done
echo "card free $(date -u +%FT%TZ)" >> $S

BFP8_B8=0 timeout 1700 $PY perf/bcx_bfp8/grade.py --card 3 --n 128 --blocks 0,1,26 \
  --seed 0 --out grade_b80.json > $L/grade_b80.log 2>&1 < /dev/null
echo "grade_b80 exit $? $(date -u +%FT%TZ)" >> $S

run() {  # run <name> <arm> <b8flag> <extramsa>
  timeout 900 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --arm "$2" $3 $4 --finite \
    --rounds 6 --seed 100 --card 3 --project perf/bcx_bfp8/runs/nan_$1 \
    --out ../bcx_bfp8/nan_$1_seed100.json > $L/nan_$1.log 2>&1 < /dev/null
  echo "nan_$1 exit $? $(date -u +%FT%TZ)" >> $S
}
# does the extra-MSA stack matter? it is the one module built AFTER set_fast_mode, so its stored
# weights are demoted too, which tenstorrent.py:1657 records as what put esmfold2 at NaN
run trace_b8_noxmsa trace --b8 ""
# and the shipped DEVICE_ZEROS-off program, for completeness
run eager_b8 eager --b8 --extra-msa
run zeros_bf16 zeros "" --extra-msa
echo "chain5 done $(date -u +%FT%TZ)" >> $S
