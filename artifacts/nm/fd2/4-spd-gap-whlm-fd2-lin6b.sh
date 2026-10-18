[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 1800
# threads: 4
[ -s /home/ubuntu/spd/spd-gap-whlm/fd2/lin6b/rc ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t6ee8ed8d5 && bash perf/spd_gap_whlm/run.sh $CHIP /home/ubuntu/spd/spd-gap-whlm/fd2/lin6b -- sh -c 'for f in HiFi4 HiFi3 HiFi2; do timeout -s INT 400 python perf/spd_gap_whlm/lin6b_sweep.py --fidelity $f --out /home/ubuntu/spd/spd-gap-whlm/fd2/lin6b/bf16_$f.jsonl || exit 1; done; for f in HiFi4 HiFi2; do timeout -s INT 400 python perf/spd_gap_whlm/lin6b_sweep.py --wdtype bfloat8_b --fidelity $f --out /home/ubuntu/spd/spd-gap-whlm/fd2/lin6b/bfp8_$f.jsonl || exit 1; done'
