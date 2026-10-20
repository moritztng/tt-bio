[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 1800
# threads: 4
# qb1: qk_rope_heads BLOCK 1 / 4 / 4+SQUARE, 300M/600M/6B shapes vs float64, bit-identity, rope_only per block, copy floor
O=/home/ubuntu/spd/spd-gap-whlm/qb1/qkrh
[ -s $O/qkrh.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7 || exit 3
env  bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1200 python perf/spd_gap_whlm/qkrh_bench.py --block 1,4,4sq --out $O/qkrh.part.jsonl && mv $O/qkrh.part.jsonl $O/qkrh.jsonl
