[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 1800
# threads: 4
# qb2: qk_rope_heads BLOCK 1 / 4 after the QKRH_HG fix, bit-identity on 300M (15 heads)
O=/home/ubuntu/spd/spd-gap-whlm/qb2/qkrh
[ -s $O/qkrh.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5 || exit 3
env  bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1200 python perf/spd_gap_whlm/qkrh_bench.py --block 1,4 --out $O/qkrh.part.jsonl && mv $O/qkrh.part.jsonl $O/qkrh.jsonl
