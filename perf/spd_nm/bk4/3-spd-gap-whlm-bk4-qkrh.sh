[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 16
# cap: 1500
# threads: 4
# bk4: qk_rope_heads BLOCK 4 default, SQUARE removed: kernel still builds, b4 bit-identical to b1
O=/home/ubuntu/spd/spd-gap-whlm/bk4/qkrh
[ -s $O/qkrh.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625 || exit 3
env  bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1200 python perf/spd_gap_whlm/qkrh_bench.py --block 1,4 --out $O/qkrh.part.jsonl && mv $O/qkrh.part.jsonl $O/qkrh.jsonl
