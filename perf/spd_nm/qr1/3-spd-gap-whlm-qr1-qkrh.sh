[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 1800
# threads: 4
# qr1: where qk_rope_heads' time goes (fidelity HiFi4/3/2/LoFi, norm off, stock copy floor), 300M/600M/6B shapes, vs float64
O=/home/ubuntu/spd/spd-gap-whlm/qr1/qkrh
[ -s $O/qkrh.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/tef62619e7 || exit 3
env  bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1200 python perf/spd_gap_whlm/qkrh_bench.py --out $O/qkrh.part.jsonl && mv $O/qkrh.part.jsonl $O/qkrh.jsonl
