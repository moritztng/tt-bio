[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 1800
# threads: 4
# qb1: page cell, qk_rope_heads.BLOCK 1 vs 4 (bit-identical expected)
O=/home/ubuntu/spd/spd-gap-whlm/qb1/esmc-600m-fast
[ -s $O/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7 || exit 3
env  bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1600 python perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.qk_rope_heads.BLOCK --on-value 4 --model esmc-600m --fast --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 3 --out $O/ab.part.jsonl && mv $O/ab.part.jsonl $O/ab.jsonl
