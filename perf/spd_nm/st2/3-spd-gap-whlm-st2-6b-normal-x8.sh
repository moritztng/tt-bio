[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 96
# cap: 3600
# threads: 8
# st2: 6B on the default tree 300d711d7 (b8 + HiFi3 both modes, exact acc normal, K288); x8-base repeats st1's chip-7 hang on bf16 batch 8
O=/home/ubuntu/spd/spd-gap-whlm/st2/6b-normal-x8
[ -s $O/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t300d711d7 || exit 3
env  bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1600 python perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.esmc._ESMC_6B_B8 --model esmc-6b  --batch 8 --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 1 --out $O/ab.part.jsonl && mv $O/ab.part.jsonl $O/ab.jsonl
