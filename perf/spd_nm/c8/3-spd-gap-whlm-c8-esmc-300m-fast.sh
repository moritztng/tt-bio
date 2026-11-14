[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 48
# cap: 2400
# threads: 8
# c8: 8x8 minimal_matmul blocks under bf16 acc (--fast bfp8 qkv split); normal cell is the no-op control
O=/home/ubuntu/spd/spd-gap-whlm/c8/esmc-300m-fast
[ -s $O/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6 || exit 3
bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 2000 python perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.esmc._ESMC_MM_CFG_BF16_ACC_8X8 --model esmc-300m --fast --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 3 --out $O/ab.part.jsonl && mv $O/ab.part.jsonl $O/ab.jsonl
