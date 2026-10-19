[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 48
# cap: 2400
# threads: 8
# rl2: residual stream + FFN LayerNorm in L1 (TT_BIO_ESMC_RESID_L1), expect bit-identical and faster
O=/home/ubuntu/spd/spd-gap-whlm/rl2/esmc-600m-fast
[ -s $O/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e || exit 3
bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 2000 python perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.esmc._ESMC_RESID_L1 --model esmc-600m --fast --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 3 --out $O/ab.part.jsonl && mv $O/ab.part.jsonl $O/ab.jsonl
