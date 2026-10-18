[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 64
# cap: 3600
# threads: 8
O=/home/ubuntu/spd/spd-gap-whlm/b8n
[ -s $O/b8/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/tab04a6ec5 || exit 3
for arm in bf16 b8; do
  B=0; [ $arm = b8 ] && B=1
  [ -s $O/$arm/ab.jsonl ] && continue
  TT_BIO_ESMC_6B_B8=$B bash perf/spd_gap_whlm/run.sh $CHIP $O/$arm -- timeout -s INT 1600 python perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.esmc._ESMC_6B_B8 --model esmc-6b --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 3 --out $O/$arm/ab.part.jsonl && mv $O/$arm/ab.part.jsonl $O/$arm/ab.jsonl
done
