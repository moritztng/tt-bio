[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 2400
# threads: 4
# b8n2: esmc-600m normal page cell. base = bf16 weights (today's normal); b8 = block-fp8 weights, fp32 acc, arms HiFi3 (off) vs HiFi2 (on).
O=/home/ubuntu/spd/spd-gap-whlm/b8n2/esmc-600m
cd /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd || exit 3
for arm in b8 base; do
  [ -s $O/$arm/ab.jsonl ] && continue
  if [ $arm = base ]; then E="TT_BIO_ESMC_B8_NORMAL=0"; R=1; else E="TT_BIO_ESMC_B8_NORMAL=1"; R=3; fi
  env $E bash perf/spd_gap_whlm/run.sh $CHIP $O/$arm -- timeout -s INT 1100 python perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.esmc._ESMC_MM_BFP8_HIFI2 --model esmc-600m --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds $R --out $O/$arm/ab.part.jsonl && mv $O/$arm/ab.part.jsonl $O/$arm/ab.jsonl
done
