[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 96
# cap: 3600
# threads: 8
# st1: the 6B stack (b8 normal + exact acc / fast HiFi3, K288) in one process vs everything off in another
O=/home/ubuntu/spd/spd-gap-whlm/st1/6b-normal
[ -s $O/stack/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1 || exit 3
for arm in base stack; do
  [ -s $O/$arm/ab.jsonl ] && continue
  if [ $arm = base ]; then E="TT_BIO_ESMC_6B_B8=0 TT_BIO_ESMC_6B_FAST_FIDELITY=HiFi4 TT_BIO_SHSDPA_K288=0"; else E=; fi
  env $E bash perf/spd_gap_whlm/run.sh $CHIP $O/$arm -- timeout -s INT 1600 python perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.esmc._ESMC_6B_B8 --model esmc-6b   --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 2 --out $O/$arm/ab.part.jsonl && mv $O/$arm/ab.part.jsonl $O/$arm/ab.jsonl
done
