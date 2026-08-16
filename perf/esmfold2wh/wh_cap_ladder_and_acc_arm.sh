#!/bin/bash
# The depth-cap arm. Same rungs, same flags; the cap is applied inside the engine
# (tenstorrent.msa_depth_cap), so nothing on this command line changes.
set -u
SRC=/home/cust-team/mthuening/esmfold2wh/src
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3
D=/home/cust-team/mthuening/esmfold2wh
for L in 788 1024 640; do
  env -i HOME=/home/cust-team PATH=/usr/bin:/bin \
    HF_HUB_CACHE=/home/cust-team/models TT_VISIBLE_DEVICES=28 \
    TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-msa-cap-p2 \
    TT_BIO_DRAM_PEAK=$D/dram_cap_$L.log PYTHONPATH=$SRC \
    timeout 3000 $PY -m tt_bio.main predict $D/cdk2_$L.fasta \
      --model esmfold2 --fast --use_msa_server --msa_dir /data/msa_cache --seed 0 \
      --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
      --out_dir $D/out_cap_$L --override > $D/run_cap_$L.log 2>&1
  echo "cap_$L exit=$? $(ls $D/out_cap_$L/*/structures/*.cif 2>/dev/null | wc -l) cif"
done
#!/bin/bash
# What the depth cap costs, priced where both depths fold. The cap is a no-op at 640 aa,
# so the only way to score it is to impose the deepest ratio it ever imposes (1024 aa gets
# 5120) at a length that folds at full depth too, and diff the two structures.
set -u
D=/home/cust-team/mthuening/esmfold2wh
SRC=$D/src
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3
while pgrep -f cap_ladder.sh > /dev/null; do sleep 20; done
env -i HOME=/home/cust-team PATH=/usr/bin:/bin \
  HF_HUB_CACHE=/home/cust-team/models TT_VISIBLE_DEVICES=28 \
  TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-msa-cap-p2 \
  TT_BIO_DRAM_PEAK=$D/dram_acc_640_m5120.log PYTHONPATH=$SRC \
  timeout 3000 $PY -m tt_bio.main predict $D/cdk2_640.fasta \
    --model esmfold2 --fast --use_msa_server --msa_dir /data/msa_cache --seed 0 \
    --max_msa_seqs 5120 --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
    --out_dir $D/out_acc_640_m5120 --override > $D/run_acc_640_m5120.log 2>&1
echo "acc_640_m5120 exit=$? $(ls $D/out_acc_640_m5120/*/structures/*.cif 2>/dev/null | wc -l) cif"
