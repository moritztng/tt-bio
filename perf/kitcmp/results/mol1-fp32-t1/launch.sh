#!/bin/bash
# mol1-gpu-fp32 tail box 11-03: r1's 4 leftovers at its cap, the 3 CUDA OOM jobs (8bjj s42/s66, upstream 7wux s66) with
# expandable_segments (allocator only, numerics unchanged), and the last shard-0 cases r0d will not reach by its 18:50Z cap.
# Every finished job is hardlinked into qb2 inputs/_ref first, so only these run; 1.75 h cap.
cd /home/moritz/.coworker/wt/mol1-gpu-fp32/perf/kitcmp && exec env QB2=ttuser@tt-quietbox2:/home/ttuser/mol1-gpu-fp32 \
  SLUG=mol1-fp32 MINW=0 MAXW=99999 \
  BOXENV='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True REF=1 REF_K=1 REF_FIRST=8bjj-assembly1,upstream:example.json#7wux,8qa8-assembly1:smiles,8q34-assembly1:file,8w4b-assembly1:file,8sio-assembly1:smiles,8b9e-assembly1:smiles,8txe-assembly1:smiles REF_ARGS="--dtype fp32 --sets paired --seeds 42,66"' \
  bash mol1_session.sh mol1-fp32-t1 1.75 'gpu_name=RTX_4090 num_gpus=1 gpu_ram>=46 cuda_max_good>=13.0 inet_down>=200 reliability>=0.95 disk_space>=160 dph<=0.55'
