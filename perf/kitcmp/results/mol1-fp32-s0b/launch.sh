#!/bin/bash
# mol1-gpu-fp32 relaunch of shard 0/4 (s0 lost its box to a failed upstream clone, 22:44Z 11-02), 1.5 h cap.
cd /home/moritz/.coworker/wt/mol1-gpu-fp32/perf/kitcmp && exec env QB2=ttuser@tt-quietbox2:/home/ttuser/mol1-gpu-fp32 \
  SLUG=mol1-fp32 MINW=0 MAXW=99999 \
  BOXENV='REF=1 REF_K=1 REF_FIRST=8f5n-assembly1,8aok-assembly1,8w4b-assembly1,8fo5-assembly1:nomsa,8u12-assembly1,7y37-assembly1,7fxn-assembly1:nomsa REF_ARGS="--dtype fp32 --sets paired --seeds 42,66 --shard 0/4"' \
  bash mol1_session.sh mol1-fp32-s0b 1.5 'gpu_name=RTX_4090 num_gpus=1 gpu_ram>=46 cuda_max_good>=13.0 inet_down>=200 reliability>=0.95 disk_space>=160 dph<=0.55'
