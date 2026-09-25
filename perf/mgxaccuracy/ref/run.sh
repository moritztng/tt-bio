#!/bin/bash
# One upstream-fp32 CPU reference draw of the BoltzGen pipeline. $1 = size tag, $2 = threads.
#
# Three deviations from the shipped GPU step configs, all recorded, all in the direction of a
# MORE exact reference: accelerator gpu -> cpu, precision bf16-mixed -> 32, matmul_precision
# high (TF32 on an NVIDIA card) -> highest. Everything the design itself depends on is left at
# the upstream default: sampling_steps 500, recycling_steps 3, diffusion_samples 1.
size=$1; thr=$2
export OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr NUMEXPR_NUM_THREADS=$thr
export CUDA_VISIBLE_DEVICES="" TT_VISIBLE_DEVICES=""
cd ~/bgref-work
CPU="trainer.accelerator=cpu"
t0=$(date +%s)
~/bgref-env/bin/boltzgen run ~/bgref-work/fx/bg${size}.yaml \
    --protocol protein-anything \
    --output ~/bgref-work/out${size} \
    --num_designs 1 --budget 1 --devices 1 --num_workers 2 --use_kernels false \
    --config design          $CPU trainer.precision=32 matmul_precision=highest \
    --config inverse_folding $CPU \
    --config folding         $CPU trainer.precision=32 \
    --config design_folding  $CPU trainer.precision=32 \
    --config affinity        $CPU trainer.precision=32
rc=$?
echo "REF_DONE size=${size} rc=${rc} wall_s=$(( $(date +%s) - t0 ))"
