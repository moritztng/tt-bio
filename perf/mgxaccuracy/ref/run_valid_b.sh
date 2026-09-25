#!/bin/bash
# One upstream-fp32 CPU reference draw against the VALID target (gpb_dimer_1646 crop), which
# is what the device ran. $1 = size tag, $2 = threads.
#
# The earlier reference (run.sh, ~/bgref-work/fx/) used the DISQUALIFIED disconnected rung --
# chains A=1008/B=528 with 162.40 A between them -- so its numbers could not be set beside the
# device's. fx_valid/ is built from perf/mgxaccuracy/targets/gpb_dimer_1646.cif and its
# coord_sha matches the device fixture exactly (34d0b6f4... at 1536, 7c38b308... at 512),
# checked with perf/mgxaccuracy/fixture_id.py before this script was written.
#
# Three deviations from the shipped GPU step configs, all in the direction of a MORE exact
# reference: accelerator gpu -> cpu, precision bf16-mixed -> 32, matmul_precision -> highest.
# Everything the design depends on stays at the upstream default: sampling_steps 500,
# recycling_steps 3, diffusion_samples 1.
#
# --steps omits `folding`, the whole-complex refold. That step is the expensive one at 1616
# tokens (68 min on the old fixture with zero structures written) and design_folding does not
# consume its output -- it reads design_dir, and the two step configs differ by two lines.
# scRMSD comes from design_folding. Nothing the comparison reads is skipped.
set -u
size=$1; thr=$2
export OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr NUMEXPR_NUM_THREADS=$thr
export CUDA_VISIBLE_DEVICES="" TT_VISIBLE_DEVICES=""
cd "$HOME/bgref-work"
CPU="trainer.accelerator=cpu"
t0=$(date +%s)
"$HOME/bgref-env/bin/boltzgen" run "$HOME/bgref-work/fx_valid/bg${size}.yaml" \
    --protocol protein-anything \
    --output "$HOME/bgref-work/out${size}_valid_b" \
    --steps design inverse_folding design_folding analysis \
    --num_designs 1 --budget 1 --devices 1 --num_workers 2 --use_kernels false \
    --config design          $CPU trainer.precision=32 matmul_precision=highest \
    --config inverse_folding $CPU \
    --config design_folding  $CPU trainer.precision=32
rc=$?
echo "REF_VALID_DONE size=${size}b rc=${rc} wall_s=$(( $(date +%s) - t0 ))"
