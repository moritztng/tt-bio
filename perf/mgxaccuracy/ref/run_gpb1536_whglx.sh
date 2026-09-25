#!/bin/bash
# One upstream-fp32 CPU reference draw of the 1GPB 1536 cell, ON WHGLX.
#
# WHY HERE AND NOT qb2. The qb2 pair (out1536_valid, out1536_valid_b) is the deciding gate for
# clause 3 and has written nothing in 23 h and 18 h: qb2 has 16 cores at load 40, so each draw
# holds ~3 of them and its OMP threads spin against the oversubscription. whglx has 64 cores at
# load 22.8 and finished the SAME step on a 1536-residue GroEL target in 5:56:06. The qb2 draws
# are left running; these are additional draws, not replacements.
#
# Fixture is the byte-identical one, sha256 ffa258bb599f182ce49b9f78ab089653ca35d464471d41b1e490
# d535421e9caf, the same file the device's three 1536 offset-0 cells used and the same bytes as
# qb2's fx_valid/bgt1536.cif (results/upstream_1536_fixture_precleared.txt). Spec text identical.
#
# Three deviations from the shipped GPU step configs, all toward a MORE exact reference:
# accelerator gpu -> cpu, precision bf16-mixed -> 32, matmul_precision -> highest. Everything the
# design depends on stays at upstream's default: sampling_steps 500, recycling_steps 3,
# diffusion_samples 1. --steps omits `folding`, the whole-complex refold, exactly as
# ref/run_valid_b.sh does: design_folding reads design_dir, not folding's output, and scRMSD
# comes from design_folding. Nothing the comparison reads is skipped.
set -u
tag=$1; thr=$2
# The output dir is this draw's identity, so claim it before computing for 9 h. Both draws on
# 2026-09-24 were launched with tag=1536 while their LOGS were named wa and wb, so they shared
# one dir, overwrote each other's per-step artefacts and were both discarded
# (results/whglx_gpb1536_draws_collided.txt). `mkdir` without -p is the whole guard: it fails
# when the directory already exists, which is precisely the collision.
out="$HOME/bgref-work/outgpb1536_${tag}"
mkdir "$out" || { echo "REFUSED: $out already exists -- another draw owns it. Pick a fresh tag."; exit 1; }
export OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr NUMEXPR_NUM_THREADS=$thr
export CUDA_VISIBLE_DEVICES="" TT_VISIBLE_DEVICES=""
cd "$HOME/bgref-work"
CPU="trainer.accelerator=cpu"
t0=$(date +%s)
"$HOME/bgref-env/bin/boltzgen" run "$HOME/bgref-work/fx_gpb1536/bg1536.yaml" \
    --protocol protein-anything \
    --output "$out" \
    --steps design inverse_folding design_folding analysis \
    --num_designs 1 --budget 1 --devices 1 --num_workers 2 --use_kernels false \
    --config design          $CPU trainer.precision=32 matmul_precision=highest \
    --config inverse_folding $CPU \
    --config design_folding  $CPU trainer.precision=32
rc=$?
echo "GPB1536_WHGLX_DONE tag=${tag} rc=${rc} wall_s=$(( $(date +%s) - t0 ))"
