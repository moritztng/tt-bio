#!/bin/bash
# bci-seventeen: the HOST-JAX arm of #17, on the rented GPU. Runs ON the box.
#
# Pairs row for row with the on-card arm: same seed 42, same binder 173 against hPDL1's 115
# residues (288 tokens), same BindCraft 2 v1.0.1 filters, same design_dropout=false, --full so a
# redesign and a validation ensemble actually run and an acceptance count exists. Both arms at the
# same seed and binder length draw the same 8 trajectory hashes, so "the same 8 seeds" holds by
# construction rather than by arrangement.
#
# Never re-enters a project folder: BindCraft 2 charges a claimed-but-unfinished trajectory against
# the budget and never retries it, so a resume would silently run fewer than 8.
set -u
ROOT=/root/bci-seventeen
LOG=$ROOT/host_8traj_288.log
PROJ=$ROOT/proj_host_8traj_288
mkdir -p "$ROOT"

export PYTHONPATH=/root/tt-bio:/root/bcx_shipped/bc2
cd /root/tt-bio || exit 1

exec >> "$LOG" 2>&1
echo "=== host JAX arm: 8 trajectories, 288 tokens, seed 42, GPU ==="
date -u +"start %Y-%m-%dT%H:%M:%SZ"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('jax', jax.__version__, jax.devices())"

[ -e "$PROJ" ] && { echo "$PROJ exists; a resume is not a rerun, move it aside"; exit 1; }

( while :; do printf "%s %s\n" "$(date -u +%H:%M:%SZ)" \
    "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"; \
    sleep 60; done > "$ROOT/host_8traj_288.gpu" ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

python perf/bci_accept/capture_logits.py \
  --trunk jax --full \
  --target-pdb /root/bcx_shipped/bc2/settings/target/structures/hPDL1.pdb \
  --af2-weights /root/bcx_shipped/af2_params \
  --out "$ROOT/logits_host_8traj_288.npz" \
  --project "$PROJ" \
  --design-dropout false --trajectories 8 --trajectories-per-card 1 \
  --binder-lengths 173 173
echo "=== rc=$? ==="
date -u +"end %Y-%m-%dT%H:%M:%SZ"
kill $SAMPLER 2>/dev/null
