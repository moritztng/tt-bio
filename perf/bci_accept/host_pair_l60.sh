#!/bin/bash
# The host half of the #17 acceptance pair: binder 60 against hPDL1, full pipeline with the
# BindCraft 2 filters, 3 trajectories, seed schedule untouched so the trajectory hashes match the
# on-card arm row for row.
#
# design_dropout is FALSE on purpose. tt-bio's Evoformer applies no dropout, so false is the
# setting at which the two arms are the same model; an arm run with dropout on is a third point
# beside the pair, not the pair.
#
# Paths default to pc, where the host-JAX venv and the AF2 params live. Override by environment.
set -u
ROOT=${BCI_ROOT:-/home/moritz/.bci-accept-pc}          # the tree capture_logits.py runs from
OUTDIR=${BCI_OUT:-/home/moritz/.bci-accept-pair}
PY=${BCI_PY:-/home/moritz/bcx_hostcut_venv/bin/python}
BC2=${BCI_BC2:-/home/moritz/bcx_shipped/bc2}
AF2=${BCI_AF2:-/home/moritz/bcx_shipped/af2_params}
export JAX_PLATFORMS=cpu
export PYTHONPATH=$ROOT:$BC2
cd "$ROOT" || exit 1
mkdir -p "$OUTDIR/.bci"
LOG=$OUTDIR/.bci/host_full_l60_nodropout.log
{
  echo "=== HOST-JAX arm, binder 60, full pipeline with filters, 3 trajectories, dropout OFF ==="
  date -u +"start %Y-%m-%dT%H:%M:%SZ"
  nice -n 10 timeout 86400 "$PY" perf/bci_accept/capture_logits.py \
    --trunk jax --full \
    --target-pdb "$BC2/settings/target/structures/hPDL1.pdb" \
    --af2-weights "$AF2" \
    --out "$OUTDIR/.bci/logits_host_full_l60_nodropout.npz" \
    --project "$OUTDIR/proj_host_full_l60_nodropout" \
    --design-dropout false --trajectories 3 --trajectories-per-card 1 \
    --binder-lengths 60 60
  echo "=== rc=$? ==="
  date -u +"end %Y-%m-%dT%H:%M:%SZ"
} > "$LOG" 2>&1
