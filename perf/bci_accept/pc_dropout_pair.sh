#!/bin/bash
# The dropout test as a PAIR on one host, both arms at once.
#
# dropout_arms.sh runs the two arms back to back on qb2, which is fine for the result and bad for
# the clock: screen alone took 2.6 h there under a load average of 23, so the arm that actually
# carries the hypothesis (design_dropout=false, what the on-card Evoformer does) would not report
# until the small hours. This runs both arms side by side on pc, which is idle and has 12 cores,
# so the comparison lands in one arm's wall time instead of two.
#
# Same host, same stack, same seeds for both arms, so the only difference between them is the flag.
# jax 0.11.2 and BindCraft 2 7a2dfdb on both boxes; tt_bio is rsynced from the qb2 worktree at the
# revision in SRC_REV.txt, so the two hosts run the same code and not merely the same version.
# qb2's already-running design_dropout=true arm is then a cross-host replicate of this script's
# true arm, which is the only measurement of how much the host itself moves a trajectory, a number
# NOISE needs and nothing else here provides.
#
# pc has no Tenstorrent card in play and opens none: both arms are host JAX by construction.
set -u
ROOT=/home/moritz/.bci-accept-pc
PY=/home/moritz/bcx_hostcut_venv/bin/python
BC2=/home/moritz/bcx_shipped/bc2
export JAX_PLATFORMS=cpu
export PYTHONPATH=$ROOT:$BC2
cd "$ROOT" || exit 1
TRAJ=${1:-3}
LEN=${2:-60}

mkdir -p "$ROOT/.bci"
for arm in true false; do
  project=$ROOT/proj_$arm
  rm -rf "$project"
  nice -n 10 timeout 28800 "$PY" perf/bci_accept/capture_logits.py \
    --target-pdb "$BC2/settings/target/structures/hPDL1.pdb" \
    --af2-weights /home/moritz/bcx_shipped/af2_params \
    --out "$ROOT/.bci/logits_$arm.npz" --dump-states "$ROOT/.bci/harden_states_$arm.pkl" \
    --project "$project" --design-dropout "$arm" --trajectories "$TRAJ" \
    --binder-lengths "$LEN" "$LEN" > "$ROOT/.bci/pc_arm_$arm.log" 2>&1 &
  echo "arm $arm pid $!"
done
wait
echo "=== both pc arms done ==="
