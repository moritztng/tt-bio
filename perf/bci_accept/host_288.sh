#!/bin/bash
# The HOST arm at the reporter's token axis, to pair with the card arm queued on qb1 card 1.
# hPDL1 is 115 residues, so binder 173 gives 288 tokens. One trajectory, trajectory-only: the
# question at this axis is whether the stage curve holds through `anneal` the way #17 reports,
# and that is a trajectory property. Acceptance at 288 would cost a ProteinMPNN redesign and a
# validation ensemble at 288 tokens on 12 CPU cores, which is not affordable tonight and is not
# what this arm is for.
#
# It waits on ONE NAMED PID -- the dropout discriminator -- rather than on a guess about how
# long something has left. pc has 12 cores and two arms on it; this is the third and the least
# urgent of the three, so it starts when a core frees rather than by taking one.
set -u
ROOT=/home/moritz/.bci-accept-288
SRC=/home/moritz/.bci-accept-pc
PY=/home/moritz/bcx_hostcut_venv/bin/python
BC2=/home/moritz/bcx_shipped/bc2
WAIT_FOR=${1:?pid to wait on}
PROJ=$ROOT/proj_host_288
LOG=$ROOT/.bci/host_288.log

export JAX_PLATFORMS=cpu
export PYTHONPATH=$SRC:$BC2
cd "$SRC" || exit 1
mkdir -p "$ROOT/.bci"
[ -e "$PROJ" ] && { echo "$PROJ exists; move it aside, a resume is not a rerun" >&2; exit 1; }

echo "waiting on pid $WAIT_FOR from $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while kill -0 "$WAIT_FOR" 2>/dev/null; do sleep 60; done
echo "pid $WAIT_FOR gone at $(date -u +%Y-%m-%dT%H:%M:%SZ), load $(cut -d' ' -f1-3 /proc/loadavg)"

{
  echo "=== host arm at 288 tokens: binder 173 + hPDL1 115, 1 trajectory, trajectory-only ==="
  date -u +'start %Y-%m-%dT%H:%M:%SZ'
  nice -n 10 timeout 86400 "$PY" perf/bci_accept/capture_logits.py \
    --trunk jax \
    --target-pdb "$BC2/settings/target/structures/hPDL1.pdb" \
    --af2-weights /home/moritz/bcx_shipped/af2_params \
    --out "$ROOT/.bci/logits_host_288.npz" \
    --project "$PROJ" --design-dropout true \
    --trajectories 1 --trajectories-per-card 1 \
    --binder-lengths 173 173
  echo "=== rc=$? ==="
  date -u +'end %Y-%m-%dT%H:%M:%SZ'
} >> "$LOG" 2>&1
