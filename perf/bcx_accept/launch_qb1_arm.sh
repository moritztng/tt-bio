#!/bin/bash
# bcx-accept: one detached acceptance trajectory arm on a qb1 p150a card.
#
# Differences from launch_accept_arm.sh, and each one is deliberate:
#
#  * `perf/bcx_exact/traj_arm.py --exact off` instead of `run_arm.py`. `origin/main` arms a host
#    float64 `exact_training` instrument that BindCraft 2's loop is inside by construction
#    (`state/bcx/MAIN-PAYS-EXACT.md`), so the SHIPPABLE configuration on a current tree is the
#    campaign wrapped in `ag.exact_training(False)`, which is what this arm runs. The earlier
#    qb2 draws ran on `c014a2a7d`, which predates the instrument, so they are already the
#    instrument-off arithmetic -- `traj_arm.py` dumps `EXACT_SOFTMAX_STATS` and
#    `EXACT_LAYER_NORM_STATS` and an OFF arm whose counters moved is a failed run, which is the
#    check that decides whether the two sets pool.
#
#  * the clock is sampled by THIS script into a CSV, not only by `meter.Clock`. `Clock` keeps its
#    samples in a list and they die with the process, so an arm that is killed loses the clock
#    that every perf number it produced has to carry.
#
#  * sysfs, not `tt-smi -s`. tt-smi has hung for minutes at a time on qb1 with the box idle
#    (`cardblock-qb1-1`, three attempts at rc=124), and `perf/bcx_stack/stack.py` reads the same
#    number straight off the class node with no device open. The tt_* attributes live on the
#    CLASS node; `.../device/tt_aiclk` reads empty for every card including a computing one.
#
#  * `--trajectories 6`. `AUTOTUNE_REVIEW_TRAJECTORIES = 10` with autotune on by default, so
#    trajectory 11 is a different process from trajectory 1 and a ledger may not cross ten
#    without splitting there. Six stays under it with room.
set -u
ART=${ART:-/home/ttuser/bcx_accept_art/qb1}
TREE=${TREE:-/home/ttuser/bcx_accept_art/qb1tree}
SEED=${SEED:?set SEED}
CARD=${CARD:?set CARD}
TRAJ=${TRAJ:-6}
EXACT=${EXACT:-off}
OMP=${OMP:-8}
PY=${PY:-/home/ttuser/bcx_e2e_venv/bin/python3}
TAG=qb1_s$SEED
OUT=$ART/$TAG

mkdir -p "$ART"
cd "$TREE" || exit 1
rm -rf "$OUT"

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bcx-accept
export OMP_NUM_THREADS=$OMP

# The class node this card's AICLK lives on, resolved the same way the harness resolves it.
NODE=$($PY -c "
import sys; sys.path.insert(0, 'perf/bcx_stack')
from stack import sysfs_node
print(sysfs_node()[0])" 2>/dev/null)

{
  echo "launched   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host       $(hostname)   board p150a"
  echo "tree       $TREE"
  echo "commit     $(git rev-parse HEAD)"
  echo "dirty      $(git status --porcelain | wc -l) files"
  echo "exact      $EXACT"
  echo "card       $CARD   sysfs $NODE"
  echo "seed       $SEED   trajectories $TRAJ   omp $OMP"
  echo "loadavg    $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$ART/${TAG}_stamp.txt"

$PY -u perf/bcx_exact/traj_arm.py --exact "$EXACT" --seed "$SEED" \
    --trajectories "$TRAJ" --resident 1 \
    --params /home/ttuser/bcx_e2e/af2_params \
    --out "$OUT" > "$ART/$TAG.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$ART/$TAG.pid"
echo "arm pid    $PID" >> "$ART/${TAG}_stamp.txt"

# AICLK DURING the fold. Flushed per sample, so a killed arm still leaves the clock it ran at.
(
  while kill -0 "$PID" 2>/dev/null; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$(cat "$NODE/tt_aiclk" 2>/dev/null),$(cut -d' ' -f1 /proc/loadavg)" \
      >> "$ART/${TAG}_clock.csv"
    sleep 60
  done
) &
CLOCK=$!

wait "$PID"
RC=$?
kill "$CLOCK" 2>/dev/null
# The ARM's status, not a `kill -0` loop's: accept_s3 crashed with a KeyError and its stamp still
# read `arm exit 0`. Read the log tail before believing a trajectory completed either way.
echo "arm exit   $RC $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ART/${TAG}_stamp.txt"
