#!/bin/bash
# AFTER's host arm, launched the moment pc frees its cores rather than when a turn notices.
#
#   nohup bash pc_host_after.sh [trajectories] [binder_length] &
#
# This is the host-JAX control for card_campaign.sh: same campaign seed, same 50/25/45/5 schedule,
# same binder length, same --trajectories-per-card 1, --design-dropout true (BindCraft 2's own
# default, which is what the reporter's host arm ran), and --full so there is an accepted count to
# put beside the per-stage table. It opens no device and never touches a chip.
#
# It waits instead of launching now because pc has 12 cores and a load average of 16: the two
# dropout arms already running are the decisive measurement for DIVERGE, and starting an 11 h third
# campaign beside them would slow the thing this row is actually graded on. Those arms are in their
# last stage, so the wait is short and the cost of waiting is zero.
#
# Disk is not a constraint here and the earlier note in the state doc that said otherwise was a
# guess: the --full smoke's whole project folder measured 484 KB, so three trajectories at binder 60
# is single-digit MB against pc's hundreds of MB free. It is checked below anyway, because the arm
# runs for hours unattended and a full root is the one failure that would lose all of it.
set -u
ROOT=/home/moritz/.bci-accept-pc
PY=/home/moritz/bcx_hostcut_venv/bin/python
BC2=/home/moritz/bcx_shipped/bc2
TRAJ=${1:-3}
LEN=${2:-60}
PROJ=$ROOT/proj_host_after
LOG=$ROOT/.bci/pc_host_after.log

export JAX_PLATFORMS=cpu
export PYTHONPATH=$ROOT:$BC2
cd "$ROOT" || exit 1
mkdir -p "$ROOT/.bci"

# BindCraft 2 resumes against an existing project folder and would skip the trajectories it thinks
# it already attempted, which un-pairs the arms by trajectory number. A resume is not a rerun.
[ -e "$PROJ" ] && { echo "$PROJ exists; move it aside, a resume is not a rerun" >&2; exit 1; }

# Wait for the dropout pair to finish. Poll the pids, not the logs: a log goes quiet mid-stage
# because BindCraft 2 prints only at stage end, so a quiet log is not a finished arm.
# The wait above is a guess about how long the pair has left, and a guess decays. Set
# BCI_AFTER_NOWAIT=1 when you have looked at the load and the pair's actual progress and decided
# there are cores to spare: this arm is on the critical path for AFTER, and idling it behind a
# restarted arm that turned out to have hours left costs more than the contention does.
if [ "${BCI_AFTER_NOWAIT:-0}" = 1 ]; then
  echo "BCI_AFTER_NOWAIT=1, starting beside the pair at $(date -u +%Y-%m-%dT%H:%M:%SZ), load $(cut -d' ' -f1-3 /proc/loadavg)"
else
  echo "waiting for the dropout pair to free pc's cores, from $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  while pgrep -f 'capture_logits.py .*--design-dropout (true|false) --trajectories' >/dev/null 2>&1; do
    sleep 60
  done
  echo "pair gone at $(date -u +%Y-%m-%dT%H:%M:%SZ), load $(cut -d' ' -f1-3 /proc/loadavg)"
fi

FREE_MB=$(df -Pm / | awk 'NR==2 {print $4}')
[ "$FREE_MB" -lt 200 ] && { echo "only ${FREE_MB}MB free on /, refusing to start an hours-long arm" >&2; exit 1; }

# Refresh the code from the worktree, and do it HERE rather than at launch: pc's tree was rsynced at
# d663b3e8f and --full only landed in 152e5f2cc, so the copy this arm needs does not exist on pc
# until now. Doing it after the pair exits means the running arms never have their sources swapped
# underneath them, and it pins what this arm ran to a revision instead of to a date.
rsync -a --delete ttuser@tt-quietbox2:/home/ttuser/.coworker/wt/bci-accept/perf/bci_accept/ \
  "$ROOT/perf/bci_accept/" || { echo "rsync of perf/bci_accept failed, not starting" >&2; exit 1; }
ssh ttuser@tt-quietbox2 'git -C /home/ttuser/.coworker/wt/bci-accept rev-parse HEAD' \
  | sed 's/^/SRC_REV /' > "$ROOT/SRC_REV.txt"
grep -q 'add_argument("--full"' "$ROOT/perf/bci_accept/capture_logits.py" \
  || { echo "refreshed capture_logits.py still has no --full, not starting" >&2; exit 1; }

{
  echo "=== AFTER host arm: trunk jax, $TRAJ trajectories, binder $LEN, --full ==="
  cat "$ROOT/SRC_REV.txt"
  date -u +'start %Y-%m-%dT%H:%M:%SZ'
  nice -n 10 timeout 57600 "$PY" perf/bci_accept/capture_logits.py \
    --trunk jax --full \
    --target-pdb "$BC2/settings/target/structures/hPDL1.pdb" \
    --af2-weights /home/moritz/bcx_shipped/af2_params \
    --out "$ROOT/.bci/logits_host_after.npz" \
    --project "$PROJ" --design-dropout true \
    --trajectories "$TRAJ" --trajectories-per-card 1 \
    --binder-lengths "$LEN" "$LEN"
  echo "=== rc=$? ==="
  date -u +'end %Y-%m-%dT%H:%M:%SZ'
} >> "$LOG" 2>&1
