#!/usr/bin/env bash
# Stand up a CPU host for the DFD graded accuracy run, then dock PoseBusters on it.
#
#   scripts/diffdock_port/eval_box_setup.sh <ssh-target> [workers] [threads]
#   SSH_CMD='ssh -p 12345 root@1.2.3.4' scripts/diffdock_port/eval_box_setup.sh - [workers] [threads]
#   DFD_ROOT=~/dfd-eval scripts/diffdock_port/eval_box_setup.sh anx-glx 12 2
#
# The host can be a rented box or a fleet host with idle cores. DFD_ROOT keeps the whole run in
# one directory so it cannot collide with another row's tree on a shared host, and the
# checkpoint ship is skipped when the host already carries the two model directories.
#
# Ships this branch, the PoseBusters benchmark set and the released v1.1 checkpoints, builds
# one pinned CPU environment, and starts `pose_eval.py dock` detached under nohup. The dock is
# resumable: re-running the script picks up the complexes without a non-empty rank1.sdf.
#
# Nothing here opens a Tenstorrent device: this is the host-torch half of the grade, and the
# box exists only because 428 complexes is tens of CPU-hours that pc cannot spare.
set -euo pipefail

TARGET=${1:?usage: eval_box_setup.sh <ssh-target>|- [workers] [threads]}   # SHARD=i/n splits the set
WORKERS=${2:-8}   # `auto` sizes the pool from the box's own nproc, so one query can take
THREADS=${3:-3}   # a 32-core and a 64-core offer without leaving half the box idle.
# A vast box is reached as `ssh -p <port> root@<ip>`, not as a bare target, so the caller may
# hand the whole command over SSH_CMD. rsync gets the same command minus its target.
SSH=${SSH_CMD:-ssh -o StrictHostKeyChecking=accept-new $TARGET}
RT=${SSH##* }    # the target is the last word of the command
RSH=${SSH% *}    # ... and the rest is what rsync runs as its shell
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PB=${DFD_PB:-$HOME/scratch/dfd/pb/posebusters_benchmark_set}
CKPT=$HOME/.boltz/diffdock/models
ROOT=${DFD_ROOT:-\~/dfd}   # a remote path, expanded by the remote shell, not by this one

if [ "$WORKERS" = auto ]; then
  # From the cgroup QUOTA, not nproc. A vast box reports the whole host's core count: instance
  # 62771089 answered `nproc` with 128 while /sys/fs/cgroup/cpu.max said 3071999/100000, i.e. the
  # 30.7 cores we actually rented. Sizing on nproc started 42 workers x 3 threads on 30.7 cores,
  # a 4x oversubscription that left every worker running at 0.73 of a core and put the 428-complex
  # set on a 60 h track (2026-11-27). Quota first, nproc only if there is no quota.
  NP=$($SSH 'q=$(cut -d" " -f1 /sys/fs/cgroup/cpu.max 2>/dev/null); p=$(cut -d" " -f2 /sys/fs/cgroup/cpu.max 2>/dev/null)
             case "$q" in ""|max) nproc;; *) echo $(( q / p ));; esac')
  WORKERS=$(( NP / THREADS )); [ "$WORKERS" -lt 1 ] && WORKERS=1
  # ... and capped by RAM. A worker holds the score model, the confidence model and its graphs:
  # 2.1 GB resident, measured across the 43 workers on 62771089. An OOM kill costs the complexes
  # that worker held, so leave 8 GB of headroom and take the smaller of the two limits.
  GB=$($SSH 'awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo')
  RW=$(( (GB - 8) / 3 )); [ "$RW" -lt 1 ] && RW=1
  [ "$RW" -lt "$WORKERS" ] && { echo "== RAM-limited: $GB GB available -> $RW workers, not $WORKERS"; WORKERS=$RW; }
  echo "== $NP cores of quota, $GB GB -> $WORKERS workers x $THREADS threads"
fi

echo "== base packages on $RT"
$SSH 'command -v curl >/dev/null || [ "$(id -u)" != 0 ] ||
      (export DEBIAN_FRONTEND=noninteractive; apt-get -qq update &&
       apt-get -qq install -y curl ca-certificates >/dev/null)'

echo "== shipping to $RT"
# Everything but perf/, which is 772 of the tree's 921 MB and holds nothing the dock imports.
PATHS=$(git -C "$REPO" ls-tree --name-only HEAD | grep -vx perf)
# shellcheck disable=SC2086
git -C "$REPO" archive --format=tar HEAD $PATHS | $SSH "mkdir -p $ROOT/tt-bio && tar -x -C $ROOT/tt-bio"
# One tar stream per tree, not rsync. PoseBusters is 428 directories of small files, and the
# 2026-11-26 run to a Texas box spent its first ~3 minutes on the file list before any bytes
# moved; a tar stream starts sending immediately and needs no per-file handshake. Neither is
# resumable here, which is fine: the dock itself is.
$SSH "[ -d $ROOT/pb/$(basename "$PB") ]" \
  || tar -C "$(dirname "$PB")" -chf - "$(basename "$PB")" \
     | $SSH "mkdir -p $ROOT/pb && tar -xf - -C $ROOT/pb"
# The checkpoints are 134 MB and some fleet hosts already carry them from another DFD row; ship
# them only when both model directories are missing, and never overwrite a host's own copy.
$SSH '[ -d ~/.boltz/diffdock/models/score_model ] && [ -d ~/.boltz/diffdock/models/confidence_model ]' \
  || tar -C "$CKPT" -chf - . | $SSH "mkdir -p ~/.boltz/diffdock/models && tar -xf - -C ~/.boltz/diffdock/models"

echo "== environment"
$SSH "ROOT=$ROOT bash -s" <<'REMOTE'
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d $ROOT/venv ] || uv venv --python 3.11 $ROOT/venv
VIRTUAL_ENV=$ROOT/venv uv pip install --quiet \
    torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=$ROOT/venv uv pip install --quiet $ROOT/tt-bio
# From $HOME, so the install is what gets imported and not the shipped source tree. This is
# the check that caught the package-data gap on 2026-11-26: recipe.yaml, w3j_tables.json and
# score_norm_tables.json are loaded by path at import time and were not in the wheel.
cd ~ && $ROOT/venv/bin/python -c "import tt_bio.diffdock.run; print('tt_bio ok')"
REMOTE

echo "== dock, detached"
# `timeout`, because this ssh does not come back: the remote launch writes its pid and exits, but
# sshd on these images holds the channel open and the launcher blocked for 55 min on 2026-11-27,
# so nothing pulled poses and nothing watched the cap. The dock is detached by construction, so
# the launch does not need its channel back; it needs the pid, which a second ssh reads.
timeout 120 $SSH "cd $ROOT/tt-bio && DFD_PB=$ROOT/pb/posebusters_benchmark_set PYTHONHASHSEED=0 \
      PYTHONUNBUFFERED=1 setsid nohup $ROOT/venv/bin/python scripts/diffdock_port/pose_eval.py dock \
      --ids all --shard ${SHARD:-0/1} --out $ROOT/poses --workers $WORKERS --threads $THREADS --seed 42 \
      > $ROOT/dock.log 2>&1 < /dev/null & echo \$! > $ROOT/dock.pid"
timeout 60 $SSH "kill -0 \$(cat $ROOT/dock.pid) && echo \"dock pid \$(cat $ROOT/dock.pid) alive\"" \
  || { echo "dock did not start"; exit 1; }
echo "== log: $RT:$ROOT/dock.log  poses: $RT:$ROOT/poses"
