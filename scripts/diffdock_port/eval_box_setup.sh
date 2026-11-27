#!/usr/bin/env bash
# Stand up a rented CPU box for the DFD graded accuracy run, then dock PoseBusters on it.
#
#   scripts/diffdock_port/eval_box_setup.sh <ssh-target> [workers] [threads]
#   SSH_CMD='ssh -p 12345 root@1.2.3.4' scripts/diffdock_port/eval_box_setup.sh - [workers] [threads]
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

if [ "$WORKERS" = auto ]; then
  # From the cgroup QUOTA, not nproc. A vast box reports the whole host's core count: instance
  # 62771089 answered `nproc` with 128 while /sys/fs/cgroup/cpu.max said 3071999/100000, i.e. the
  # 30.7 cores we actually rented. Sizing on nproc started 42 workers x 3 threads on 30.7 cores,
  # a 4x oversubscription that left every worker running at 0.73 of a core and put the 428-complex
  # set on a 60 h track (2026-11-27). Quota first, nproc only if there is no quota.
  NP=$($SSH 'q=$(cut -d" " -f1 /sys/fs/cgroup/cpu.max 2>/dev/null); p=$(cut -d" " -f2 /sys/fs/cgroup/cpu.max 2>/dev/null)
             case "$q" in ""|max) nproc;; *) echo $(( q / p ));; esac')
  WORKERS=$(( NP / THREADS )); [ "$WORKERS" -lt 1 ] && WORKERS=1
  echo "== $NP cores of quota -> $WORKERS workers x $THREADS threads"
fi

echo "== base packages on $RT"
$SSH 'command -v rsync >/dev/null && command -v curl >/dev/null ||
      (export DEBIAN_FRONTEND=noninteractive; apt-get -qq update &&
       apt-get -qq install -y rsync curl ca-certificates >/dev/null)'

echo "== shipping to $RT"
# Everything but perf/, which is 772 of the tree's 921 MB and holds nothing the dock imports.
PATHS=$(git -C "$REPO" ls-tree --name-only HEAD | grep -vx perf)
# shellcheck disable=SC2086
git -C "$REPO" archive --format=tar HEAD $PATHS | $SSH "mkdir -p ~/dfd/tt-bio && tar -x -C ~/dfd/tt-bio"
# One tar stream per tree, not rsync. PoseBusters is 428 directories of small files, and the
# 2026-11-26 run to a Texas box spent its first ~3 minutes on the file list before any bytes
# moved; a tar stream starts sending immediately and needs no per-file handshake. Neither is
# resumable here, which is fine: the dock itself is.
tar -C "$(dirname "$PB")" -chf - "$(basename "$PB")" | $SSH "mkdir -p ~/dfd/pb && tar -xf - -C ~/dfd/pb"
tar -C "$CKPT" -chf - . | $SSH "mkdir -p ~/.boltz/diffdock/models && tar -xf - -C ~/.boltz/diffdock/models"

echo "== environment"
$SSH bash -s <<'REMOTE'
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/dfd/venv ] || uv venv --python 3.11 ~/dfd/venv
VIRTUAL_ENV=~/dfd/venv uv pip install --quiet \
    torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=~/dfd/venv uv pip install --quiet ~/dfd/tt-bio
# From $HOME, so the install is what gets imported and not the shipped source tree. This is
# the check that caught the package-data gap on 2026-11-26: recipe.yaml, w3j_tables.json and
# score_norm_tables.json are loaded by path at import time and were not in the wheel.
cd ~ && ~/dfd/venv/bin/python -c "import tt_bio.diffdock.run; print('tt_bio ok')"
REMOTE

echo "== dock, detached"
# `timeout`, because this ssh does not come back: the remote launch writes its pid and exits, but
# sshd on these images holds the channel open and the launcher blocked for 55 min on 2026-11-27,
# so nothing pulled poses and nothing watched the cap. The dock is detached by construction, so
# the launch does not need its channel back; it needs the pid, which a second ssh reads.
timeout 120 $SSH "cd ~/dfd/tt-bio && DFD_PB=~/dfd/pb/posebusters_benchmark_set PYTHONHASHSEED=0 \
      PYTHONUNBUFFERED=1 setsid nohup ~/dfd/venv/bin/python scripts/diffdock_port/pose_eval.py dock \
      --ids all --shard ${SHARD:-0/1} --out ~/dfd/poses --workers $WORKERS --threads $THREADS --seed 42 \
      > ~/dfd/dock.log 2>&1 < /dev/null & echo \$! > ~/dfd/dock.pid"
timeout 60 $SSH 'kill -0 $(cat ~/dfd/dock.pid) && echo "dock pid $(cat ~/dfd/dock.pid) alive"' \
  || { echo "dock did not start"; exit 1; }
echo "== log: $RT:~/dfd/dock.log  poses: $RT:~/dfd/poses"
