#!/usr/bin/env bash
# Stand up a rented CPU box for the DFD graded accuracy run, then dock PoseBusters on it.
#
#   scripts/diffdock_port/eval_box_setup.sh <ssh-target> [workers] [threads]
#
# Ships this branch, the PoseBusters benchmark set and the released v1.1 checkpoints, builds
# one pinned CPU environment, and starts `pose_eval.py dock` detached under nohup. The dock is
# resumable: re-running the script picks up the complexes without a non-empty rank1.sdf.
#
# Nothing here opens a Tenstorrent device: this is the host-torch half of the grade, and the
# box exists only because 428 complexes is tens of CPU-hours that pc cannot spare.
set -euo pipefail

TARGET=${1:?usage: eval_box_setup.sh <ssh-target> [workers] [threads]}
WORKERS=${2:-8}
THREADS=${3:-3}
SSH="ssh -o StrictHostKeyChecking=accept-new $TARGET"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PB=${DFD_PB:-$HOME/scratch/dfd/pb/posebusters_benchmark_set}
CKPT=$HOME/.boltz/diffdock/models

echo "== shipping to $TARGET"
git -C "$REPO" archive --format=tar HEAD | $SSH "mkdir -p ~/dfd/tt-bio && tar -x -C ~/dfd/tt-bio"
rsync -a --info=progress2 -L "$PB" "$TARGET:dfd/pb/"
rsync -a -L "$CKPT/" "$TARGET:.boltz/diffdock/models/"

echo "== environment"
$SSH bash -s <<'REMOTE'
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/dfd/venv ] || uv venv --python 3.11 ~/dfd/venv
VIRTUAL_ENV=~/dfd/venv uv pip install --quiet \
    torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=~/dfd/venv uv pip install --quiet ~/dfd/tt-bio
~/dfd/venv/bin/python -c "import tt_bio.diffdock.run; print('tt_bio ok')"
REMOTE

echo "== dock, detached"
$SSH "cd ~/dfd/tt-bio && DFD_PB=~/dfd/pb/posebusters_benchmark_set PYTHONHASHSEED=0 \
      PYTHONUNBUFFERED=1 nohup ~/dfd/venv/bin/python scripts/diffdock_port/pose_eval.py dock \
      --ids all --out ~/dfd/poses --workers $WORKERS --threads $THREADS --seed 42 \
      > ~/dfd/dock.log 2>&1 & echo \$! > ~/dfd/dock.pid; cat ~/dfd/dock.pid"
echo "== log: $TARGET:~/dfd/dock.log  poses: $TARGET:~/dfd/poses"
