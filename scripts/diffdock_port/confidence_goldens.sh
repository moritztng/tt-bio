#!/usr/bin/env bash
# Same environment as dfd-ref's run_goldens.sh: upstream's CWD (so3/torus cache .npy
# tables there at import), its torch hub, a fixed hash seed.
set -euo pipefail
D=~/scratch/dfd
export TORCH_HOME=$D/torchhub
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export PYTHONHASHSEED=0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$D/upstream"
exec "$D/venv-ref/bin/python" -u "$HERE/confidence_goldens.py" "$@"
