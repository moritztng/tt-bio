#!/usr/bin/env bash
# The grade runs on the PORT alone: plain torch, no e3nn, no upstream clone. It reads the
# goldens confidence_goldens.sh captured.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
PYTHONPATH="$REPO" exec ~/scratch/dfd-cli/venv/bin/python -u \
  "$HERE/confidence_parity.py" "$@"
