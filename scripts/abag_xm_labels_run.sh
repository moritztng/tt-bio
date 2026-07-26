#!/usr/bin/env bash
# Phase 4 label driver wrapper.
#
# Runs scripts/abag_xm_labels.py with a label-venv that has tmtools installed,
# while inheriting the shared tt-bio venv's site-packages (DockQ, gemmi, numpy)
# via PYTHONPATH. This unblocks the MANDATORY PSS / basin_clust labels (D5),
# which need tmtools. cdr_rmsd (anarci + hmmer, needs sudo) remains a separate
# gap and fails cleanly via labels.py's _error capture.
#
# Fallback: if the label-venv is absent, run with the shared venv python so the
# four tmtools-free labels (dockq, epitope_jaccard, interface_lddt, pae_metrics)
# still run.
#
# Usage: scripts/abag_xm_labels_run.sh <results_dir> <native.cif> <fold.yaml> [--n_samples N] [--out labels.json]

set -euo pipefail
WT="$(cd "$(dirname "$0")/.." && pwd)"
SHARED_VENV=/home/ttuser/tt-bio-dev/env
LABEL_VENV="$HOME/.abag_xm_label_venv"

if [ -x "$LABEL_VENV/bin/python3" ]; then
    PY="$LABEL_VENV/bin/python3"
    # Inherit shared venv site-packages for DockQ/gemmi/numpy.
    SP=$(ls -d "$SHARED_VENV"/lib/python*/site-packages 2>/dev/null | head -1)
    export PYTHONPATH="${SP:-}:${PYTHONPATH:-}"
else
    PY="$SHARED_VENV/bin/python3"
fi

exec "$PY" "$WT/scripts/abag_xm_labels.py" "$@"
