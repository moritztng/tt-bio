#!/bin/bash
# The charter's merge gate for this row: one on-hardware accuracy leg on the branch tree, after
# the design campaign gives the card back. The branch changes host-side campaign input checking
# only, so this is here to show the tree it lands on still folds, not to grade the fix.
set -uo pipefail
cd "$(dirname "$0")/../.."
for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
echo "=== card free $(date -u +%H:%M:%SZ)"
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bgx-inputs
echo "=== release_gate boltz2 --fast $(date -u +%H:%M:%SZ)"
timeout 3000 /home/ttuser/bcx_e2e_venv/bin/python3 -u scripts/release_gate.py --model boltz2 --fast
echo "=== release_gate rc=$? $(date -u +%H:%M:%SZ)"
