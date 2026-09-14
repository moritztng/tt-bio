#!/usr/bin/env bash
# Fresh full parity gate on the re-merged tree. The merge of origin/main moved the code
# fingerprint (esmfold2_runtime.py + weights.py), so the pass-1/2 leg cache is not resumable
# against this tree and the workdir is namespaced to the merge sha.
# HF_HUB_OFFLINE=1: esmc-6b carries no revision pin in tt_bio/weights.py, so a fetch would move
# the ESMC-6B weights out from under esmfold2 (root-caused pass 2, still open as its own row).
set -u
WT=/home/ttuser/.coworker/wt/roof-gate-epilogue-conflict-remerge
cd "$WT" || exit 1
export HF_HUB_OFFLINE=1
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export TT_BIO_LEASE_CARDS=2
export TT_BIO_LEASE_HOLDER=worker:roof-gate-epilogue-conflict-remerge
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/full_parity_gate.py \
  --workers tt-quietbox2:2 \
  --workdir "$WT/perf/roof_gate_epilogue/gate-fd5c536" \
  --out "$WT/perf/roof_gate_epilogue/out/gate_fd5c536.json"
