#!/usr/bin/env bash
# The full parity gate with TT_BIO_ATOM_KEY_WINDOW on by default (wk/b2z2-akw-ship).
#
# The window replaces a one-hot matmul in AttentionPairBias.single_to_keys, which every
# atom-attention model in the tree reaches through tenstorrent.py, so the scope is the whole
# gate and not just the boltz2 legs. Resumable: each finished leg is cached under
# perf/b2z2_gate/akw_work/<leg>.json and reused on relaunch.
#
# One card, physical 0, pinned. protenix-9ncy-msa is BLOCKED-REF-REGEN-NEEDED before this row
# touched anything: the release asset ships its provenance JSON without the reference CIFs.
set -u
cd /home/ttuser/.coworker/wt/b2z2-union-gate-ship
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export ESM_ROOT=/home/ttuser/esm
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:b2z2-union-gate-ship
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py \
  --workdir perf/b2z2_gate/akw_work \
  --out perf/b2z2_gate/gate_akw.json \
  --workers localhost:0
