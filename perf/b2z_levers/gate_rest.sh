#!/usr/bin/env bash
# The cached structure legs across the rest of the fleet. The two DST-granularity levers live in
# sdpa_generic.py and reblock_permute.py, which every model reaches through tenstorrent.py, so
# "bit-exact by torch.equal" deserves in-situ evidence and not just a unit claim.
set -u
cd /home/ttuser/.coworker/wt/b2z-levers-default-on
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export ESM_ROOT=/home/ttuser/esm
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:b2z-levers-default-on
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py \
  --workdir perf/b2z_levers/gate_work \
  --out perf/b2z_levers/gate_rest.json \
  --workers localhost:0 \
  --leg protenix-ubq-msa --leg protenix-hsa-msa --leg protenix-v1-prot-msa \
  --leg openfold3-ubq-msa --leg openfold3-prot-msa --leg openfold3-8hel-nomsa \
  --leg openfold3-8hel-msa --leg openfold3-7xi5-notmpl --leg openfold3-7xi5-tmpl \
  --leg openfold3-9bk6-complex-msa \
  --leg openbind-ubq-msa --leg openbind-fkg-ligand-msa \
  --leg opendde-trpcage-nomsa --leg opendde-prot-prod
