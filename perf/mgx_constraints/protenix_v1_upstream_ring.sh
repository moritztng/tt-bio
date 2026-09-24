#!/bin/bash
# Does upstream Protenix v0.5.0 (the protenix-v1 checkpoint) close a cyclic peptide? Our
# features (feat_parity.py --bonds: 23/24 EXACT against v0.5.0's pipeline; token_bonds carries
# the ring bond, as current upstream's include_discont_poly_poly_bonds=True inference does)
# folded by upstream's own trunk + sample_diffusion on CPU, 5 samples, 200 steps.
set -eu
cd "$(dirname "$0")/../.."
T=${TMPDIR:-/tmp}/pv1ring; mkdir -p "$T"
export PROTENIX_DATA_ROOT_DIR=/home/moritz/.coworker/protenix-ref-data TT_VISIBLE_DEVICES=
for m in cyclic linear13; do
  PYTHONPATH=$PWD /home/moritz/tt-bio/env/bin/python perf/mgx_constraints/protenix_ring_feats.py \
      perf/mgx_constraints/inputs/$m.yaml "$T/$m.pt"
  ~/protenix05_ref_venv/bin/python scripts/protenix_v1_port/harvest_ref_fold.py \
      --feats "$T/$m.pt" --seeds 0 --steps 200 --samples 5 --out "$T/ref_$m" --force
done
~/protenix05_ref_venv/bin/python - "$T" <<'Q'
import sys, torch
for m in ("cyclic", "linear13"):
    b = torch.load(f"{sys.argv[1]}/{m}.pt", weights_only=False)
    r = torch.load(f"{sys.argv[1]}/ref_{m}/seed0/raw.pt", weights_only=False)
    x = r["coords"].reshape(-1, r["n_atom"], 3)
    print(m, "N1-C", [round(float((s[b["i_n"]] - s[b["i_c"]]).norm()), 2) for s in x])
Q
