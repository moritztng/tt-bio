#!/usr/bin/env python3
"""Does upstream RF3 close a cyclic peptide? The torch reference on CPU, 5 samples, N1-C13.

    TT_VISIBLE_DEVICES= python perf/mgx_constraints/rf3_upstream_ring.py <rf3 ckpt>

Folds inputs/cyclic.yaml's 13-mer with and without upstream's cyclic_chains and prints the
N-to-C distance per sample. A closed amide is 1.33 A; score.py calls a ring closed below 2.0 A.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.rf3.featurize import featurize, n_cycle, network_input  # noqa: E402
from tt_bio.rf3.weights import load_reference  # noqa: E402

SEQ = "QLEDSEVEAVAKG"
net, _ = load_reference(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    spec = Path(td) / "cyc.json"
    spec.write_text(json.dumps([{"name": "cyc", "components": [{"seq": SEQ, "chain_id": "A"}]}]))
    for mode, cyclic in (("cyclic", ["A"]), ("linear", None)):
        out = featurize(spec, n_recycles=10, diffusion_batch_size=5, seed=0,
                        cyclic_chains=cyclic)[0]
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            res = net(input=network_input(out), n_cycle=min(10, n_cycle(out)),
                      coord_atom_lvl_to_be_noised=out["coord_atom_lvl_to_be_noised"])
        aa, x = out["atom_array"], res["X_pred_rollout_L"].float()
        i_n = np.flatnonzero((aa.res_id == aa.res_id.min()) & (aa.atom_name == "N"))[0]
        i_c = np.flatnonzero((aa.res_id == aa.res_id.max()) & (aa.atom_name == "C"))[0]
        print(json.dumps({"mode": mode, "n_c": [round(float((s[i_n] - s[i_c]).norm()), 2)
                                                for s in x]}))
