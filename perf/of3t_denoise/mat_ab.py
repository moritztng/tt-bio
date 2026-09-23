#!/usr/bin/env python3
"""of3t-denoise: does materialising the atom transformers' weights change the forward?

    mat_ab.py lazy|eager --out F.json

One adapter forward at 64 tokens, no backward, the guard muted. `lazy` makes
`materialize_device_weights` a no-op (the pre-D256 tree); `eager` is the shipped fix. Records the
rollout's final coordinates, the denoise arm's pred_xyz and every output's sum.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    mode, out = sys.argv[1], Path(sys.argv[sys.argv.index("--out") + 1])
    import ttnn
    from tt_bio.openfold3_atom_transformer import OF3AtomTransformer
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward
    if mode == "lazy":
        OF3AtomTransformer.materialize_device_weights = lambda self: 0
    OpenFold3Forward.check_registered = lambda self: None
    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", seed=20260922)
    ds = OpenFold3Dataset("/home/ttuser/of3t_fullstep64/batch_step003_t64.pt")
    o = fwd(ds.batch([0]))
    rec = {"mode": mode}
    xl = fwd.rollout_coords.numpy().astype(np.float64)
    rec["rollout_sha"] = hashlib.sha256(xl.tobytes()).hexdigest()[:16]
    rec["rollout_first"] = xl[:2].tolist()
    for k, v in o.items():
        a = ttnn.to_torch(v.value).double()
        rec[k] = {"sum": float(a.sum()), "abs": float(a.abs().sum())}
    at = fwd.model.sampler.dm.enc_at
    rec["wc"] = {str(k): [str(t.dtype), list(t.shape)] for k, t in list(at._wc.items())[:6]}
    rec["n_wc"] = len(at._wc)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("mode", "rollout_sha", "n_wc")}
                     | {"pred_xyz": rec.get("pred_xyz"), "pae": rec["pae_logits"]}), flush=True)


if __name__ == "__main__":
    main()
