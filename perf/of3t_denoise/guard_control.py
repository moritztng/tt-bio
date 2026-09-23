#!/usr/bin/env python3
"""of3t-denoise: the registration guard must fire on the pre-fix tree.

    guard_control.py --batch B.pt --out F.json

The pre-fix tree is this commit with `OF3AtomTransformer.materialize_device_weights` made a
no-op, which is exactly what `OpenFold3Forward.model` did before D256 was fixed: the diffusion
atom transformers upload lazily, inside the forward. One forward of the adapter; the guard
(`check_registered`, at the end of `__call__`) must raise and name those tensors. A guard that
cannot fail has tested nothing.
"""
import json
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    batch, out = Path(argv[argv.index("--batch") + 1]), Path(argv[argv.index("--out") + 1])
    from tt_bio.openfold3_atom_transformer import OF3AtomTransformer
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward
    OF3AtomTransformer.materialize_device_weights = lambda self: 0
    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", seed=20260922)
    rec = {"batch": str(batch), "tree": "materialize_device_weights disabled (pre-D256)",
           "denoise": fwd.denoise}
    ds = OpenFold3Dataset(batch)
    try:
        fwd(ds.batch([0]))
        rec["fired"] = False
    except RuntimeError as e:
        rec["fired"] = True
        rec["message"] = str(e)
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import walk_device_weights
    late = [p for p, _o, _k, t in walk_device_weights(fwd.model) if ag.parameter_for(t) is None]
    rec["unregistered"] = {"n": len(late), "paths": late}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("fired", "denoise")} | {"n": len(late)}), flush=True)
    return 0 if rec["fired"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
