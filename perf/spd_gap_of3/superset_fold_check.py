"""`_superset_device` against `_superset_host` inside a real fold: bench.py with every superset build done both ways.

    python perf/spd_gap_of3/superset_fold_check.py --log out/sfc.jsonl [--sync] -- <bench.py args>

Each build appends one line to --log (shapes, dtypes, bit-exactness, max abs difference) and the fold continues on
the device result, so its digest is the device path's. superset_check.py proves the two equal on random inputs;
this answers whether they are equal on the fold's own inputs, which is what decides why the fold digests differ.
The comparison reads tensors back, which synchronizes the device; --sync instead runs the device path alone followed
by ttnn.synchronize_device, to tell a sync-sensitive ordering bug from a value difference.
"""
import json, runpy, sys
from pathlib import Path

import torch

i = sys.argv.index("--")
log = Path(sys.argv[sys.argv.index("--log") + 1])
sync_only = "--sync" in sys.argv[:i]
sys.argv = ["perf/spd/bench.py", *sys.argv[i + 1:]]
log.parent.mkdir(parents=True, exist_ok=True)

import ttnn  # noqa: E402
from tt_bio.openfold3_atom_transformer import OF3AtomTransformer  # noqa: E402

_dev = OF3AtomTransformer._superset_device


def both(self, zb, mask_bias, off, W, nb):
    if sync_only:
        out = _dev(self, zb, mask_bias, off, W, nb)
        ttnn.synchronize_device(self.device)
        return out
    h = ttnn.to_torch(self._superset_host(zb, mask_bias, off, W, nb))
    out = _dev(self, zb, mask_bias, off, W, nb)
    d = ttnn.to_torch(out)
    rec = {"zb": [list(zb.shape), str(zb.dtype), str(zb.layout)],
           "mask": [list(mask_bias.shape), str(mask_bias.dtype)], "act": str(self._act_dtype),
           "out": [list(out.shape), str(out.dtype), str(out.memory_config().buffer_type)],
           "nb": nb, "W": W, "runs": len(set(off.tolist())), "finite": bool(torch.isfinite(d.float()).all()),
           "bit_exact": h.shape == d.shape and torch.equal(h, d)}
    if h.shape == d.shape:
        diff = (h.float() - d.float()).abs()
        rec |= {"n_diff": int((diff > 0).sum()), "max_abs": float(diff.max())}
    with log.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return out


OF3AtomTransformer._superset_device = both
runpy.run_path("perf/spd/bench.py", run_name="__main__")
