#!/usr/bin/env python3
"""Which wheel `*_bw` ops accept the dtypes tt_bio's tape actually carries.

The 0.68.0 docstrings list BFLOAT16/BFLOAT8_B for every one of them and tt_bio's
`add_grad` typecasts contributions to fp32, so the docstring alone decides nothing.
This runs each candidate on the card at fp32 and bf16 and records what happened.
"""
import json
import pathlib
import sys

import numpy as np
import ttnn

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/of3t/of3t-wheelbw/dtypes.json")

CAND = {
    "mul_bw":     lambda g, a, b: ttnn.mul_bw(g, a, b),
    "add_bw":     lambda g, a, b: ttnn.add_bw(g, a, b),
    "relu_bw":    lambda g, a, b: ttnn.relu_bw(g, a),
    "sigmoid_bw": lambda g, a, b: ttnn.sigmoid_bw(g, a),
    "silu_bw":    lambda g, a, b: ttnn.silu_bw(g, a),
    "concat_bw":  lambda g, a, b: ttnn.concat_bw(g, a, b, 0),
}


def main() -> int:
    dev = ttnn.open_device(device_id=0)
    res = {}
    try:
        for dt_name, dt in (("float32", ttnn.float32), ("bfloat16", ttnn.bfloat16)):
            rng = np.random.default_rng(0)
            np_dt = np.float32
            a = rng.standard_normal((64, 64)).astype(np_dt)
            b = rng.standard_normal((64, 64)).astype(np_dt)
            g = rng.standard_normal((64, 64)).astype(np_dt)
            mk = lambda v: ttnn.from_torch_or_numpy(v) if False else ttnn.Tensor(
                v, dt).to(ttnn.TILE_LAYOUT).to(dev)
            ta, tb, tg = mk(a), mk(b), mk(g)
            # concat_bw needs a grad of the CONCATENATED shape
            gcat = mk(rng.standard_normal((128, 64)).astype(np_dt))
            for name, fn in CAND.items():
                key = f"{name}|{dt_name}"
                try:
                    out = fn(gcat if name == "concat_bw" else tg, ta, tb)
                    n = len(out) if isinstance(out, (list, tuple)) else 1
                    res[key] = {"ok": True, "outputs": n,
                                "out_dtype": str((out[0] if n > 1 or isinstance(out, (list, tuple))
                                                  else out).dtype)}
                except Exception as e:
                    res[key] = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:300]}
                print(key, res[key].get("ok"), res[key].get("error", "")[:160], flush=True)
        # ttnn.pad, the candidate for narrow's zero-padding backward
        for dt_name, dt in (("float32", ttnn.float32), ("bfloat16", ttnn.bfloat16)):
            rng = np.random.default_rng(1)
            v = rng.standard_normal((64, 64)).astype(np.float32)
            t = ttnn.Tensor(v, dt).to(ttnn.TILE_LAYOUT).to(dev)
            for label, padding in (("lead_32_32", [(32, 32), (0, 0)]),
                                   ("lead_5_3", [(5, 3), (0, 0)])):
                key = f"pad|{dt_name}|{label}"
                try:
                    o = ttnn.pad(t, padding=padding, value=0.0)
                    host = ttnn.to_torch(o).float().numpy() if hasattr(ttnn, "to_torch") else None
                    res[key] = {"ok": True, "shape": list(o.shape)}
                    if host is not None:
                        lo = padding[0][0]
                        res[key]["zeros_above"] = float(np.abs(host[:lo]).max()) if lo else 0.0
                        res[key]["body_exact"] = bool(
                            np.array_equal(host[lo:lo + 64], ttnn.to_torch(t).float().numpy()))
                except Exception as e:
                    res[key] = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:300]}
                print(key, res[key], flush=True)
    finally:
        ttnn.close_device(dev)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
