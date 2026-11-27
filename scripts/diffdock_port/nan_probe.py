"""Which ttnn op removes a NaN? `where` appears to compute both lanes, so a NaN in the branch
not taken still reaches the output. One short run on a card settles the guard's implementation.
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    x = torch.tensor([[float("nan")], [float("inf")], [float("-inf")], [2.0], [-3.0]] * 8)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    zero = ttnn.multiply(t, 0.0)
    one = ttnn.add(zero, 1.0)
    finite = ttnn.isfinite(t)

    cases = {
        "where_false_lane_is_clean": lambda: ttnn.where(finite, t, one),
        "clamp": lambda: ttnn.clamp(t, min=-9.0, max=9.0),
        "maximum_then_minimum": lambda: ttnn.minimum(ttnn.maximum(t, ttnn.multiply(one, -9.0)),
                                                     ttnn.multiply(one, 9.0)),
        "multiply_by_zero": lambda: ttnn.multiply(t, 0.0),
        "div_no_nan_by_one": lambda: ttnn.div_no_nan(t, one),
        "where_on_isnan": lambda: ttnn.where(ttnn.isnan(t), one, t),
        "sign": lambda: ttnn.sign(t),
    }
    out = {}
    for name, fn in cases.items():
        try:
            got = ttnn.to_torch(fn())[:5, 0]
        except Exception as e:                      # an op may not take these operands at all
            out[name] = f"raised {type(e).__name__}"
        else:
            out[name] = [None if torch.isnan(v) else float(v) for v in got]
        print(name, out[name], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
