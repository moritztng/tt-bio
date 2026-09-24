"""Compare the logit gradient two trees hand BindCraft 2 at the same step, in float64.

Bit identity is checked first because it is free; the float64 relative L2 and the largest
absolute difference are reported beside it so a bf16-rounding difference reads as a size.
"""
import json
import sys

import numpy as np


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    out = {}
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            out[k] = {"missing_in": "a" if k not in a.files else "b"}
            continue
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        d = x - y
        out[k] = {"shape": list(x.shape),
                  "bit_identical": bool(np.array_equal(a[k].view(np.uint32), b[k].view(np.uint32))),
                  "max_abs_diff": float(np.abs(d).max()) if d.size else 0.0,
                  "rel_l2_f64": float(np.linalg.norm(d) / max(np.linalg.norm(y), 1e-300)),
                  "norm_a": float(np.linalg.norm(x)), "norm_b": float(np.linalg.norm(y))}
    return out


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=1))
