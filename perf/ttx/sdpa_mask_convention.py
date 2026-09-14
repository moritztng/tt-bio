#!/usr/bin/env python3
"""Which `attn_mask` convention does each ttnn's SDPA implement?

tt-bio pre-multiplies the pair bias by `sqrt(head_dim)` (`_bias_scale`, `tenstorrent.py:7578`)
and then passes `scale=head_dim**-0.5` to `ttnn.transformer.scaled_dot_product_attention`. That
only comes out at 1x if the op scales the mask together with QK^T. Torch's own
`scaled_dot_product_attention`, which the ttnn docstring says this API mimics, does the opposite:
it scales QK^T and adds `attn_mask` afterwards.

Scores each arm's real output against BOTH float64 references, each built from that arm's own
operands, dumped by `op_trace.py --dump-args`.

    sdpa_mask_convention.py <dumpdir with one subdir per ttnn version>
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
SCALE = 0.17677669529663687          # head_dim**-0.5 at head_dim 32, the value the fold passes
CALL = 346                           # the first call whose cross-stack L2 step is not rounding


def main() -> int:
    base = Path(sys.argv[1])
    arms = sorted(p.name for p in base.iterdir() if p.is_dir())
    for arm in arms:
        def L(n):
            return torch.from_numpy(np.load(base / arm / f"arg{CALL}_{n}.npy")).double()
        q, k, v, m, o = (L(n) for n in ("in0", "in1", "in2", "kw_attn_mask", "out0"))
        s = q @ k.transpose(-1, -2)
        refs = {
            "mask after scale (torch, and the docstring)": torch.softmax(s * SCALE + m, -1) @ v,
            "mask inside scale (what tt-bio assumes)": torch.softmax((s + m) * SCALE, -1) @ v,
        }
        print(f"=== ttnn {arm}")
        for name, r in refs.items():
            pcc = float(torch.corrcoef(torch.stack([o.flatten(), r.flatten()]))[0, 1])
            print(f"   vs {name:44s} rel_l2 {float((o - r).norm() / r.norm()):.6e}  pcc {pcc:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
