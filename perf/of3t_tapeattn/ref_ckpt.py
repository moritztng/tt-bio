#!/usr/bin/env python3
"""`of3t-bwdaccum/ref_cot.py` with the reference stack's blocks activation-checkpointed.

Not a copy of it. The one thing that had to change at padded 384 is memory: the plain float64
arm keeps every block's internal activations live, and it was OOM-killed twice on qb1 at 318 GB
and 494 GB of a 503 GB box. `of3t-frame384` hit the same wall and answered it with ref_grad's
`--checkpoint`, which `ref_cot.py` never exposed. So the flag is applied where it belongs, on
upstream's own `PairFormerBlock.forward`, and `ref_cot.main()` runs unmodified underneath.

Activation checkpointing recomputes a block's forward during its backward. It changes the
ORDER of nothing and the arithmetic of nothing in float64, and the quantity this row reads --
the cotangent at each block boundary -- is a block OUTPUT, which `ref_cot` retains either way.
The c64 pair is built both ways and compared, so "inert" is a reading and not a claim.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "of3t_bwdaccum"))

import torch                                                          # noqa: E402
from torch.utils.checkpoint import checkpoint                         # noqa: E402


def main() -> int:
    tree = sys.argv[sys.argv.index("--tree") + 1]
    sys.path.insert(0, tree)
    from openfold3.core.model.latent.pairformer import PairFormerBlock

    real = PairFormerBlock.forward
    STATE = {"checkpointed": 0}

    def forward(self, *a, **kw):
        if not torch.is_grad_enabled() or not any(
                isinstance(x, torch.Tensor) and x.requires_grad for x in a):
            return real(self, *a, **kw)
        STATE["checkpointed"] += 1
        return checkpoint(real, self, *a, use_reentrant=False, **kw)

    PairFormerBlock.forward = forward
    import ref_cot
    rc = ref_cot.main()
    print(f'{{"checkpointed_block_calls": {STATE["checkpointed"]}}}')
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
