#!/usr/bin/env python3
"""D174's break control at the op, on CPU: the mask must do exactly three things and no fourth.

  1. an ALL-ONES mask reproduces the unmasked call BIT-IDENTICALLY -- if it does not, the lever
     is not the mask and nothing measured with it means what it says;
  2. the pair mask leaves every REAL cell bit-identical;
  3. and zeroes every PAD cell.

Run on `tt_bio/reference.py`'s `Transition`, which is `boltz2.Transition`, the shared torch one
the device class mirrors. `fc3` is zeroed by `init.final_init_` and the LayerNorm bias by
`init.bias_init_zero_`, so a freshly constructed transition returns 0 on a zero pad row whatever
the mask does; both are given trained-like values first or this control passes vacuously.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/home/ttuser/.coworker/wt/of3t-ditmodel")
os.environ["TT_BIO_MASK_TRANS"] = "1"
import tt_bio.reference as ref  # noqa: E402

torch.manual_seed(0)
tr = ref.Transition(16, 64).eval()
torch.manual_seed(2)
tr.norm.bias.data.normal_(0.0, 0.1)
tr.fc3.weight.data.normal_(0.0, 0.02)

torch.manual_seed(1)
N = 8
x = torch.randn(1, N, N, 16)
m = torch.zeros(1, N)
m[:, :4] = 1.0
pm = m[:, :, None] * m[:, None, :]
x = x * pm[..., None]
real = pm[0] > 0

with torch.no_grad():
    plain = tr(x)
    masked = tr(x, mask=pm.unsqueeze(-1))
    ones = tr(x, mask=torch.ones_like(pm).unsqueeze(-1))


def n(t):
    return float(torch.linalg.vector_norm(t))


out = {
    "what": __doc__.strip().splitlines()[0],
    "ones_mask_reproduces_unmasked_bit_identically": bool(torch.equal(plain, ones)),
    "ones_mask_absdiff": n(plain - ones),
    "pair_mask_real_cells_bit_identical": bool(torch.equal(plain[0][real], masked[0][real])),
    "pair_mask_real_cells_absdiff": n(plain[0][real] - masked[0][real]),
    "pad_cells_norm_unmasked": n(plain[0][~real]),
    "pad_cells_norm_masked": n(masked[0][~real]),
}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
