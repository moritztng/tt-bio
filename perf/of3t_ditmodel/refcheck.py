#!/usr/bin/env python3
"""D174 on the torch reference, on CPU, in seconds: what can a REAL token see of the pad?

`tt_bio/reference.py`'s `PairformerLayer` shared our port's omission until this row, so it is the
cheapest place to ask the question the device arms take 425 s each to answer. One layer, one set
of weights, one input, the flag flipped between two forwards of the SAME object -- not two
constructions, because then weight init is a second variable.

Three readings:
  * A/A -- the same arm twice, which must be bit-identical before any difference below is read;
  * the op-level ones control, run separately in `opcheck.py`: `Transition` with an all-ones
    mask must reproduce the unmasked call bit-identically, and with the pair mask must leave
    every REAL cell bit-identical and zero every pad cell;
  * the arm itself, split into the REAL block and the PADDED region.

Two initialisation facts make or break this control, and the first two runs of it read 0.0
everywhere because neither was handled. `boltz2.Transition` initialises its LayerNorm bias to
ZERO and its fc3 with `init.final_init_`, which zeroes it. A freshly constructed transition
therefore returns exactly 0 on a zero pad row whatever the mask does, so the control confirms
nothing. The trained checkpoint is not like that: `of3-p2-155k.pt` carries 104 pairformer
transition `layer_norm` biases with norms from 1.11648 to 5.73926, median 2.15894. Both tensors
are given trained-like values here.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/home/ttuser/.coworker/wt/of3t-ditmodel")
os.environ.setdefault("TT_BIO_MASK_TRANS", "0")
import tt_bio.reference as ref  # noqa: E402

torch.manual_seed(0)
layer = ref.PairformerLayer(32, 16, num_heads=4, dropout=0.0,
                            pairwise_head_width=8, pairwise_num_heads=2).eval()
torch.manual_seed(2)
for tr in (layer.transition_z, layer.transition_s):
    tr.norm.bias.data.normal_(0.0, 0.1)
    tr.fc3.weight.data.normal_(0.0, 0.02)

torch.manual_seed(1)
N, R = 24, 10
s0 = torch.randn(1, N, 32)
z0 = torch.randn(1, N, N, 16)
m = torch.zeros(1, N)
m[:, :R] = 1.0
# A real batch reaches the trunk already zeroed on the pad, which is what makes the transition's
# own bias the only thing that can be nonzero there.
s0 = s0 * m[..., None]
pm = m[:, :, None] * m[:, None, :]
z0 = z0 * pm[..., None]
real1 = m.reshape(-1) > 0
real2 = pm[0] > 0


def arm(flag):
    """Flip the module-level gate between two forwards of the SAME layer object, so weight
    initialisation is not a second variable."""
    ref._MASK_TRANS = flag
    with torch.no_grad():
        return layer(s0, z0, m, pm)


def n(t):
    return float(torch.linalg.vector_norm(t))


off_s, off_z = arm(False)
off2_s, off2_z = arm(False)
on_s, on_z = arm(True)

out = {
    "what": __doc__.strip().splitlines()[0],
    "tokens": int(m.numel()), "real_tokens": int(real1.sum()),
    "A_over_A": {"s_bit_identical": bool(torch.equal(off_s, off2_s)),
                 "z_bit_identical": bool(torch.equal(off_z, off2_z))},
    "real_block": {
        "s_absdiff": n(on_s[0][real1] - off_s[0][real1]),
        "z_absdiff": n(on_z[0][real2] - off_z[0][real2]),
        "s_rel": n(on_s[0][real1] - off_s[0][real1]) / n(off_s[0][real1]),
        "z_rel": n(on_z[0][real2] - off_z[0][real2]) / n(off_z[0][real2]),
        "s_norm": n(off_s[0][real1]), "z_norm": n(off_z[0][real2]),
    },
    "padded_region": {
        "s_norm_unmasked": n(off_s[0][~real1]), "s_norm_masked": n(on_s[0][~real1]),
        "z_norm_unmasked": n(off_z[0][~real2]), "z_norm_masked": n(on_z[0][~real2]),
    },
}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
