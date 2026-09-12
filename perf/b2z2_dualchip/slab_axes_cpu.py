"""Which axis each Pairformer op's output row lives on, checked in the torch reference. NO DEVICE.

The device check (`slab_bitexact.py`) can only say pass or fail. This one says WHY, and it runs on
any machine with the package importable, which makes it the thing to run first when a slab comes
back wrong: if these five lines are exact, the axes are right and the failure is a kernel-routing
difference, not the algebra.

The claims, one per op, each an identity over the reference's own forward:

  trimul start   out[i,j] = sum_k a[i,k] b[j,k]   -> i is a's FIRST axis:  a takes a row slab
  trimul end     out[i,j] = sum_k a[k,i] b[k,j]   -> i is a's SECOND axis: a takes a column slab
  triatt start   attends within row i             -> i is the attention's BATCH axis
  triatt end     attends over i for fixed j       -> after the transpose i is the QUERY axis, so
                                                     q and the bias's query axis take the slab and
                                                     k/v keep every i
  transition     elementwise per (i,j)            -> the slab is a smaller input

`b` in both triangle products, and k/v in the ending attention, read the WHOLE pair tensor. That is
the shape of the dual-chip design: both chips hold all of z and each owns half of the output rows.

    python3 perf/b2z2_dualchip/slab_axes_cpu.py
"""

import pathlib
import sys

# Score the checkout this script LIVES IN, not whatever `tt_bio` the env has installed. The
# editable install in /home/ttuser/tt-bio-dev/env resolves to a different tree, and `python
# perf/.../slab_*.py` puts the SCRIPT's directory on sys.path rather than the cwd, so without this
# the run reports a verdict about code nobody edited.
_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)


import torch

from tt_bio import reference as ref
from tt_bio.reference import permute_final_dims

def randomize_(module):
    """Give every parameter a nonzero value.

    Boltz-2's reference init is `final_init_` on both output projections of the triangle
    multiplication and `gating_init_`/zero elsewhere, so a freshly constructed layer returns
    EXACTLY ZERO from trimul and transition. Every comparison in this file would then hold for
    reasons that have nothing to do with slabs -- the first run of the negative control below
    reported max abs diff 0.0 against the WRONG rows, which is what caught it. A parity fixture
    has to carry weights that make the op produce structure.
    """
    for name, p in module.named_parameters():
        if p.dim() == 1:
            p.data = (1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p)
        else:
            p.data = torch.randn_like(p) * (p.shape[-1] ** -0.5)


print(f"tt_bio from {ref.__file__}", flush=True)

torch.manual_seed(0)
S, C = 64, 32
R0, R1 = 32, 64

rl = ref.PairformerLayer(48, C, 4, 0.0, 8, 2, v2=True).eval()
randomize_(rl)
z = torch.randn(1, S, S, C)
# Ragged, so a mask sliced on the wrong axis cannot pass.
m1 = torch.zeros(1, S)
m1[:, :S - 8] = 1.0
mask = m1[:, :, None] * m1[:, None, :]


def trimul_slab(m, z, mask, ending, r0, r1):
    """The reference forward with `a` cut to the output rows and `b` left whole."""
    x = m.norm_in(z)
    x_in = x
    x = m.p_in(x) * m.g_in(x).sigmoid()
    x = x * mask.unsqueeze(-1)
    a, b = torch.chunk(x.float(), 2, dim=-1)
    if ending:
        o = torch.einsum("bkid,bkjd->bijd", a[:, :, r0:r1], b)
    else:
        o = torch.einsum("bikd,bjkd->bijd", a[:, r0:r1], b)
    # The tail is per output row for both variants.
    return m.p_out(m.norm_out(o)) * m.g_out(x_in[:, r0:r1]).sigmoid()


def triatt_slab(m, z, mask, r0, r1):
    """The reference forward with the slab on whichever axis that variant puts the row on."""
    x, msk = z, mask
    if not m.starting:
        x = x.transpose(-2, -3)
        msk = msk.transpose(-1, -2)
    x = m.layer_norm(x)
    msk = msk[..., :, None, None, :]
    mask_bias = m.inf * (msk - 1)
    tri = permute_final_dims(m.linear(x), (2, 0, 1)).unsqueeze(-4)
    if m.starting:
        return m.mha(x[:, r0:r1], x[:, r0:r1], tri, mask_bias[:, r0:r1], msk[:, r0:r1])
    # `tri` is [*, 1, H, query, key] and the key mask broadcasts over the query, so only the
    # bias takes the slab here, not the mask.
    return m.mha(x[..., r0:r1, :], x, tri[..., r0:r1, :], mask_bias, msk).transpose(-2, -3)


cases = [
    ("trimul_start", lambda: rl.tri_mul_out(z, mask),
     lambda: trimul_slab(rl.tri_mul_out, z, mask, False, R0, R1)),
    ("trimul_end", lambda: rl.tri_mul_in(z, mask),
     lambda: trimul_slab(rl.tri_mul_in, z, mask, True, R0, R1)),
    ("triatt_start", lambda: rl.tri_att_start(z, mask),
     lambda: triatt_slab(rl.tri_att_start, z, mask, R0, R1)),
    ("triatt_end", lambda: rl.tri_att_end(z, mask),
     lambda: triatt_slab(rl.tri_att_end, z, mask, R0, R1)),
    ("transition_z", lambda: rl.transition_z(z), lambda: rl.transition_z(z[:, R0:R1])),
]

bad = []
with torch.no_grad():
    for name, whole_fn, slab_fn in cases:
        whole, slab = whole_fn(), slab_fn()
        d = float((whole[:, R0:R1] - slab).abs().max())
        ok = d == 0.0
        print(f"{name:13s} rows[{R0}:{R1}] max abs diff {d:.3e}  "
              f"{'OK' if ok else 'AXIS WRONG'}", flush=True)
        if not ok:
            bad.append(name)

# The control: the same comparison against the OTHER half's rows has to fail, or the slice is
# lining up with itself and the five lines above mean nothing.
with torch.no_grad():
    whole = rl.transition_z(z)
    wrong = float((whole[:, 0:R1 - R0] - rl.transition_z(z[:, R0:R1])).abs().max())
print(f"negative control: slab compared against the WRONG rows -> max abs diff {wrong:.3e} "
      f"{'(rejected, good)' if wrong > 0 else '(ACCEPTED, the comparison is blind)'}")
if wrong == 0:
    bad.append("negative control")

sys.exit(1 if bad else 0)
