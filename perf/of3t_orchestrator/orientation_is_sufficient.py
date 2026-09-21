"""What is the 0.4.3 -> 0.5.0 ending-node bias orientation worth?

First attempt was VACUOUS: TriangleAttention zero-initialises its output projection, so an
untrained module emits exactly 0.0 on both arms and the ratio was 0/0 = nan. An arm that cannot
differ is not evidence (the campaign has filed that class twice). Fixed by giving every
parameter non-degenerate values, so the module actually computes something.
"""
import sys, torch
sys.path.insert(0, "/home/moritz/.coworker/scratch/of3t-data/up0.5.0")
from openfold3.core.model.layers.triangular_attention import TriangleAttention

torch.manual_seed(0)
m = TriangleAttention(128, 32, 4).double().eval()
with torch.no_grad():                      # defeat the zero-init that made arm 1 vacuous
    for p in m.parameters():
        p.normal_(0.0, 0.05)
nz = sum(int((p != 0).any()) for p in m.parameters())
print(f"parameters made non-degenerate: {nz}/{len(list(m.parameters()))} tensors non-zero")

z = torch.randn(1, 64, 64, 128, dtype=torch.double)
mask = torch.ones(1, 64, 64, dtype=torch.double)
with torch.no_grad():
    a = m(z, mask=mask, transpose_bias=False)   # 0.4.3's only behaviour
    b = m(z, mask=mask, transpose_bias=True)    # what 0.5.0's caller passes
print(f"  arm norms: 0.4.3 {float(a.norm()):.6e}   0.5.0 {float(b.norm()):.6e}")
assert float(a.norm()) > 0 and float(b.norm()) > 0, "still degenerate -- refusing to report"
rel = float((b - a).norm() / a.norm())
cos = float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))
print(f"  relative L2 between the two orientations : {rel:.6e}")
print(f"  cosine                                   : {cos:+.6f}")
print()
# control: the same call twice must be bit-identical, or the comparison means nothing
with torch.no_grad():
    c = m(z, mask=mask, transpose_bias=False)
print(f"  CONTROL, same arm twice: max abs diff {float((c-a).abs().max()):.3e} "
      f"(must be 0.0, else the module is not deterministic)")
