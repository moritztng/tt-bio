"""Does eliding the triangle-attention gate change the numbers? The block-level sum digest said no,
which would make the timing ablation meaningless, so ask the question where the answer is visible."""
import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import torch, ttnn
from tt_bio import tenstorrent as tt
from tt_bio import reference as ref

S = 512
dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                      fp32_dest_acc_en=True, packer_l1_acc=True)
_real = ttnn.multiply_
ELIDE = [False]
N = [0]
def shim(a, b, *ar, **kw):
    sh = list(getattr(a, "shape", []))
    if (kw.get("input_tensor_b_activations") is not None and len(sh) == 4
            and sh == list(getattr(b, "shape", [])) and sh[0] != 1 and sh[-1] == 32):
        N[0] += 1
        if ELIDE[0]:
            ttnn.deallocate(b); return a
    return _real(a, b, *ar, **kw)
ttnn.multiply_ = shim

torch.manual_seed(0)
rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
w = {k: v.float() for k, v in rl.state_dict().items()}
layer = tt.PairformerLayer(32, 4, 24, 16, True, w, KC)
s = torch.randn(1, S, 384); z = torch.randn(1, S, S, 128)
m1 = torch.ones(1, S); pm = m1[:, :, None] * m1[:, None, :]
attn = (1 - m1).unsqueeze(1).unsqueeze(1) * -1e9
f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
mt, at = f(pm), f(attn)

outs = {}
for e in (False, True):
    ELIDE[0] = e
    os_, oz = layer(f(s), f(z), mt, at, at)
    outs[e] = (ttnn.to_torch(os_).float(), ttnn.to_torch(oz).float())
    ttnn.deallocate(os_); ttnn.deallocate(oz)

zg, ze = outs[False][1], outs[True][1]
sg, se = outs[False][0], outs[True][0]
d = (zg - ze).abs()
print("gate multiplies intercepted:", N[0])
print("z  equal:", torch.equal(zg, ze), " max|dz|:", float(d.max()),
      " mean|dz|:", float(d.mean()), " frac elements changed:", float((d > 0).float().mean()))
print("z  sum gated/elided:", float(zg.sum()), float(ze.sum()))
print("z  absum gated/elided:", float(zg.abs().sum()), float(ze.abs().sum()))
print("s  equal:", torch.equal(sg, se))
