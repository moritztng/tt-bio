"""Which softmax spelling survives the IPA logits on the card? Logits from ipa_fused_opdiff --dump-logits."""
import sys
import numpy as np
import torch
from tt_bio import tenstorrent as T
import ttnn

x = np.load(sys.argv[1])
dev = T.get_device()
ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                      fp32_dest_acc_en=True, packer_l1_acc=True)
e = np.exp(x - x.max(-1, keepdims=True)); ref = e / e.sum(-1, keepdims=True)
put = lambda a: ttnn.from_torch(torch.as_tensor(np.asarray(a, np.float32)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
get = lambda a: ttnn.to_torch(a).float().numpy().astype(np.float64)
X = put(x)
def rep(nm, y):
    y = get(y)
    bad = ~np.isfinite(y)
    yy = np.where(bad, 0, y)
    print(f"{nm:<28} rel {np.linalg.norm(yy - ref) / np.linalg.norm(ref):.3e} max {np.abs(yy - ref).max():.3e} nonfinite {bad.sum()}", flush=True)
    return y
rep("softmax", ttnn.softmax(X, dim=-1, compute_kernel_config=ck))
rep("softmax numeric_stable", ttnn.softmax(X, dim=-1, numeric_stable=True, compute_kernel_config=ck))
m = ttnn.max(X, dim=-1, keepdim=True)
mm = get(m); print("max: shape", tuple(m.shape), "err", np.abs(mm.reshape(-1) - x.max(-1).reshape(-1)).max(), "nonfinite", (~np.isfinite(mm)).sum())
d = ttnn.subtract(X, m); dd = get(d); print("x-max err", np.abs(dd - (x - x.max(-1, keepdims=True))).max(), "nonfinite", (~np.isfinite(dd)).sum())
ex = ttnn.exp(d); ee = get(ex); print("exp err", np.abs(ee - e).max(), "nonfinite", (~np.isfinite(ee)).sum())
s = ttnn.sum(ex, dim=-1, keepdim=True, compute_kernel_config=ck); ss = get(s); print("sum err", np.abs(ss.reshape(-1) - e.sum(-1).reshape(-1)).max(), "nonfinite", (~np.isfinite(ss)).sum())
rep("manual", ttnn.multiply(ex, ttnn.reciprocal(s)))
# masked keys at -5.9e4: clamp them before the card's exp
rep("manual, clamped", ttnn.multiply(ttnn.exp(ttnn.clamp(d, min=-80.0)), ttnn.reciprocal(ttnn.sum(ttnn.exp(ttnn.clamp(d, min=-80.0)), dim=-1, keepdim=True, compute_kernel_config=ck))))
xs = x - x.max(-1, keepdims=True)
rep("softmax of host-centred", ttnn.softmax(put(xs), dim=-1, compute_kernel_config=ck))
rep("softmax of host-centred+clamped", ttnn.softmax(put(np.maximum(xs, -80)), dim=-1, compute_kernel_config=ck))
