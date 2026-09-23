"""Transition(add_to_input=True) under a tape, input a leaf that requires grad (as z does in the
trunk). Is the forward x + t(x), and does the weight gradient reach fc1..fc3 and the norm?"""
import json, torch, ttnn
from tt_bio import tenstorrent as T
import tt_bio.autograd as ag
torch.manual_seed(0)
C = 128
w = {"norm.weight": 1 + 0.1 * torch.randn(C), "norm.bias": 0.1 * torch.randn(C),
     "fc1.weight": torch.randn(4 * C, C) / C**.5, "fc2.weight": torch.randn(4 * C, C) / C**.5,
     "fc3.weight": torch.randn(C, 4 * C) / (4 * C)**.5}
dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
tr = T.Transition(w, ckc)
names = ("norm_weight", "norm_bias", "fc1_weight", "fc2_weight", "fc3_weight")
P = {n: ag.parameter(getattr(tr, n)) for n in names}
xt = torch.randn(1, 64, 64, C)
host = lambda t: ttnn.to_torch(getattr(t, "value", t)).float()
def arm(add):
    for p in P.values():
        p.grad = None
    x = ag.Tensor(ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
                  requires_grad=True)
    with ag.tape():
        y = tr(x, add_to_input=True) if add else tr(x)
    yv = host(y)
    ag.backward([y])
    g = {n: (None if P[n].grad is None else float(host(P[n].grad).norm())) for n in names}
    return yv, g, y is x
x0 = xt.bfloat16().float()
u, g_u, _ = arm(False)
y, g_y, same = arm(True)
print("PROBE", json.dumps({
    "add_to_input returns its input wrapper": same,
    "max|u|": float(u.abs().max()),
    "max|y - x|": float((y - x0).abs().max()),
    "max|y - (x + u)|": float((y - (x0 + u)).abs().max()),
    "weight grad norms, t(x)": g_u, "weight grad norms, x + t(x) via add_to_input": g_y}))
