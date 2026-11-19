"""Device probe: trunk block 43's `fp32_single_transition` stage by stage vs float64 with the same bf16 weights.
TT_VISIBLE_DEVICES=N python tr_stage_probe.py X0.pt OUT.json  (X0.pt from x0_dump.py)
Per stage (ln, a, b, h, out): rel err of the token part (x - token mean) over all channels, and for `out` over
the 8 small channels of x + out as well (what the confidence head reads)."""
import json, sys
import torch
import torch.nn.functional as F
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
import tt_bio.tenstorrent as T
d = torch.load(sys.argv[1], weights_only=False)
bf = lambda t: t.to(torch.bfloat16).double()
x = d["x"].double(); n = x.shape[0]
lw, lb, w1, w2, w3 = (bf(d[k]) for k in ("w", "b", "w1", "w2", "w3"))
mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
ref = {"ln": (x - mu) / (var + 1e-5).sqrt() * lw + lb}
ref["a"] = ref["ln"] @ w1.T; ref["b"] = ref["ln"] @ w2.T
ref["h"] = F.silu(ref["a"]) * ref["b"]; ref["out"] = ref["h"] @ w3.T
dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=False)
up = lambda t, dt: ttnn.from_torch(t.float().reshape(1, -1, t.shape[-1]), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
wt = lambda t: ttnn.from_torch(t.float().T.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
back = lambda t: ttnn.to_torch(t).double().reshape(-1, t.shape[-1])[:n]
tok = lambda v: v - v.mean(0)
def g(k, o): return float((tok(o) - tok(ref[k])).norm() / tok(ref[k]).norm())
X = up(x, ttnn.float32)
W, B = up(lw.reshape(1, -1), ttnn.bfloat16), up(lb.reshape(1, -1), ttnn.bfloat16)
W1, W2, W3 = wt(w1), wt(w2), wt(w3)
L1 = ttnn.L1_MEMORY_CONFIG
dv = {"ln": T.fp32_layer_norm(X, W, B, 1e-5, ttnn.float32, L1)}
dv["a"] = T._centred_linear(T._centre(dv["ln"], L1), W1, ckc, ttnn.float32, L1, L1)
dv["b"] = T._centred_linear(T._centre(dv["ln"], L1), W2, ckc, ttnn.float32, L1, L1)
dv["h"] = ttnn.multiply(ttnn.silu(dv["a"]), dv["b"])
dv["out"] = T._centred_linear(T._centre(dv["h"], L1), W3, ckc, ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, L1)
res = {k: g(k, back(v)) for k, v in dv.items()}
# each stage fed the float64 reference input, so its own error is separated from what it inherits
res["a_from_ref_ln"] = g("a", back(T._centred_linear(T._centre(up(ref["ln"], ttnn.float32), L1), W1, ckc, ttnn.float32, L1, L1)))
res["out_from_ref_h"] = g("out", back(T._centred_linear(T._centre(up(ref["h"], ttnn.float32), L1), W3, ckc, ttnn.float32, L1, L1)))
res["silu_mul_from_ref_ab"] = g("h", back(ttnn.multiply(ttnn.silu(up(ref["a"], ttnn.float32)), up(ref["b"], ttnn.float32))))
small = (x + ref["out"]).abs().amax(0) < 512
s_ref = (x + ref["out"])[:, small]
res["block_small_tok"] = float((tok((x + back(dv["out"]))[:, small]) - tok(s_ref)).norm() / tok(s_ref).norm())
full = T.fp32_single_transition(X, W, B, W1, W2, W3, ckc, ttnn.float32)
res["fp32_single_transition_small_tok"] = float((tok((x + back(full))[:, small]) - tok(s_ref)).norm() / tok(s_ref).norm())
for k, v in res.items():
    print(k, v, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
