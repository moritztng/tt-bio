#!/usr/bin/env python3
"""DiT attention layout microbench: today's 48-wide head split vs heads padded to 64 in the weights.

    gap_dit_heads.py OUT.json [L=685] [REPS=50]

A (shipped RFD3_DIT_SDPA arm): q,k,v [1,L,768] -> reshape+permute x3 -> pad head_dim 48->64 x3 -> pad keys
  -> fused SDPA -> slice -> permute+reshape -> o linear.
B: q,k,v come out of their linears as [1,L,1024], each head's 48 columns followed by 16 zero columns (zero
  weight columns), rms_norm with its weight scaled by sqrt(768/1024) and eps by 768/1024 so the zero columns
  do not change the row's rms; concat -> nlp_create_qkv_heads -> pad keys -> fused SDPA -> nlp_concat_heads ->
  o linear whose padded rows are zero. Same math; reported as maxabs vs a float64 reference of the block's
  attention output, both arms, plus ms/call (synchronized, REPS calls after 3 warm).
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                       # noqa: E402

ensure_p300_mesh_descriptor()
import torch                                                              # noqa: E402
import ttnn                                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402

OUT = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 685
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
H, D, DP, C = 16, 48, 64, 768
torch.manual_seed(0)
dev = T.get_device()
ckc = M._default_compute_kernel_config()
bf = ttnn.bfloat16
nk = M.align_tile(L)

x = torch.randn(1, L, C)
Wq, Wk, Wv, Wo = (torch.randn(C, C) / math.sqrt(C) for _ in range(4))
lnq, lnk = 1 + 0.1 * torch.randn(C), 1 + 0.1 * torch.randn(C)
mask = torch.zeros(1, H, L, L)
mask[..., torch.rand(L, L) > 0.3] = -1e4          # a dense-neighbour-like mask
mask[..., torch.arange(L), torch.arange(L)] = 0
bias = 0.5 * torch.randn(1, H, L, L)


def rms(t, w, eps=1e-6):
    return t / torch.sqrt((t * t).mean(-1, keepdim=True) + eps) * w


def reference():
    xd = x.double()
    q, k, v = rms(xd @ Wq.double(), lnq.double()), rms(xd @ Wk.double(), lnk.double()), xd @ Wv.double()
    sp = lambda t: t.view(1, L, H, D).permute(0, 2, 1, 3)
    s = sp(q) @ sp(k).transpose(-1, -2) * D ** -0.5 + bias.double() + mask.double()
    o = (s.softmax(-1) @ sp(v)).permute(0, 2, 1, 3).reshape(1, L, C)
    return o @ Wo.double()


def pad_heads_cols(w):          # [C_in, H*D] -> [C_in, H*DP], zero columns after each head
    return torch.nn.functional.pad(w.view(-1, H, D), (0, DP - D)).reshape(w.shape[0], H * DP)


def tt(t, dtype=bf):
    return ttnn.from_torch(t.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


xt = tt(x)
lin = dict(compute_kernel_config=ckc, dtype=bf, core_grid=M.BATCH_INVARIANT_GRID)
pb = (bias + mask) * D ** 0.5            # the arm pre-scales the bias, the mask's -1e4 stays masked
mask_dev = M.pad_axis(M.pad_axis(tt(pb), nk, 3, -1e4), nk, 2, -1e4)
cfg = M._dense_sdpa_config(nk)

A_w = dict(q=tt(Wq), k=tt(Wk), v=tt(Wv), o=tt(Wo), lnq=tt(lnq.view(1, -1)), lnk=tt(lnk.view(1, -1)))


def arm_a():
    q = ttnn.rms_norm(ttnn.linear(xt, A_w["q"], **lin), weight=A_w["lnq"], epsilon=1e-6, compute_kernel_config=ckc)
    k = ttnn.rms_norm(ttnn.linear(xt, A_w["k"], **lin), weight=A_w["lnk"], epsilon=1e-6, compute_kernel_config=ckc)
    v = ttnn.linear(xt, A_w["v"], **lin)
    q, k, v = (ttnn.permute(ttnn.reshape(t, (1, L, H, D)), (0, 2, 1, 3)) for t in (q, k, v))
    k, v = M.pad_axis(k, nk, 2, 0.0), M.pad_axis(v, nk, 2, 0.0)
    q, k, v = (M.pad_axis(t, DP, 3, 0.0) for t in (q, k, v))
    o = M.fused_sdpa(M.pad_axis(q, nk, 2, 0.0), k, v, attn_mask=mask_dev, scale=D ** -0.5,
                     program_config=cfg, compute_kernel_config=ckc)
    o = ttnn.slice(o, [0, 0, 0, 0], [1, H, L, D])
    o = M._merge_heads(o, (1, L, C))
    return ttnn.linear(o, A_w["o"], **lin)


s = math.sqrt(C / (H * DP))
B_w = dict(q=tt(pad_heads_cols(Wq)), k=tt(pad_heads_cols(Wk)), v=tt(pad_heads_cols(Wv)),
           o=tt(pad_heads_cols(Wo.T).T),
           lnq=tt(pad_heads_cols((lnq * s).view(1, -1))), lnk=tt(pad_heads_cols((lnk * s).view(1, -1))))
EPS_B = 1e-6 * C / (H * DP)


def arm_b():
    q = ttnn.rms_norm(ttnn.linear(xt, B_w["q"], **lin), weight=B_w["lnq"], epsilon=EPS_B, compute_kernel_config=ckc)
    k = ttnn.rms_norm(ttnn.linear(xt, B_w["k"], **lin), weight=B_w["lnk"], epsilon=EPS_B, compute_kernel_config=ckc)
    v = ttnn.linear(xt, B_w["v"], **lin)
    qkv = ttnn.reshape(ttnn.concat([q, k, v], dim=-1), (1, 1, L, 3 * H * DP))
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k, v = M.pad_axis(k, nk, 2, 0.0), M.pad_axis(v, nk, 2, 0.0)
    o = M.fused_sdpa(M.pad_axis(q, nk, 2, 0.0), k, v, attn_mask=mask_dev, scale=D ** -0.5,
                     program_config=cfg, compute_kernel_config=ckc)
    o = ttnn.reshape(ttnn.experimental.nlp_concat_heads(o), (1, L, H * DP))
    return ttnn.linear(o, B_w["o"], **lin)


ref = reference()
res = {"L": L, "reps": REPS, "card": os.environ.get("TT_VISIBLE_DEVICES")}
for name, fn in (("A_shipped", arm_a), ("B_pad64_heads", arm_b)):
    out = ttnn.to_torch(fn()).double()[:, :L, :C]
    assert torch.isfinite(out).all(), name
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    ttnn.synchronize_device(dev)
    res[name] = {"ms_per_call": 1e3 * (time.perf_counter() - t0) / REPS,
                 "maxabs_vs_f64": float((out - ref).abs().max()),
                 "rel_l2_vs_f64": float((out - ref).norm() / ref.norm())}
    print(name, res[name], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
