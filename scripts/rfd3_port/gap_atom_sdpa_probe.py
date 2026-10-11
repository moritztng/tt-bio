#!/usr/bin/env python3
"""Atom attention at the page shape: shipped chain vs RFD3_ATOM_SDPA vs a float64 reference.

    gap_atom_sdpa_probe.py OUT.json [L] [CHUNKS ...]

q/k/v are [1, 4, L padded to 32, 32] bf16 (rms-normed scale, like the model's), the neighbour index
is 128 sorted distinct keys per row in [0, L), the compact pair bias is N(0, 1). The reference is
softmax(q k^T / sqrt(32) + bias) v in float64 on the same bf16 inputs, with -1e4 off the neighbours.
Each CHUNKS entry is "q,k" for the SDPA program config. Reports ms/call (median of 20, synced) and
the max/mean abs error and PCC of each path against the reference.
"""
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio import rfd3_bias, softmax_generic                              # noqa: E402
from tt_bio.tenstorrent import get_device, fused_sdpa, _sdpa_program_config  # noqa: E402
from tt_bio.rfd3 import model as M                                         # noqa: E402
from tt_bio.rfd3.tiles import align_tile, pad_axis                         # noqa: E402

out = pathlib.Path(sys.argv[1])
L = int(sys.argv[2]) if len(sys.argv) > 2 else 6051
chunks = [tuple(int(x) for x in c.split(",")) for c in sys.argv[3:]] or [(160, 320)]
H, D, K = 4, 32, 128
N = align_tile(L)
scale = D ** -0.5
g = torch.Generator().manual_seed(0)

dev = get_device()
ckc = M._default_compute_kernel_config()
dt = ttnn.bfloat16


def rms(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


q = rms(torch.randn(1, H, L, D, generator=g)).bfloat16()
k = rms(torch.randn(1, H, L, D, generator=g)).bfloat16()
v = torch.randn(1, H, L, D, generator=g).bfloat16()
idx = torch.stack([torch.randperm(L, generator=g)[:K].sort().values for _ in range(L)]).unsqueeze(0)  # [1,L,K]
pb = torch.randn(1, H, L, K, generator=g).bfloat16()

# float64 reference
bias = torch.full((1, H, L, L), -1e4, dtype=torch.float64)
bias.scatter_(3, idx.unsqueeze(1).expand(1, H, L, K), pb.double())
ref = torch.softmax(q.double() @ k.double().transpose(-1, -2) * scale + bias, -1) @ v.double()
del bias


def up(x):
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


qd, kd, vd, pbd = up(q), up(k), up(v), up(pb)
kd, vd = pad_axis(kd, N, 2, 0.0), pad_axis(vd, N, 2, 0.0)
idx_rm = M._sparse_attn_index_rm(idx, dev)


def shipped():
    s = ttnn.matmul(qd, ttnn.permute(kd, (0, 1, 3, 2)), compute_kernel_config=ckc)
    s = rfd3_bias.fused_scores_bias_fp32(s, pbd, idx_rm, scale)
    o = softmax_generic.softmax_pv_fused(s, vd, dt, ckc)
    if o is None:
        o = M.attn_value_matmul(softmax_generic.softmax_bf16(s, dt), vd, ckc, dt)
    return o


def mask():
    b = rfd3_bias.sparse_bias_fp32(pbd, idx_rm)
    s = ttnn.multiply(b, D ** 0.5)
    return pad_axis(ttnn.typecast(s, dt), N, 2, 0.0)


def sdpa(m, qc, kc):
    o = fused_sdpa(pad_axis(qd, N, 2, 0.0), kd, vd, attn_mask=m, scale=scale,
                   program_config=_sdpa_program_config(qc, kc), compute_kernel_config=ckc)
    return ttnn.slice(o, [0, 0, 0, 0], [1, H, L, D])


def timed(fn, n=20):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def grade(o):
    o = ttnn.to_torch(o).double()[:, :, :L, :D]
    assert torch.isfinite(o).all(), "non-finite output"
    e = (o - ref).abs()
    pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
    return {"maxabs": e.max().item(), "meanabs": e.mean().item(), "pcc": pcc}


res = {"L": L, "N": N, "arch": str(dev.arch()), "shipped": {"ms": timed(shipped), **grade(shipped())}}
m = mask()
res["mask_build_ms"] = timed(mask)
for qc, kc in chunks:
    try:
        res[f"sdpa_{qc}_{kc}"] = {"ms": timed(lambda: sdpa(m, qc, kc)), **grade(sdpa(m, qc, kc))}
    except Exception as e:                                                   # noqa: BLE001
        res[f"sdpa_{qc}_{kc}"] = {"error": str(e)[:300]}
    print(json.dumps(res), flush=True)
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
