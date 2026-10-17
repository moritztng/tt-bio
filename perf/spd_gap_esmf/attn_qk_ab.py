"""ESMFold2 token attention (_attn_fp32) with and without TT_BIO_ESMFOLD2_ATTN_QK_BF16, on one chip.

Shape of the token DiT at 512 aa: q/k/v [1,16,512,64] bf16 (head 48 padded to 64), bias [1,16,512,512] bf16,
logits up to ~100. Reports per-arm ms/call, the max |diff| between the arms and each arm's rel-RMS against a
float64 reference.

    python perf/spd_gap_esmf/attn_qk_ab.py --n 512 --out r1.jsonl
"""
import argparse, json, time

import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import esmfold2 as E

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = T.get_device()
g = torch.Generator().manual_seed(0)
H, D, Dr, N = 16, 64, 48, a.n
q = torch.zeros(1, H, N, D); k = torch.zeros(1, H, N, D); v = torch.zeros(1, H, N, D)
q[..., :Dr] = torch.randn(1, H, N, Dr, generator=g) * 3.0   # q.k/sqrt(48) reaches ~100 like the real logits
k[..., :Dr] = torch.randn(1, H, N, Dr, generator=g) * 3.0
v[..., :Dr] = torch.randn(1, H, N, Dr, generator=g)
bias = torch.randn(1, H, N, N, generator=g) * 4.0
q, k, v, bias = (x.bfloat16() for x in (q, k, v, bias))
up = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
tq, tk, tv, tb = (up(x) for x in (q, k, v, bias))
scale = Dr ** -0.5

lg = (q.double() @ k.double().transpose(-1, -2)) * scale + bias.double()
ref = torch.softmax(lg, -1) @ v.double()
print(f"logit max |.| {lg.abs().max().item():.1f}", flush=True)


def ck(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=True)


rows = []
for fname, fid in (("HiFi4", ttnn.MathFidelity.HiFi4), ("HiFi2", ttnn.MathFidelity.HiFi2)):
    outs = {}
    for arm, on in (("O", False), ("N", True)):
        E._ATTN_QK_BF16 = on
        run = lambda: E._attn_fp32(tq, tk, tv, tb, scale, ck(fid))
        try:
            y = run(); ttnn.synchronize_device(dev)
            outs[arm] = ttnn.to_torch(y).double()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                ttnn.deallocate(run())
            ttnn.synchronize_device(dev)
            ms = (time.perf_counter() - t0) / a.iters * 1e3
            rr = ((outs[arm] - ref).norm() / ref.norm()).item()
            rows.append({"n": N, "ck": fname, "arm": arm, "ms": round(ms, 4), "rel_rms_f64": rr})
        except Exception as e:  # noqa: BLE001 -- report which arm the op set refuses
            rows.append({"n": N, "ck": fname, "arm": arm, "error": repr(e)[:300]})
        print(rows[-1], flush=True)
    if len(outs) == 2:
        rows.append({"n": N, "ck": fname, "max_diff_O_N": (outs["O"] - outs["N"]).abs().max().item(),
                     "bit_identical": bool(torch.equal(outs["O"], outs["N"]))})
        print(rows[-1], flush=True)

with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
