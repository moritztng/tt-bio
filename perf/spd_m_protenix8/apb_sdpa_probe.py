"""apb_sdpa and apb_qkvf vs the fp32 accurate-softmax chain at Protenix-v2's pairformer attention shape, against float64.

q, k, v: [1, 16, n, 24] padded to 32 head columns (zeros), bf16 as the trunk feeds them. bias: a per-head pair bias
whose within-row spread runs from ~1 to ~135 (the range `_accurate_softmax` documents), plus a -1e9 key mask on the
last `--masked` tokens, added before the call as AttentionPairBias does. Prints one JSON line per n:
rel_rms of each output vs the float64 softmax((q k^T + bias) * d^-0.5) @ v on the bf16-rounded inputs, the max abs
error, and device ms per call (median of --reps, synchronized). `qkvf` is apb_qkvf's call: one fp32 `_sdpa32_fused`
on the fused [1, 1, n, 3 * 16 * 32] qkv (q heads, then k, then v), bf16 out, heads concatenated on their padded lanes.

    TT_VISIBLE_DEVICES=<card> python perf/spd_m_protenix8/apb_sdpa_probe.py 512 256
"""
import argparse
import json
import statistics
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("sizes", nargs="+", type=int)
ap.add_argument("--masked", type=int, default=20)
ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

dev = T.get_device()
H, D, DP = 16, 24, 32
scale = D ** -0.5
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)


def tt(x):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def chain(q, k, v, z):
    """AttentionPairBias's unfused branch, verbatim."""
    kt = ttnn.transpose(k, -2, -1)
    logits = T.batched_matmul(q, kt, compute_kernel_config=ckc)
    ttnn.deallocate(kt)
    logits = ttnn.add_(logits, z)
    logits = ttnn.multiply_(logits, scale)
    probs = T._accurate_softmax(logits, ckc)
    ttnn.deallocate(logits)
    o = T.batched_matmul(probs, v, compute_kernel_config=ckc)
    ttnn.deallocate(probs)
    return o


def qkvf(qkv, _k, _v, z):
    """AttentionPairBias._qkvf's apb_qkvf branch: fp32 operands, bf16 out."""
    q32, z32 = ttnn.typecast(qkv, ttnn.float32), ttnn.typecast(z, ttnn.float32)
    o = T._sdpa32_fused(q32, z32, scale, H, out_dtype=ttnn.bfloat16)
    ttnn.deallocate(q32)
    ttnn.deallocate(z32)
    return o


def timed(f, *args):
    ts = []
    for _ in range(a.reps):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        o = f(*args)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t) * 1e3)
        ttnn.deallocate(o)
    return statistics.median(ts)


torch.manual_seed(0)
for n in a.sizes:
    q, k, v = (torch.zeros(1, H, n, DP) for _ in range(3))
    for t in (q, k, v):
        t[..., :D] = torch.randn(1, H, n, D)
    q[..., :D] *= 2.0   # logits with a few units of spread from q k^T alone
    # pair bias: per head a different spread, ~1 (head 0) to ~135 (head 15) within a row, pre-scaled by sqrt(d)
    spread = torch.logspace(0, 2.13, H).reshape(1, H, 1, 1)
    z = torch.randn(1, H, n, n) * spread / 4.0 / scale
    z[..., n - a.masked:] = -1e9
    qb, kb, vb, zb = (x.to(torch.bfloat16) for x in (q, k, v, z))
    ref = torch.softmax(((qb.double() @ kb.double().transpose(-1, -2)) + zb.double()) * scale, -1) @ vb.double()
    out = {"n": n}
    with T.levers({"apb_sdpa", "apb_qkvf"}):
        for name, f in (("chain", chain), ("sdpa", lambda *x: T._apb_sdpa32(*x, scale)), ("qkvf", qkvf)):
            Q, K, V, Z = tt(qb), tt(kb), tt(vb), tt(zb)
            if name == "qkvf":
                Q = tt(torch.cat([t.permute(0, 2, 1, 3).reshape(1, 1, n, H * DP) for t in (qb, kb, vb)], -1))
            o = f(Q, K, V, Z)
            if o is None:
                out[name] = "declined"
                continue
            got = ttnn.to_torch(o).double()
            if name == "qkvf":   # [1, 1, n, H * DP] -> [1, H, n, DP]
                got = got.reshape(1, n, H, DP).permute(0, 2, 1, 3)
            got = got[..., :n, :DP]
            err = got - ref
            out[name] = {"rel_rms": float(err[..., :D].pow(2).mean().sqrt() / ref[..., :D].pow(2).mean().sqrt()),
                         "max_abs": float(err[..., :D].abs().max()), "finite": bool(torch.isfinite(got).all()),
                         "ms": timed(f, Q, K, V, Z) if name != "chain" else None}
            if name == "chain":
                # the chain consumes z in place (add_), so time it on fresh copies
                ts = []
                for _ in range(a.reps):
                    Z2 = tt(zb)
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    o2 = chain(Q, K, V, Z2)
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(o2)
                    ttnn.deallocate(Z2)
                out[name]["ms"] = statistics.median(ts)
    print(json.dumps(out), flush=True)
