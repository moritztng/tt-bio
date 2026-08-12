#!/usr/bin/env python3
"""Cut openfold3's attention traffic WITHOUT touching its softmax precision.

Where this comes from. The fused flash-SDPA route is 1.354x on the fold but costs 0.108 plDDT, and
that loss is not recoverable by configuration: `fp32_dest_acc` does not change CB data formats, and
ttnn's SDPA op hard-rejects fp32 tensors outright (`sdpa_device_operation.cpp:44`, MEASURED in
`s3_fp32_sdpa.json`). So the fused route needs a real kernel change and an accuracy decision.

This screen attacks the same 33.034 s from the other side: `_fp32_softmax_attention` keeps the whole
SxSxH score tensor in fp32 and walks it FIVE times -- typecast up, multiply by scale, add bias,
softmax, typecast down. The softmax reduction itself is only one of those five. The other four are
bookkeeping, and `ttnn.scale_mask_softmax` folds scale and mask-add into the softmax's own pass.

DERIVED byte model at openfold3's dominant call (B=512, H=4, S=512, D=32; `sc` bf16 1.074 GB,
fp32 2.147 GB; bias is [1,H,S,S] and negligible at 4.19 MB fp32):

    today        typecast 3.22 + multiply 4.29 + add 4.29 + softmax 4.29 + typecast 3.22 = 19.31 GB
                 on the score tensor, plus matmuls -> ~21.9 GB total
    recomposed   typecast 3.22 + scale_mask_softmax 4.29 + typecast 3.22 = 10.73 GB -> ~13.4 GB
    + fp32 mm    matmul writes fp32 directly, first typecast gone -> ~11.2 GB

so 1.63x-1.95x on the component at an UNCHANGED softmax: same fp32 storage, same reduction, same
order. That is the whole point -- this route needs no accuracy decision from anyone.

The bar here is exactness against today's path, not against a gold: this is a rewrite of an op
sequence, so `torch.equal` is the target and anything else has to be justified.
"""
from __future__ import annotations

import json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn                                                            # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor  # noqa: E402
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
B, H, S, D = 512, 4, 512, 32
SCALE_INV = float(D) ** -0.5          # what the call site passes as `scale_inv`
torch.manual_seed(0)

CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(r)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return round(st.median(ts) * 1e3, 4)


def today(q, k, v, bias):
    """Verbatim `_fp32_softmax_attention` with openfold3's arguments (bias_scale_inv = 1.0)."""
    return T._fp32_softmax_attention(q, k, v, bias, scale_inv=SCALE_INV, compute_kernel_config=CKC,
                                     out_dtype=ttnn.bfloat16, bias_scale_inv=1.0)


def recomposed(q, k, v, bias, fp32_matmul: bool):
    kt = ttnn.permute(k, (0, 1, 3, 2))
    if fp32_matmul:
        sc = T.batched_matmul(q, kt, compute_kernel_config=CKC, dtype=ttnn.float32)
    else:
        sc = T.batched_matmul(q, kt, compute_kernel_config=CKC)
        sc = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
    ttnn.deallocate(kt)
    bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
    attn = ttnn.scale_mask_softmax(sc, SCALE_INV, bias_f)
    ttnn.deallocate(bias_f)
    ttnn.deallocate(sc)
    attn_bf = ttnn.typecast(attn, ttnn.bfloat16, memory_config=attn.memory_config())
    ttnn.deallocate(attn)
    o = T.batched_matmul(attn_bf, v, compute_kernel_config=CKC, dtype=ttnn.bfloat16)
    ttnn.deallocate(attn_bf)
    return o


def stats(got, ref):
    a, b = got.float().flatten(), ref.float().flatten()
    d = a - b
    return {"equal": bool(torch.equal(got, ref)),
            "max_abs": round(d.abs().max().item(), 9),
            "rmsd_over_std": round((d.pow(2).mean().sqrt() / b.std()).item(), 9)}


def main():
    def dt(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    q = dt(torch.randn(B, H, S, D, dtype=torch.bfloat16))
    k = dt(torch.randn(B, H, S, D, dtype=torch.bfloat16))
    v = dt(torch.randn(B, H, S, D, dtype=torch.bfloat16))
    bias = dt(torch.randn(1, H, S, S, dtype=torch.bfloat16) * 0.5)

    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "shape": {"B": B, "H": H, "S": S, "D": D}, "rows": []}

    o_ref = today(q, k, v, bias)
    ref = ttnn.to_torch(o_ref)
    ttnn.deallocate(o_ref)
    row = {"name": "today (_fp32_softmax_attention)", "ms": bench(lambda: today(q, k, v, bias))}
    res["rows"].append(row)
    print(json.dumps(row), flush=True)

    for name, f32mm in (("scale_mask_softmax", False), ("scale_mask_softmax + fp32 matmul", True)):
        try:
            o = recomposed(q, k, v, bias, f32mm)
            row = {"name": name}
            row.update(stats(ttnn.to_torch(o), ref))
            ttnn.deallocate(o)
            row["ms"] = bench(lambda: recomposed(q, k, v, bias, f32mm))
            row["speedup_vs_today"] = round(res["rows"][0]["ms"] / row["ms"], 4)
        except Exception as e:                                                # noqa: BLE001
            row = {"name": name, "error": f"{type(e).__name__}: {str(e)[:400]}"}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)

    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[1], flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
