#!/usr/bin/env python3
"""Per-op decomposition of `_fp32_softmax_attention`, and which passes over the score tensor can go.

The byte model said ~21.9 GB/call across 8 ops and the fold measured 67.7 ms per TriangleAttention
module call. This measures the 8 ops individually at openfold3's own shape so every proposal after
it is grounded in a measured share rather than a derived one, and so `bytes / time` can be checked
per op against the card-0 roof of ~388 GB/s (`roofs_card0.json`) in both directions.

It also probes the two precision-preserving rewrites that would remove a whole pass over the score
tensor, because whether they are even expressible in ttnn 0.68.0 is a fact, not a judgement:

  * `ttnn.add(bias, sc, alpha=scale_inv)` -- fold `multiply by scale` into the bias add, one pass
    over `sc` instead of two.
  * `batched_matmul(..., dtype=float32)` -- have the q@kT matmul write fp32 directly, deleting the
    bf16->fp32 typecast entirely.

`ttnn.scale_mask_softmax` was already ruled out: it demands `mask.padded_shape()[-2] == 1` and a
matching batch dim (`softmax_device_operation.cpp:533,538`), so it expresses a per-key padding or
causal mask and cannot take triangle attention's dense SxS pair bias.
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
B, H, S, D = 512, 4, 512, 32
SCALE_INV = float(D) ** -0.5
ROOF_GBPS = 388.0                      # MEASURED on this card, perf/other512/roofs_card0.json
torch.manual_seed(0)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev)
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    return st.median(ts) * 1e3


def dt(x, d_=ttnn.bfloat16):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=d_, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    q = dt(torch.randn(B, H, S, D, dtype=torch.bfloat16))
    k = dt(torch.randn(B, H, S, D, dtype=torch.bfloat16))
    v = dt(torch.randn(B, H, S, D, dtype=torch.bfloat16))
    bias = dt(torch.randn(1, H, S, S, dtype=torch.bfloat16) * 0.5)

    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "shape": {"B": B, "H": H, "S": S, "D": D}, "roof_GBps": ROOF_GBPS, "ops": []}

    def rec(name, ms, gb):
        row = {"op": name, "ms": round(ms, 4), "GB": round(gb, 4),
               "GBps": round(gb / (ms * 1e-3), 1), "pct_of_roof": round(100 * gb / (ms * 1e-3) / 1e9 * 1e9 / ROOF_GBPS, 1)}
        res["ops"].append(row)
        print(json.dumps(row), flush=True)
        return row

    sc_bf, sc_f = B * H * S * S * 2 / 1e9, B * H * S * S * 4 / 1e9
    qkv = B * H * S * D * 2 / 1e9
    bias_bf, bias_f = H * S * S * 2 / 1e9, H * S * S * 4 / 1e9

    kt = ttnn.permute(k, (0, 1, 3, 2))
    rec("permute k", bench(lambda: ttnn.permute(k, (0, 1, 3, 2))), 2 * qkv)

    sc = T.batched_matmul(q, kt, compute_kernel_config=CKC)
    rec("matmul q@kT -> bf16", bench(lambda: T.batched_matmul(q, kt, compute_kernel_config=CKC)),
        2 * qkv + sc_bf)

    scf = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
    rec("typecast bf16->fp32", bench(lambda: ttnn.typecast(sc, ttnn.float32,
                                                          memory_config=sc.memory_config())),
        sc_bf + sc_f)
    rec("multiply by scale", bench(lambda: ttnn.multiply(scf, SCALE_INV)), 2 * sc_f)

    biasf = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
    rec("add bias", bench(lambda: ttnn.add(scf, biasf)), 2 * sc_f + bias_f)
    rec("softmax fp32", bench(lambda: ttnn.softmax(scf, dim=-1)), 2 * sc_f)

    attn_bf = ttnn.typecast(scf, ttnn.bfloat16, memory_config=scf.memory_config())
    rec("typecast fp32->bf16", bench(lambda: ttnn.typecast(scf, ttnn.bfloat16,
                                                           memory_config=scf.memory_config())),
        sc_f + sc_bf)
    rec("matmul attn@v", bench(lambda: T.batched_matmul(attn_bf, v, compute_kernel_config=CKC,
                                                        dtype=ttnn.bfloat16)),
        sc_bf + qkv + qkv)

    # ---- the two rewrites, feasibility first ------------------------------------------------
    probes = {}
    try:
        r = ttnn.add(biasf, scf, alpha=SCALE_INV)
        probes["add_alpha"] = {"ok": True, "ms": round(bench(
            lambda: ttnn.add(biasf, scf, alpha=SCALE_INV)), 4)}
        ttnn.deallocate(r)
    except Exception as e:                                                    # noqa: BLE001
        probes["add_alpha"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:220]}"}
    try:
        r = T.batched_matmul(q, kt, compute_kernel_config=CKC, dtype=ttnn.float32)
        probes["matmul_fp32_out"] = {"ok": True, "dtype": str(r.dtype), "ms": round(bench(
            lambda: T.batched_matmul(q, kt, compute_kernel_config=CKC, dtype=ttnn.float32)), 4)}
        ttnn.deallocate(r)
    except Exception as e:                                                    # noqa: BLE001
        probes["matmul_fp32_out"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:220]}"}
    res["rewrite_probes"] = probes
    print(json.dumps(probes), flush=True)

    res["sum_ms"] = round(sum(o["ms"] for o in res["ops"]), 4)
    print("sum of ops:", res["sum_ms"], "ms", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[1], flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
