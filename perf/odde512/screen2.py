#!/usr/bin/env python3
"""Screen 2: can an opendde `_MM_BLOCK` entry be bit-exact at kt=12, and what is the roof?

Screen 1 (`mm_sweep.json`) found the fastest bit-exactness-preserving config the table's own rule
prescribes -- K_block == kt, one K block for the whole contraction -- is NOT `torch.equal` at kt=12,
at any M_block, at any of 512 / 576 / 640 / 995. Every candidate differs by max_abs 0.5, which is
one bf16 ULP at the largest output magnitude, i.e. the last-bit K-fold difference the
`_PAIR_PROJ_BW` comment already describes. That means the property that made all six shipped entries
byte-identical (`K_block == kt` reproduces the unconfigured op's order) does NOT hold at c_z=384.

So the unconfigured `minimal_matmul` must be splitting K at kt=12. This screen asks the only
question that can restore bit-exactness: which K_block reproduces the default's order? Sweep every
divisor of 12 and look for `torch.equal`. If one exists, tune M_block on top of IT.

It also prices the accuracy of every candidate against an fp32 torch reference, so a
not-bit-exact entry can be judged on whether it moves TOWARD or AWAY from the true product --
the same test `_PAIR_PROJ_BW = 16` was shipped on.

And it measures this card's DRAM copy roof, because every byte model in the state doc is checked
against it and a roof carried from another card is an assertion.
"""
from __future__ import annotations
import json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn                                                              # noqa: E402
import tt_bio.tenstorrent as T                                                  # noqa: E402
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor  # noqa: E402

if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
GRID = tuple(T.COMPUTE_GRID_MAIN)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)


def bench(fn, n=7, warm=2):
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
    return st.median(ts) * 1e3


def roofs():
    out = []
    for shape, lab in (((512, 512, 384), "opendde pair tensor"),
                       ((512, 512, 512), "trimul fused in-proj out"),
                       ((512, 512, 1152), "triatt qkv out")):
        t = ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ms = bench(lambda: ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG), n=9)
        by = shape[0] * shape[1] * shape[2] * 2
        out.append({"shape": list(shape), "label": lab, "GB": round(by / 1e9, 4),
                    "clone_ms": round(ms, 4),
                    "copy_roof_GBs": round(2 * by / (ms * 1e-3) / 1e9, 1)})
        print("  roof %-26s %8.4f ms  %6.1f GB/s (read+write)"
              % (lab, ms, 2 * by / (ms * 1e-3) / 1e9), flush=True)
        ttnn.deallocate(t)
    return out


# (label, rows, kt, nt)
CASES = [
    ("triatt qkv   [512,512,384]@[384,1152]", (512, 512), 12, 36),
    ("triatt gate  [512,512,384]@[384,384]", (512, 512), 12, 12),
    ("trimul inproj[512,512,384]@[384,512]", (512, 512), 12, 16),
]
# K_block sweep at a fixed, known-good M: which one reproduces the default's fold order?
KSWEEP = [(4, 1, 2, 1), (4, 2, 2, 1), (4, 3, 2, 1), (4, 4, 2, 1), (4, 6, 2, 1), (4, 12, 2, 1)]
# then the M/subblock ladder at whichever K wins, appended dynamically
MLADDER = [(1, 1, 1), (2, 2, 1), (4, 2, 1), (4, 4, 1), (8, 2, 1), (8, 4, 1)]


def stats(ref32, a, b):
    """rel-RMSD of a and b against the fp32 reference, and against each other."""
    d = lambda x, y: float((x - y).pow(2).mean().sqrt() / y.pow(2).mean().sqrt())
    return {"base_vs_fp32": round(d(a, ref32), 9), "cand_vs_fp32": round(d(b, ref32), 9),
            "cand_vs_base": round(d(b, a), 9),
            "closer_to_fp32": bool(d(b, ref32) < d(a, ref32))}


def main():
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "grid": list(GRID), "rows": []}
    print("=== roofs, ttnn.clone DRAM->DRAM, median of 9 after 2 warm ===", flush=True)
    res["roofs"] = roofs()

    for label, rows, kt, nt in CASES:
        K, N = kt * 32, nt * 32
        xt = torch.randn(*rows, K, dtype=torch.bfloat16)
        wt = torch.randn(K, N, dtype=torch.bfloat16)
        ref32 = (xt.float().reshape(-1, K) @ wt.float()).reshape(*rows, N)
        x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mt = 1
        for dd in [int(v) for v in x.shape][:-1]:
            mt *= dd
        mt = (mt + 31) // 32

        def run(c):
            return ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=CKC,
                dtype=ttnn.bfloat16, config=c)

        base_ms = bench(lambda: run(None))
        base = ttnn.to_torch(run(None)).float()
        row = {"case": label, "rows": list(rows), "kt": kt, "nt": nt, "mt": mt,
               "base_ms": round(base_ms, 4),
               "base_vs_fp32_relrmsd": round(float((base - ref32).pow(2).mean().sqrt()
                                                   / ref32.pow(2).mean().sqrt()), 9),
               "cands": []}
        print("\n== %s  mt=%d  base %.4f ms  base rel-RMSD vs fp32 %.3e"
              % (label, mt, base_ms, row["base_vs_fp32_relrmsd"]), flush=True)

        cands = [(M, Kb, sh, sw) for (M, Kb, sh, sw) in KSWEEP]
        cands += [(M, 12, sh, sw) for (M, sh, sw) in MLADDER if (M, 12, sh, sw) not in cands]
        for M, Kb, sh, sw in cands:
            c = {"entry": [M, Kb, 1, sh, sw]}
            if mt % M or kt % Kb:
                c["skipped"] = "production guard rejects it"
                row["cands"].append(c)
                print("   %-18s GUARD-REJECT" % (c["entry"],), flush=True)
                continue
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=M, K_block_size=Kb, N_block_size=1, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(*GRID))
            try:
                ms = bench(lambda: run(cfg))
                got = ttnn.to_torch(run(cfg))
                eq = bool(torch.equal(ttnn.to_torch(run(None)), got))
                g = got.float()
                c.update(ms=round(ms, 4), ratio=round(base_ms / ms, 4), equal=eq,
                         max_abs=round(float((g - base).abs().max()), 8), **stats(ref32, base, g))
                print("   %-18s %9.4f ms %.4fx eq=%-5s maxabs=%-6s cand/fp32 %.3e (base %.3e) closer=%s"
                      % (c["entry"], ms, base_ms / ms, eq, c["max_abs"], c["cand_vs_fp32"],
                         c["base_vs_fp32"], c["closer_to_fp32"]), flush=True)
                del got, g
            except Exception as e:                                              # noqa: BLE001
                c["error"] = str(e)[:200]
                print("   %-18s ERROR %s" % (c["entry"], str(e)[:110]), flush=True)
            row["cands"].append(c)
        del base, ref32
        ttnn.deallocate(x); ttnn.deallocate(w)
        res["rows"].append(row)

    p = Path(__file__).with_name("screen2.json")
    p.write_text(json.dumps(res, indent=1))
    print("\nwrote", p)


main()
