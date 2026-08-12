#!/usr/bin/env python3
"""_MM_BLOCK sweep at OpenDDE's REAL 512 aa shapes.

`_MM_BLOCK` deliberately has no (12, 36) / (12, 12) entry. The stated reason (tenstorrent.py:2222)
is that opendde's 995 tokens give mt = 30939, which 4 does not divide, so only M_block = 1 passes
`_qkv_mm_config`'s `mt % M` guard, and M_block = 1 measured 0.5794x / 0.5318x.

That reasoning is anchored on 995 tokens. The MEASURED 512 aa fold (perf/other512/ab_opendde_512.json,
main c85255f5) presents (512, 512, 384) on 1048 of 1216 triangle-attention calls and (995, 995, 384)
on 8. At S = 512, mt = 512*512/32 = 8192 = 2^13, which 4 and 8 both divide. So the guard that
excluded opendde does not fire at the size this task is about, and the config that was measured to
lose is not the config the table would hold.

Every candidate keeps K_block == kt == 12: one K block for the whole contraction, so the
accumulation order is the unconfigured op's. That is the property that made all six shipped entries
bit-exact and it is the reason to predict these will be.

GO/NO-GO, pre-committed here before the numbers exist:
  GO on an entry iff torch.equal holds AND the op ratio is >= 1.03x at S = 512.
  A candidate that is faster but not torch.equal is REJECTED outright, not traded off.
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

# (label, rows, kt, nt) -- kt/nt read off the census in ab_opendde_512.json, not from source.
CASES = [
    ("qkv  c=384 S=512", (512, 512), 12, 36),
    ("gate c=384 S=512", (512, 512), 12, 12),
    ("qkv  c=384 S=576", (576, 576), 12, 36),
    ("qkv  c=384 S=640", (640, 640), 12, 36),
    ("qkv  c=384 S=995", (995, 995), 12, 36),
]
# (M_block, subblock_h, N_block, subblock_w). K_block is always kt.
CANDS = [(1,1,1,1), (2,1,1,1), (2,2,1,1), (4,1,1,1), (4,2,1,1), (4,4,1,1),
         (8,2,1,1), (8,4,1,1), (4,2,2,2), (4,1,3,3), (3,3,1,1), (3,1,1,1)]


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


def main():
    out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"), "grid": list(GRID), "rows": []}
    for label, rows, kt, nt in CASES:
        K, N = kt * 32, nt * 32
        x = ttnn.from_torch(torch.randn(*rows, K, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(torch.randn(K, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mt = 1
        for d in [int(v) for v in x.shape][:-1]:
            mt *= d
        mt = (mt + 31) // 32

        def run(c):
            return ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=CKC,
                dtype=ttnn.bfloat16, config=c)

        base = bench(lambda: run(None))
        ref = ttnn.to_torch(run(None))
        row = {"case": label, "rows": list(rows), "kt": kt, "nt": nt, "mt": mt,
               "base_ms": round(base, 4), "cands": []}
        print(f"\n{label}  mt={mt}  base {base:.4f} ms", flush=True)
        for M, sh, Nb, sw in CANDS:
            g = {"mt_mod_M": mt % M, "nt_mod_N": nt % Nb, "kt_mod_K": kt % kt}
            c = {"entry": [M, kt, Nb, sh, sw], "guards": g}
            if mt % M or nt % Nb:
                c["skipped"] = "production guard `mt % M or nt % N` rejects it"
                row["cands"].append(c)
                print("  %s GUARD-REJECT" % c["entry"], flush=True)
                continue
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=M, K_block_size=kt, N_block_size=Nb, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(*GRID))
            try:
                t = bench(lambda: run(cfg))
                got = ttnn.to_torch(run(cfg))
                eq = bool(torch.equal(ref, got))
                c.update(ms=round(t, 4), ratio=round(base / t, 4), equal=eq)
                if not eq:
                    d = (ref.float() - got.float())
                    c["max_abs"] = round(d.abs().max().item(), 8)
                del got
                print(f"  {c['entry']} {t:9.4f} ms  {base/t:6.4f}x  equal={eq}", flush=True)
            except Exception as e:
                c["error"] = str(e)[:200]
                print(f"  {c['entry']} ERROR {str(e)[:120]}", flush=True)
            row["cands"].append(c)
        del ref
        ttnn.deallocate(x); ttnn.deallocate(w)
        out["rows"].append(row)
    p = Path(__file__).with_name("mm_sweep.json")
    p.write_text(json.dumps(out, indent=1))
    print("\nwrote", p)


main()
