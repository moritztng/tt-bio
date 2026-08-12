#!/usr/bin/env python3
"""Screen for the `_MM_BLOCK` re-key: is a swept block config bit-exact and faster at the widths
boltz2, openfold3 and opendde actually present?

`_MM_BLOCK` is keyed on `nt` alone and holds two entries, `{24: (4,8,1,4,1), 8: (4,8,1,4,1)}`, both
of them protenix-v2's. The fold census MEASURED what the other models hand `_qkv_mm_config`, and
every one of them misses:

    boltz2      (kt=4, nt=12) x1120   (kt=4, nt=4)  x560
    openfold3   (kt=4, nt=12) x848    (kt=4, nt=4)  x424
                (kt=2, nt=12) x128    (kt=2, nt=2)  x64
    opendde     (kt=12, nt=36)        (kt=12, nt=12)          [from source; opendde c_z=384]

Re-keying on `(kt, nt)` is required for correctness, not tidiness. Under today's `nt`-only key,
adding `nt=12` with `K_block=4` for boltz2/openfold3 would ALSO be picked up by opendde's gate
projection, whose kt is 12: the guard is `kt % blk[1]`, and `12 % 4 == 0` passes it. opendde would
silently get 3 K blocks, a different accumulation order and a lost bit-exactness.

Every proposed entry sets `K_block == kt`, so the whole contraction is one K block and the
accumulation order is the same as the unconfigured op -- which is exactly why the two shipped
entries were bit-exact, and it is the reason to predict these will be too.

This is screens 1 and 3 of the plan's three. Screen 2 (the K1 `generic_op` transcription at the new
widths) only matters if this one passes, because K1 refuses outright without an `_MM_BLOCK` entry.

GO/NO-GO, pre-committed by the plan and not relitigated here: GO iff every shape is `torch.equal`
AND the timing shows >= 1.03x on the qkv or gate op.
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
GRID = tuple(T.COMPUTE_GRID_MAIN)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)

# (label, x rows-shape, kt, nt, calls per fold from the census). K_block is always kt.
CASES = [
    ("boltz2/of3 qkv  c_z=128", (512, 512), 4, 12, 1120 + 848),
    ("boltz2/of3 gate c_z=128", (512, 512), 4, 4, 560 + 424),
    ("of3 qkv         c_z=64",  (512, 512), 2, 12, 128),
    ("of3 gate        c_z=64",  (512, 512), 2, 2, 64),
    ("opendde qkv     c_z=384", (995, 995), 12, 36, None),
    ("opendde gate    c_z=384", (995, 995), 12, 12, None),
]


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
    return st.median(ts) * 1e3


def main():
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(GRID), "mm_block_today": {str(k): list(v) for k, v in T._MM_BLOCK.items()},
           "rows": []}

    for label, rows, kt, nt, calls in CASES:
        K, N = kt * 32, nt * 32
        x = ttnn.from_torch(torch.randn(*rows, K, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(torch.randn(K, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mt = 1
        for d in [int(v) for v in x.shape][:-1]:
            mt *= d
        mt = (mt + 31) // 32          # CEIL, exactly as `_qkv_mm_config` does it
        # Production computes mt with a CEIL and then refuses any config whose M_block does not
        # divide it (`_qkv_mm_config`: `if mt % M or nt % N: return None`). The first screen built
        # the config directly and so tested one production would never construct -- opendde's 995
        # tokens give mt = 30939, which 4 does not divide. Pick the first M_block that passes the
        # real guard, exactly as an entry in the table would have to.
        M = next((m for m in (4, 2, 1) if mt % m == 0), None)
        assert M is not None
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=M, K_block_size=kt, N_block_size=1, subblock_h=M, subblock_w=1,
            compute_with_storage_grid_size=ttnn.CoreCoord(*GRID))
        row = {"case": label, "kt": kt, "nt": nt, "mt": mt, "calls_per_fold": calls,
               "entry": [M, kt, 1, M, 1],
               "guards": {"kt_mod_Kblock": kt % kt, "mt_mod_M": mt % M, "M_block": M}}

        def run(c):
            return ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=CKC,
                dtype=ttnn.bfloat16, config=c)

        try:
            a = run(None)
            b = run(cfg)
            ta, tb = ttnn.to_torch(a), ttnn.to_torch(b)
            row["equal"] = bool(torch.equal(ta, tb))
            if not row["equal"]:
                d = (ta.float() - tb.float())
                row["max_abs"] = round(d.abs().max().item(), 8)
                row["rmsd_over_std"] = round((d.pow(2).mean().sqrt() / ta.float().std()).item(), 9)
            ttnn.deallocate(a); ttnn.deallocate(b)
            row["ms_unconfigured"] = round(bench(lambda: run(None)), 4)
            row["ms_configured"] = round(bench(lambda: run(cfg)), 4)
            row["speedup"] = round(row["ms_unconfigured"] / row["ms_configured"], 4)
            row["saving_ms"] = round(row["ms_unconfigured"] - row["ms_configured"], 4)
        except Exception as e:                                                # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        ttnn.deallocate(x); ttnn.deallocate(w)

    ok = [r for r in res["rows"] if "error" not in r]
    res["all_bit_exact"] = bool(ok) and all(r.get("equal") for r in ok)
    res["max_speedup"] = max((r.get("speedup", 0) for r in ok), default=0)
    res["verdict"] = ("GO" if res["all_bit_exact"] and res["max_speedup"] >= 1.03 else "NO-GO")
    print(json.dumps({k: res[k] for k in ("all_bit_exact", "max_speedup", "verdict")}), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[1], flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
