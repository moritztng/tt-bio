#!/usr/bin/env python3
"""THE SCREEN for arm C: a row-blocked, L1-resident pair transition.

Arm A (landed, 21.147 ms/call) still writes fc1's two d_ff-wide halves to DRAM and reads them
back: 3.892 GB per call, 42 % of the 431.1 GB/s roof, and 11.494 of the fold's 37.871 s. Arm C
blocks the call over rows so each half fits in L1 and never reaches DRAM. The floor is then a
COMPUTE floor of 3.66 ms/call (412 GFLOP at 112.7 TFLOP/s), not a memory one.

This screens the ACTUAL change: the real row-blocked chain at the production shape, several block
heights, each checked with torch.equal against the arm A path it would replace. Nothing is built
until this says so.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def bench(fn, n=7, warm=3):
    dev = T.get_device()
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    return st.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L, CZ, FF = a.L, 256, 1024
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    CK = ckc()
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "L": L, "arms": {}, "exact": {}, "l1_bytes": {}}

    torch.manual_seed(0)
    nw = f(torch.randn(CZ)); nb = f(torch.randn(CZ))
    w1a = f((torch.randn(FF, CZ) * 0.02).t())
    w1b = f((torch.randn(FF, CZ) * 0.02).t())
    w2 = f((torch.randn(CZ, FF) * 0.02).t())
    z = f(torch.randn(1, L, L, CZ))
    SILU = [ttnn.UnaryOpType.SILU]

    def lin(x, w, mc=None):
        return ttnn.linear(x, w, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                           core_grid=T.CORE_GRID_MAIN,
                           **({"memory_config": mc} if mc is not None else {}))

    def arm_a():
        xn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)
        h1, h2 = lin(xn, w1a), lin(xn, w1b)
        ttnn.deallocate(xn)
        gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU)
        ttnn.deallocate(h1); ttnn.deallocate(h2)
        out = lin(gt, w2)
        ttnn.deallocate(gt)
        return out

    def arm_c(rows, l1):
        mc = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG

        def go():
            xn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)
            parts = ttnn.chunk(xn, -(-L // rows), dim=1)
            ttnn.deallocate(xn)
            outs = []
            for p in parts:
                h1, h2 = lin(p, w1a, mc), lin(p, w1b, mc)
                ttnn.deallocate(p)
                gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU, memory_config=mc)
                ttnn.deallocate(h1); ttnn.deallocate(h2)
                outs.append(lin(gt, w2, ttnn.DRAM_MEMORY_CONFIG))
                ttnn.deallocate(gt)
            out = ttnn.concat(outs, dim=1)
            for o in outs:
                ttnn.deallocate(o)
            return out
        return go

    ref_t = ttnn.to_torch(arm_a())
    R["arms"]["A_landed"] = round(bench(arm_a), 3)
    print(f"  A (landed, DRAM)          {R['arms']['A_landed']:8.3f} ms", flush=True)

    # The engine's own L1-output route: `_pair_proj_linear(l1_out=True)` builds a tuned program
    # config at `_PAIR_PROJ_L1_BW` whose circular buffers are meant to leave room for an L1
    # output, which plain `ttnn.linear` does not. rows 8/16 are here because the refusal above is
    # a CB-vs-L1 clash with only ~216 KB per core left, so the block has to be small enough to fit
    # what the CBs leave, not just small enough to fit L1.
    def arm_c_engine(rows):
        def go():
            xn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)
            parts = ttnn.chunk(xn, -(-L // rows), dim=1)
            ttnn.deallocate(xn)
            outs = []
            for p_ in parts:
                h1 = T._pair_proj_linear(p_, w1a, CK, ttnn.bfloat16, l1_out=True)
                h2 = T._pair_proj_linear(p_, w1b, CK, ttnn.bfloat16, l1_out=True)
                ttnn.deallocate(p_)
                gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(h1); ttnn.deallocate(h2)
                outs.append(lin(gt, w2, ttnn.DRAM_MEMORY_CONFIG))
                ttnn.deallocate(gt)
            out = ttnn.concat(outs, dim=1)
            for o in outs:
                ttnn.deallocate(o)
            return out
        return go

    for rows in (8, 16, 32):
        key = f"C_engineL1_rows{rows}"
        R["l1_bytes"][key] = round(rows * L * FF * 2 * 3 / 1e6, 1)
        fn = arm_c_engine(rows)
        try:
            got = ttnn.to_torch(fn())
            R["exact"][key] = bool(torch.equal(ref_t, got))
            del got
            R["arms"][key] = round(bench(fn, n=5, warm=2), 3)
        except Exception as e:
            R["arms"][key] = None
            R["exact"][key] = f"REFUSED: {type(e).__name__}: {str(e)[:140]}"
        print(f"  {key:26s} {str(R['arms'][key]):>8s} ms  exact={R['exact'][key]} "
              f"(L1 need {R['l1_bytes'][key]} MB)", flush=True)
        a.out.write_text(json.dumps(R, indent=1))
        R["l1_refused_cache"] = len(T._L1_OUT_REFUSED)

    for rows in (32, 64, 128, 256):
        # bytes one row block's two d_ff halves + the gated result need resident in L1
        R["l1_bytes"][f"rows{rows}"] = round(rows * L * FF * 2 * 3 / 1e6, 1)
        for l1 in (True, False):
            key = f"C_rows{rows}_{'L1' if l1 else 'DRAM'}"
            fn = arm_c(rows, l1)
            try:
                got = ttnn.to_torch(fn())
                R["exact"][key] = bool(torch.equal(ref_t, got))
                del got
                R["arms"][key] = round(bench(fn), 3)
            except Exception as e:
                R["arms"][key] = None
                R["exact"][key] = f"REFUSED: {type(e).__name__}: {str(e)[:160]}"
            print(f"  {key:26s} {str(R['arms'][key]):>8s} ms  exact={R['exact'][key]} "
                  f"(L1 need {R['l1_bytes'][f'rows{rows}']} MB)", flush=True)
            a.out.write_text(json.dumps(R, indent=1))
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R["arms"], indent=1), flush=True)


if __name__ == "__main__":
    main()
