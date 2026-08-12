#!/usr/bin/env python3
"""THE SCREEN for ESMFold2 512 aa: price the pair transition op-by-op, then measure the
ACTUAL candidate rewrite at the production shape before anything is built.

Fold decomposition (perf/esm512/decomp_512_c2.json, 45.816 s median, A/A 0.156 s):
  body:SwiGLUFFN            538 pair calls   15.9 s   34.7 % of the fold
  body:TriangleMultiplication 1076 calls     19.5 s   42.6 %
Measured roofs on this card (perf/esm512/roofs_512_c2.json): DRAM add 431.1 GB/s,
matmul bf16 4096^3 HiFi4 112.68 TFLOP/s => machine balance 261 FLOP/byte.

`SwiGLUFFN.fuse_swiglu` is FALSE on ttnn 0.68.0: the flag is gated on the string "fuse_swiglu"
appearing in `ttnn.experimental.minimal_matmul.__doc__`, and this wheel's minimal_matmul has no
such kwarg. So production materialises the 2*d_ff intermediate and then walks it three more
times (chunk, silu, multiply). This script measures exactly what that costs and what each
candidate rewrite actually returns, and checks every candidate with torch.equal, not PCC.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def bench(fn, n=10, warm=3):
    dev = T.get_device()
    outs = []
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


def ckc(fid="HiFi4", fp32acc=True):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fid),
        math_approx_mode=False, fp32_dest_acc_en=fp32acc, packer_l1_acc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L, CZ, HID, FF = a.L, 256, 256, 1024
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    GRID = T.CORE_GRID_MAIN
    CK = ckc()
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "L": L, "small_grid": T._IS_SMALL_GRID,
         "pair_row_tile": T.pair_row_tile(L), "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
         "ops": {}, "chain": {}, "exact": {}, "trimul": {}}
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    MB = 1e6
    zbytes = L * L * CZ * 2 / MB          # 134.2 MB pair tensor
    hbytes = L * L * 2 * FF * 2 / MB      # 1073.7 MB fc1 output
    gbytes = L * L * FF * 2 / MB          # 536.9 MB gated

    from tt_bio.esmc import SwiGLUFFN
    from tt_bio.tenstorrent import WeightScope
    torch.manual_seed(0)
    w12 = (torch.randn(2 * FF, CZ) * 0.02)
    w3 = (torch.randn(CZ, FF) * 0.02)
    sd = WeightScope({"0.weight": torch.randn(CZ), "0.bias": torch.randn(CZ),
                      "1.weight": w12, "3.weight": w3})
    ffn = SwiGLUFFN(sd, CK, fuse_swiglu=True)
    R["fuse_swiglu_flag"] = ffn.fuse_swiglu
    R["minimal_matmul_has_fuse_swiglu"] = "fuse_swiglu" in (
        ttnn.experimental.minimal_matmul.__doc__ or "")

    z = f(torch.randn(1, L, L, CZ))
    R["chain"]["baseline_ms"] = round(bench(lambda: ffn(z)), 4)

    # ---- op-by-op, production path ---------------------------------------------------------
    ln = lambda: ttnn.layer_norm(z, weight=ffn.norm_weight, bias=ffn.norm_bias, epsilon=1e-5,
                                 compute_kernel_config=CK)
    R["ops"]["layer_norm"] = {"ms": round(bench(ln), 4), "MB": round(2 * zbytes, 1)}
    zn = ln()
    mm1 = lambda: ttnn.linear(zn, ffn.fc1_weight, compute_kernel_config=CK,
                              dtype=ttnn.bfloat16, core_grid=GRID)
    R["ops"]["fc1_linear_N2048"] = {"ms": round(bench(mm1), 4), "MB": round(zbytes + hbytes, 1),
                                    "GFLOP": round(2 * L * L * CZ * 2 * FF / 1e9, 1)}
    h = mm1()
    R["ops"]["chunk2"] = {"ms": round(bench(lambda: ttnn.chunk(h, 2, dim=-1)[0]), 4),
                          "MB": round(2 * hbytes, 1)}
    x1, x2 = ttnn.chunk(h, 2, dim=-1)
    R["ops"]["silu"] = {"ms": round(bench(lambda: ttnn.silu(x1)), 4), "MB": round(2 * gbytes, 1)}
    s1 = ttnn.silu(x1)
    R["ops"]["multiply"] = {"ms": round(bench(lambda: ttnn.multiply(s1, x2)), 4),
                            "MB": round(3 * gbytes, 1)}
    gated = ttnn.multiply(s1, x2)
    mm2 = lambda: ttnn.linear(gated, ffn.fc2_weight, compute_kernel_config=CK,
                              dtype=ttnn.bfloat16, core_grid=GRID)
    R["ops"]["fc2_linear_N256"] = {"ms": round(bench(mm2), 4), "MB": round(gbytes + zbytes, 1),
                                   "GFLOP": round(2 * L * L * FF * CZ / 1e9, 1)}
    ref_out = ttnn.to_torch(mm2())

    # ---- candidate B: ttnn.swiglu replaces chunk+silu+multiply ------------------------------
    try:
        R["ops"]["ttnn_swiglu"] = {"ms": round(bench(lambda: ttnn.swiglu(h, -1)), 4),
                                   "MB": round(hbytes + gbytes, 1)}
        R["exact"]["ttnn_swiglu_vs_chunk_silu_mul"] = bool(
            torch.equal(ttnn.to_torch(ttnn.swiglu(h, -1)), ttnn.to_torch(gated)))
    except Exception as e:                                                        # noqa: BLE001
        R["ops"]["ttnn_swiglu"] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    # multiply with SiLU fused into operand A
    try:
        mA = lambda: ttnn.multiply(x1, x2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        R["ops"]["multiply_siluA"] = {"ms": round(bench(mA), 4), "MB": round(3 * gbytes, 1)}
        R["exact"]["multiply_siluA_vs_silu_then_mul"] = bool(
            torch.equal(ttnn.to_torch(mA()), ttnn.to_torch(gated)))
    except Exception as e:                                                        # noqa: BLE001
        R["ops"]["multiply_siluA"] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    ttnn.deallocate(h); ttnn.deallocate(x1); ttnn.deallocate(x2)
    ttnn.deallocate(s1); ttnn.deallocate(gated)

    # ---- candidate A: split fc1 into two N=1024 matmuls, no chunk ---------------------------
    wa = f(w12[:FF].t().contiguous())
    wb = f(w12[FF:].t().contiguous())
    R["ops"]["fc1_half_N1024"] = {"ms": round(bench(
        lambda: ttnn.linear(zn, wa, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                            core_grid=GRID)), 4), "MB": round(zbytes + gbytes, 1)}
    # silu fused into the first half-matmul's epilogue
    try:
        R["ops"]["fc1_half_N1024_silu_epilogue"] = {"ms": round(bench(
            lambda: ttnn.linear(zn, wa, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                                core_grid=GRID, activation="silu")), 4),
            "MB": round(zbytes + gbytes, 1)}
    except Exception as e:                                                        # noqa: BLE001
        R["ops"]["fc1_half_N1024_silu_epilogue"] = f"ERR {type(e).__name__}: {str(e)[:200]}"

    def cand_split():
        zz = ttnn.layer_norm(z, weight=ffn.norm_weight, bias=ffn.norm_bias, epsilon=1e-5,
                             compute_kernel_config=CK)
        ha = ttnn.linear(zz, wa, compute_kernel_config=CK, dtype=ttnn.bfloat16, core_grid=GRID)
        hb = ttnn.linear(zz, wb, compute_kernel_config=CK, dtype=ttnn.bfloat16, core_grid=GRID)
        ttnn.deallocate(zz)
        gt = ttnn.multiply(ha, hb, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(ha); ttnn.deallocate(hb)
        o = ttnn.linear(gt, ffn.fc2_weight, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                        core_grid=GRID)
        ttnn.deallocate(gt)
        return o

    try:
        R["chain"]["cand_split_ms"] = round(bench(cand_split), 4)
        R["exact"]["cand_split_vs_baseline"] = bool(torch.equal(ttnn.to_torch(cand_split()), ref_out))
    except Exception as e:                                                        # noqa: BLE001
        R["chain"]["cand_split_ms"] = f"ERR {type(e).__name__}: {str(e)[:300]}"

    def cand_swiglu():
        zz = ttnn.layer_norm(z, weight=ffn.norm_weight, bias=ffn.norm_bias, epsilon=1e-5,
                             compute_kernel_config=CK)
        hh = ttnn.linear(zz, ffn.fc1_weight, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                         core_grid=GRID)
        ttnn.deallocate(zz)
        gt = ttnn.swiglu(hh, -1)
        ttnn.deallocate(hh)
        o = ttnn.linear(gt, ffn.fc2_weight, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                        core_grid=GRID)
        ttnn.deallocate(gt)
        return o

    try:
        R["chain"]["cand_swiglu_ms"] = round(bench(cand_swiglu), 4)
        R["exact"]["cand_swiglu_vs_baseline"] = bool(torch.equal(ttnn.to_torch(cand_swiglu()), ref_out))
    except Exception as e:                                                        # noqa: BLE001
        R["chain"]["cand_swiglu_ms"] = f"ERR {type(e).__name__}: {str(e)[:300]}"
    ttnn.deallocate(zn)
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in ("ops", "chain", "exact")}, indent=1), flush=True)

    # ---- trimul, and the E6 gate at this shape ---------------------------------------------
    from tt_bio.tenstorrent import TriangleMultiplication
    import tt_bio.reblock_permute as RB
    tsd = WeightScope({"norm_in.weight": torch.randn(CZ), "norm_in.bias": torch.randn(CZ),
                       "norm_out.weight": torch.randn(CZ), "norm_out.bias": torch.randn(CZ),
                       "p_in.weight": torch.randn(2 * HID, CZ) * 0.02,
                       "g_in.weight": torch.randn(2 * HID, CZ) * 0.02,
                       "p_out.weight": torch.randn(CZ, CZ) * 0.02,
                       "g_out.weight": torch.randn(CZ, CZ) * 0.02})
    mc = T._triangle_mul_memory_config(L)
    C = T._trimul_chunk_size(L, HID, 1)
    R["trimul"]["chunk_size"] = C
    R["trimul"]["memcfg"] = str(mc.buffer_type)
    R["trimul"]["n_pairs"] = HID // C
    R["trimul"]["group"] = (T._trimul_inproj_group(L, C, 1, HID // C)
                            if mc.buffer_type == ttnn.BufferType.DRAM else 1)
    for ending in (False, True):
        tm = TriangleMultiplication(ending, tsd, CK)
        tag = "end" if ending else "start"
        for e6 in (False, True):
            RB.set_enabled_gated(e6)
            RB.STATS_GATED[0] = RB.STATS_GATED[1] = 0
            ms = bench(lambda: tm(z, None), n=6)
            R["trimul"][f"{tag}_e6{int(e6)}_ms"] = round(ms, 4)
            R["trimul"][f"{tag}_e6{int(e6)}_gated_served_declined"] = list(RB.STATS_GATED)
        RB.set_enabled_gated(False)
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R["trimul"], indent=1), flush=True)


if __name__ == "__main__":
    main()
