#!/usr/bin/env python3
"""Measured roofs on THIS card, then the two bodies that own 80 % of the ESMFold2 512 aa fold.

The fold decomposition says `body:SwiGLUFFN` (the pair transition) and
`body:TriangleMultiplication` are 35.9 s of a 45.816 s fold. This prices both against roofs
MEASURED here rather than inherited: a DRAM copy roof and a matmul roof at the fidelity and
core grid the model actually runs (HiFi4, fp32_dest_acc_en, 11x10), plus LoFi/HiFi2 and the
full 13x10 grid as the arms that name WHICH roof binds.

Nothing here proposes a change. It is Phase 1: name the binding limit per component.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def bench(fn, n=8, warm=3):
    dev = T.get_device()
    for _ in range(warm):
        out = fn()
        ttnn.synchronize_device(dev)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    return st.median(ts) * 1e3   # ms


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
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "core_grid_main": list(T.COMPUTE_GRID_MAIN), "L": L, "roofs": {},
         "bodies": {}, "arms": {}}
    GRID_MAIN = T.CORE_GRID_MAIN
    GRID_FULL = ttnn.CoreGrid(y=g.y, x=g.x)
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    # ---------------- roofs ----------------------------------------------------------------
    z = f(torch.randn(1, L, L, CZ))
    zb = f(torch.randn(1, L, L, CZ))
    zbytes = L * L * CZ * 2
    ms = bench(lambda: ttnn.clone(z, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    R["roofs"]["clone_GBs"] = round(2 * zbytes / (ms * 1e-3) / 1e9, 1)
    R["roofs"]["clone_ms"] = round(ms, 4)
    ms = bench(lambda: ttnn.add(z, zb, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    R["roofs"]["add_GBs"] = round(3 * zbytes / (ms * 1e-3) / 1e9, 1)
    R["roofs"]["add_ms"] = round(ms, 4)

    # square matmul roof, at the two fidelities and the two grids
    A = f(torch.randn(1, 4096, 4096)); B = f(torch.randn(1, 4096, 4096))
    flop = 2 * 4096 ** 3
    for fid in ("LoFi", "HiFi2", "HiFi4"):
        for fp32acc in (True, False):
            for gname, grid in (("main11x10", GRID_MAIN), ("full", GRID_FULL)):
                k = f"mm4096_{fid}_{'fp32acc' if fp32acc else 'nofp32acc'}_{gname}"
                try:
                    ms = bench(lambda: ttnn.matmul(A, B, compute_kernel_config=ckc(fid, fp32acc),
                                                   core_grid=grid, dtype=ttnn.bfloat16), n=5, warm=2)
                    R["roofs"][k] = round(flop / (ms * 1e-3) / 1e12, 2)
                except Exception as e:                                            # noqa: BLE001
                    R["roofs"][k] = f"ERR {type(e).__name__}"
    ttnn.deallocate(A); ttnn.deallocate(B)
    print(json.dumps(R["roofs"], indent=1), flush=True)
    a.out.write_text(json.dumps(R, indent=1))

    # ---------------- pair transition (body:SwiGLUFFN) --------------------------------------
    from tt_bio.esmc import SwiGLUFFN
    from tt_bio.tenstorrent import WeightScope
    sd = WeightScope({"0.weight": torch.randn(CZ), "0.bias": torch.randn(CZ),
                      "1.weight": torch.randn(2 * FF, CZ) * 0.02,
                      "3.weight": torch.randn(CZ, FF) * 0.02})
    ffn = SwiGLUFFN(sd, ckc(), fuse_swiglu=True)
    R["bodies"]["swiglu_fused_flag"] = ffn.fuse_swiglu
    R["bodies"]["pair_row_tile"] = T.pair_row_tile(L)
    R["bodies"]["swiglu_ms"] = round(bench(lambda: ffn(z)), 4)
    # the three internal ops, at the same shapes
    ln = lambda: ttnn.layer_norm(z, weight=ffn.norm_weight, bias=ffn.norm_bias, epsilon=1e-5,
                                 compute_kernel_config=ffn.compute_kernel_config)
    R["bodies"]["swiglu_layer_norm_ms"] = round(bench(ln), 4)
    zn = ln()
    R["bodies"]["swiglu_mm1_fused_ms"] = round(bench(
        lambda: ttnn.experimental.minimal_matmul(input_tensor=zn, weight_tensor=ffn.fc1_weight,
                                                 compute_kernel_config=ffn.compute_kernel_config,
                                                 dtype=ffn.fc1_weight.dtype, fuse_swiglu=True)), 4)
    gated = f(torch.randn(1, L, L, FF))
    R["bodies"]["swiglu_mm2_ms"] = round(bench(
        lambda: ttnn.linear(gated, ffn.fc2_weight, compute_kernel_config=ffn.compute_kernel_config,
                            dtype=ttnn.bfloat16, core_grid=GRID_MAIN)), 4)
    # arms on mm2: fidelity / grid, to name which roof binds
    for fid in ("LoFi", "HiFi2", "HiFi4"):
        for gname, grid in (("main11x10", GRID_MAIN), ("full", GRID_FULL)):
            try:
                R["arms"][f"mm2_{fid}_{gname}_ms"] = round(bench(
                    lambda: ttnn.linear(gated, ffn.fc2_weight, compute_kernel_config=ckc(fid),
                                        dtype=ttnn.bfloat16, core_grid=grid)), 4)
            except Exception as e:                                                # noqa: BLE001
                R["arms"][f"mm2_{fid}_{gname}_ms"] = f"ERR {type(e).__name__}"
    for fid in ("LoFi", "HiFi2", "HiFi4"):
        try:
            R["arms"][f"mm1_{fid}_ms"] = round(bench(
                lambda: ttnn.experimental.minimal_matmul(
                    input_tensor=zn, weight_tensor=ffn.fc1_weight, compute_kernel_config=ckc(fid),
                    dtype=ffn.fc1_weight.dtype, fuse_swiglu=True)), 4)
        except Exception as e:                                                    # noqa: BLE001
            R["arms"][f"mm1_{fid}_ms"] = f"ERR {type(e).__name__}"
    ttnn.deallocate(zn); ttnn.deallocate(gated)
    print(json.dumps(R["bodies"], indent=1), flush=True)
    a.out.write_text(json.dumps(R, indent=1))

    # ---------------- triangle multiplication ----------------------------------------------
    from tt_bio.tenstorrent import TriangleMultiplication
    tsd = WeightScope({"norm_in.weight": torch.randn(CZ), "norm_in.bias": torch.randn(CZ),
                       "norm_out.weight": torch.randn(CZ), "norm_out.bias": torch.randn(CZ),
                       "p_in.weight": torch.randn(2 * HID, CZ) * 0.02,
                       "g_in.weight": torch.randn(2 * HID, CZ) * 0.02,
                       "p_out.weight": torch.randn(CZ, CZ) * 0.02,
                       "g_out.weight": torch.randn(CZ, CZ) * 0.02})
    for ending in (False, True):
        tm = TriangleMultiplication(ending, tsd, ckc())
        R["bodies"][f"trimul_{'end' if ending else 'start'}_ms"] = round(bench(lambda: tm(z, None), n=6), 4)
    R["bodies"]["trimul_chunk_size"] = T._trimul_chunk_size(L, HID, 1)
    mc = T._triangle_mul_memory_config(L)
    R["bodies"]["trimul_memcfg"] = str(mc.buffer_type)
    C = R["bodies"]["trimul_chunk_size"]
    grp = T._trimul_inproj_group(L, C, 1, HID // C) if mc.buffer_type == ttnn.BufferType.DRAM else 1
    R["bodies"]["trimul_inproj_group"] = grp
    R["bodies"]["trimul_n_pairs"] = HID // C
    # the per-channel contraction, at the real chunk width
    ac = f(torch.randn(1, C, L, L)); bc = f(torch.randn(1, C, L, L))
    pc = T._triangle_mul_program_config((L + 31) // 32)
    R["bodies"]["trimul_contract_ms"] = round(bench(
        lambda: ttnn.matmul(ac, bc, compute_kernel_config=ckc(), memory_config=mc,
                            program_config=pc, dtype=ttnn.bfloat16), n=6), 4)
    for fid in ("LoFi", "HiFi2", "HiFi4"):
        try:
            R["arms"][f"trimul_contract_{fid}_ms"] = round(bench(
                lambda: ttnn.matmul(ac, bc, compute_kernel_config=ckc(fid), memory_config=mc,
                                    program_config=pc, dtype=ttnn.bfloat16), n=6), 4)
        except Exception as e:                                                    # noqa: BLE001
            R["arms"][f"trimul_contract_{fid}_ms"] = f"ERR {type(e).__name__}"
    # the fused in-projection, at the real (chunk, group)
    gp = tm._gp_in_chunks(C, grp)[0]
    R["bodies"]["trimul_inproj_ms"] = round(bench(
        lambda: ttnn.experimental.minimal_matmul(z, gp, memory_config=mc, dtype=ttnn.bfloat16,
                                                 compute_kernel_config=ckc()), n=6), 4)
    R["bodies"]["trimul_inproj_nwide"] = int(gp.shape[-1])
    print(json.dumps(R["bodies"], indent=1), flush=True)
    print(json.dumps(R["arms"], indent=1), flush=True)
    a.out.write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
