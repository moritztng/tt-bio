#!/usr/bin/env python3
"""Per-op wall of ESMFold2's TriangleMultiplication at the production shape (L=512, c_z=256).

TriMul is 16.071 s = 42 % of the 37.871 s fold and sits at ~60 % of the measured DRAM roof. That
verdict came from a whole-op wall and a byte model; it does not say WHICH op is off its roof. This
screen times every op in the E6 channel-loop path individually, at the exact shapes the fold runs,
and checks the sum against the whole-op wall so the transcription is verified rather than asserted.

It also prices, in the same session, every cheap structural alternative the plan considers:
transpose_b on the channel matmul, an L1-resident row-blocked in-projection, a fused
gated-residual op, and the non-gated (pre-E6) path for reference.

Nothing here is a fold result. It is a screen whose job is to say which change is worth building.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RP
from tt_bio.tenstorrent import WeightScope

MB = 2 ** 20


def ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--L", type=int, default=512)
    a = ap.parse_args()

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    CK = ckc()
    L, CZ = a.L, 256
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "L": L, "c_z": CZ, "n": a.n, "warm": a.warm, "ops": {}, "probe": {}}
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def timed(make, label, bytes_mb=None, flops=None, keep=False):
        """Median ms over a.n synced calls of make(), warm first. Results freed unless keep."""
        outs = []
        for i in range(a.warm + a.n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = make()
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) * 1e3
            if i >= a.warm:
                outs.append(dt)
            if not keep and isinstance(r, ttnn.Tensor):
                ttnn.deallocate(r)
        ms = statistics.median(outs)
        rec = {"ms": round(ms, 4), "all_ms": [round(x, 4) for x in outs]}
        if bytes_mb:
            rec["MB"] = round(bytes_mb, 1)
            rec["GBps"] = round(bytes_mb * MB / (ms * 1e-3) / 1e9, 1)
        if flops:
            rec["GFLOP"] = round(flops / 1e9, 1)
            rec["TFLOPps"] = round(flops / (ms * 1e-3) / 1e12, 1)
        R["ops"][label] = rec
        print(f"  {label:28s} {ms:8.3f} ms"
              + (f"  {rec['GBps']:7.1f} GB/s" if bytes_mb else "")
              + (f"  {rec['TFLOPps']:6.1f} TFLOP/s" if flops else ""), flush=True)
        return ms

    # ---------------- the roof, in this session, on this card -------------------------------
    pair_mb = L * L * CZ * 2 / MB          # one pair tensor, bf16
    z0, z1 = f(torch.randn(1, L, L, CZ)), f(torch.randn(1, L, L, CZ))
    timed(lambda: ttnn.add(z0, z1), "roof_add_3x_pair", 3 * pair_mb)
    print(f"  [pair tensor = {pair_mb:.1f} MB]", flush=True)

    # ---------------- the real object, whole-op wall ----------------------------------------
    torch.manual_seed(0)
    tsd = WeightScope({
        "norm_in.weight": torch.randn(CZ), "norm_in.bias": torch.randn(CZ),
        "norm_out.weight": torch.randn(CZ), "norm_out.bias": torch.randn(CZ),
        "g_in.weight": torch.randn(2 * CZ, CZ) * 0.02,
        "p_in.weight": torch.randn(2 * CZ, CZ) * 0.02,
        "g_out.weight": torch.randn(CZ, CZ) * 0.02,
        "p_out.weight": torch.randn(CZ, CZ) * 0.02})
    tm = T.TriangleMultiplication(False, tsd, CK, gated_moves=True)   # tri_mul_out (ending=False)
    tm_end = T.TriangleMultiplication(True, tsd, CK, gated_moves=True)  # tri_mul_in
    x = f(torch.randn(1, L, L, CZ) * 0.5)

    chunk_size = T._trimul_chunk_size(L, tm._hidden, 1)
    n_pairs = tm._hidden // chunk_size
    large = T._triangle_mul_memory_config(L).buffer_type == ttnn.BufferType.DRAM
    group = T._trimul_inproj_group(L, chunk_size, 1, n_pairs) if large else 1
    R["shape"] = {"chunk_size": chunk_size, "n_pairs": n_pairs, "group": group,
                  "groups": n_pairs // group, "dram": large,
                  "host_concat": bool(large and T._host_concat(x))}
    print(f"  shape gate: chunk={chunk_size} n_pairs={n_pairs} group={group} "
          f"groups={n_pairs // group} dram={large} host_concat={R['shape']['host_concat']}",
          flush=True)

    prev = RP.set_enabled_gated(True)
    timed(lambda: tm(x), "WHOLE_trimul_e6_on")
    timed(lambda: tm_end(x), "WHOLE_trimul_e6_on_end")
    RP.set_enabled_gated(False)
    timed(lambda: tm(x), "WHOLE_trimul_e6_off")
    RP.set_enabled_gated(prev)

    # ---------------- op by op, at the exact shapes the loop runs ----------------------------
    MC = ttnn.DRAM_MEMORY_CONFIG
    ln = lambda t, w, b: ttnn.layer_norm(t, weight=w, bias=b, epsilon=1e-5,
                                         compute_kernel_config=CK)
    timed(lambda: ln(x, tm.in_norm_weight, tm.in_norm_bias), "1_layer_norm_in", 2 * pair_mb)
    x_norm = ln(x, tm.in_norm_weight, tm.in_norm_bias)

    w_gp = tm._gp_in_chunks(chunk_size, group)[0]           # [256, 1024]
    n_wide = int(w_gp.shape[-1])
    inproj_mb = pair_mb + pair_mb * n_wide / CZ
    inproj_fl = 2 * L * L * CZ * n_wide
    timed(lambda: ttnn.experimental.minimal_matmul(
        x_norm, w_gp, memory_config=MC, dtype=ttnn.bfloat16, compute_kernel_config=CK),
        "2_inproj_minimal_mm_N%d" % n_wide, inproj_mb, inproj_fl)
    gp = ttnn.experimental.minimal_matmul(x_norm, w_gp, memory_config=MC, dtype=ttnn.bfloat16,
                                          compute_kernel_config=CK)
    sc = int(gp.shape[-1]) // 4
    R["shape"]["slice_c"] = sc
    assert RP.eligible_gated(gp, sc, MC), "E6 declines at this shape -- screen invalid"

    # perm_a = (0,3,1,2) for ending=False: the gated move alone. perm_b = (0,3,2,1): move+transpose.
    timed(lambda: RP.reblock_permute_gated(gp, 2 * sc, 0, sc, memory_config=MC),
          "3_gated_move", 2 * pair_mb + pair_mb)
    a_ch = RP.reblock_permute_gated(gp, 2 * sc, 0, sc, memory_config=MC)
    b_ch = RP.reblock_permute_gated(gp, 3 * sc, sc, sc, memory_config=MC)
    timed(lambda: ttnn.transpose(b_ch, -2, -1, memory_config=MC),
          "4_transpose_inner", 2 * pair_mb)
    b_t = ttnn.transpose(b_ch, -2, -1, memory_config=MC)

    pc = T._triangle_mul_program_config((L + 31) // 32)
    mm_fl = 2 * CZ * L * L * L
    timed(lambda: ttnn.matmul(a_ch, b_t, compute_kernel_config=CK, memory_config=MC,
                              program_config=pc, dtype=ttnn.bfloat16),
          "5_channel_matmul", 3 * pair_mb, mm_fl)
    x_ch = ttnn.matmul(a_ch, b_t, compute_kernel_config=CK, memory_config=MC,
                       program_config=pc, dtype=ttnn.bfloat16)
    timed(lambda: T._channel_move_back(x_ch, MC), "6_channel_move_back", 2 * pair_mb)
    x_mix = T._channel_move_back(x_ch, MC)
    timed(lambda: ln(x_mix, tm.out_norm_weight, tm.out_norm_bias), "7_layer_norm_out", 2 * pair_mb)
    x_ln = ln(x_mix, tm.out_norm_weight, tm.out_norm_bias)
    proj_fl = 2 * L * L * CZ * CZ
    timed(lambda: T._trimul_out_proj(x_ln, tm.out_p_weight, CK), "8_out_proj", 2 * pair_mb, proj_fl)
    p_out = T._trimul_out_proj(x_ln, tm.out_p_weight, CK)
    g_out = T._trimul_out_proj(x_norm, tm.g_out_weight, CK)
    timed(lambda: ttnn.multiply(p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
          "9_gate_multiply", 3 * pair_mb)

    # ---------------- probes: the structural alternatives, priced now ------------------------
    # P1: skip the standalone inner transpose by asking the matmul to transpose its rhs.
    try:
        m = timed(lambda: ttnn.matmul(a_ch, b_ch, transpose_b=True, compute_kernel_config=CK,
                                      memory_config=MC, dtype=ttnn.bfloat16),
                  "P1_matmul_transpose_b", 3 * pair_mb, mm_fl)
        ref = ttnn.to_torch(ttnn.matmul(a_ch, b_t, compute_kernel_config=CK, memory_config=MC,
                                        program_config=pc, dtype=ttnn.bfloat16))
        alt = ttnn.to_torch(ttnn.matmul(a_ch, b_ch, transpose_b=True, compute_kernel_config=CK,
                                        memory_config=MC, dtype=ttnn.bfloat16))
        R["probe"]["P1_transpose_b"] = {"ok": True, "ms": m,
                                        "torch_equal": bool(torch.equal(ref, alt)),
                                        "max_abs_diff": float((ref - alt).abs().max())}
        del ref, alt
    except Exception as e:
        R["probe"]["P1_transpose_b"] = {"ok": False, "err": repr(e)[:300]}
        print(f"  P1 transpose_b: UNAVAILABLE {repr(e)[:120]}", flush=True)

    # P2: an L1-resident row-blocked in-projection -- the only route that deletes the
    # gp round trip. Priced for feasibility and speed, not built here.
    for rows in (32, 64):
        try:
            xr = x_norm[:, 0:rows]
            ms = timed(lambda: ttnn.experimental.minimal_matmul(
                xr, w_gp, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                compute_kernel_config=CK), f"P2_inproj_L1_rows{rows}",
                (rows * L * CZ * 2 + rows * L * n_wide * 2) / MB,
                2 * rows * L * CZ * n_wide)
            R["probe"][f"P2_L1_rows{rows}"] = {"ok": True, "ms": ms}
        except Exception as e:
            R["probe"][f"P2_L1_rows{rows}"] = {"ok": False, "err": repr(e)[:300]}
            print(f"  P2 rows={rows}: REFUSED {repr(e)[:120]}", flush=True)
    # and the same row block with a DRAM output, to price row blocking's own overhead
    try:
        xr = x_norm[:, 0:64]
        timed(lambda: ttnn.experimental.minimal_matmul(
            xr, w_gp, memory_config=MC, dtype=ttnn.bfloat16, compute_kernel_config=CK),
            "P2_inproj_DRAM_rows64", (64 * L * CZ * 2 + 64 * L * n_wide * 2) / MB,
            2 * 64 * L * CZ * n_wide)
    except Exception as e:
        print(f"  P2 DRAM rows64 failed: {repr(e)[:120]}", flush=True)
    # does the E6 reader accept a non-square (row-blocked) fused projection?
    try:
        gp_rows = ttnn.experimental.minimal_matmul(x_norm[:, 0:64], w_gp, memory_config=MC,
                                                   dtype=ttnn.bfloat16, compute_kernel_config=CK)
        R["probe"]["P2_e6_accepts_rowblock"] = bool(RP.eligible_gated(gp_rows, sc, MC))
        ttnn.deallocate(gp_rows)
    except Exception as e:
        R["probe"]["P2_e6_accepts_rowblock"] = f"err {repr(e)[:200]}"
    print(f"  P2 E6 accepts a row-blocked projection: "
          f"{R['probe']['P2_e6_accepts_rowblock']}", flush=True)

    # P3: is there a single op for the gated residual  z + p*sigmoid(g)?
    have = {n: hasattr(ttnn, n) for n in ("addcmul", "addcdiv", "mac", "addalpha")}
    R["probe"]["P3_ops_present"] = have
    if have.get("addcmul"):
        try:
            ms = timed(lambda: ttnn.addcmul(z0, p_out, g_out, 1.0), "P3_addcmul", 4 * pair_mb)
            R["probe"]["P3_addcmul"] = {"ok": True, "ms": ms}
        except Exception as e:
            R["probe"]["P3_addcmul"] = {"ok": False, "err": repr(e)[:300]}
    print(f"  P3 fused-residual candidates present: {have}", flush=True)

    # P4: the two out projections back to back vs one N=512 matmul (they take different
    # inputs, so this only prices what a fusion WOULD be worth if the inputs were shared).
    w512 = ttnn.from_torch(torch.cat([ttnn.to_torch(tm.out_p_weight),
                                      ttnn.to_torch(tm.g_out_weight)], dim=-1),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    timed(lambda: ttnn.experimental.minimal_matmul(x_ln, w512, memory_config=MC,
                                                   dtype=ttnn.bfloat16, compute_kernel_config=CK),
          "P4_out_proj_fused_N512", 3 * pair_mb, 2 * proj_fl)

    # ---------------- the reconstruction check ----------------------------------------------
    o = R["ops"]
    parts = {"1_layer_norm_in": 1, "2_inproj_minimal_mm_N%d" % n_wide: 1, "3_gated_move": 2,
             "4_transpose_inner": 1, "5_channel_matmul": 1, "6_channel_move_back": 1,
             "7_layer_norm_out": 1, "8_out_proj": 2, "9_gate_multiply": 1}
    tot = sum(o[k]["ms"] * m for k, m in parts.items())
    whole = o["WHOLE_trimul_e6_on"]["ms"]
    R["reconstruction"] = {"sum_of_ops_ms": round(tot, 4), "whole_op_ms": round(whole, 4),
                           "ratio": round(tot / whole, 4),
                           "share": {k: round(o[k]["ms"] * m / tot, 4) for k, m in parts.items()}}
    print(f"\n  sum of ops {tot:.3f} ms vs whole-op {whole:.3f} ms  "
          f"ratio {tot / whole:.4f}", flush=True)
    for k, m in sorted(parts.items(), key=lambda kv: -o[kv[0]]["ms"] * kv[1]):
        print(f"    {k:28s} x{m}  {o[k]['ms'] * m:7.3f} ms  "
              f"{100 * o[k]['ms'] * m / tot:5.1f} %", flush=True)

    a.out.write_text(json.dumps(R, indent=1))
    print("\nwrote " + str(a.out), flush=True)


if __name__ == "__main__":
    main()
