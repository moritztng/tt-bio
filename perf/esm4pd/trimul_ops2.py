#!/usr/bin/env python3
"""The per-op TriMul table, re-measured BATCHED on current code, plus the floor it implies.

`trimul_reprice.py` taped the shipped `__call__` and established the op list, the call counts and
the shapes with no hand transcription. It also showed why a synced tape cannot price the floor: the
same two `ttnn.linear` calls read 28.9 ms in one arm and 2.2 ms in the other, because a tape syncs
every op and a sync between two adjacent ops is a cost the fold never pays
(`tt-bio-isolated-op-timing-oversync-inflates-cost`).

So this script re-runs exactly the op list the tape found -- same ops, same shapes, same configs,
same order -- but times each one BATCHED: `n` back-to-back calls with a single
`synchronize_device` at the end. Then it checks the sum against the whole-op batched wall and
reports the ratio, which is the transcription check. Roofs are measured in the same session.

Nothing here is a fold result. It re-prices §4's floor table, which is what §6 step 1 asks for.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RB

MiB = 2 ** 20
GB = 10 ** 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    CK = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    L, CZ, LATENT = a.L, 256, 256
    pair = L * L * CZ * 2 / MiB
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "L": L, "c_z": CZ, "pair_MiB": round(pair, 1),
         "n": a.n, "warm": a.warm, "reps": a.reps, "ops": {}}
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    print(f"grid {g.x}x{g.y}  pair {pair:.1f} MiB", flush=True)

    def batched(fn):
        """Median over `reps` of (n back-to-back calls, ONE sync) / n. No per-op sync."""
        got = []
        for _ in range(a.reps):
            outs = [fn() for _ in range(a.warm)]
            ttnn.synchronize_device(dev)
            for o in outs:
                if isinstance(o, ttnn.Tensor):
                    ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            outs = [fn() for _ in range(a.n)]
            ttnn.synchronize_device(dev)
            got.append((time.perf_counter() - t0) * 1e3 / a.n)
            for o in outs:
                if isinstance(o, ttnn.Tensor):
                    ttnn.deallocate(o)
        return st.median(got), [round(v, 4) for v in got]

    # ---------------- roofs, measured on this card in this session ---------------------------
    z0, z1 = f(torch.randn(1, L, L, CZ)), f(torch.randn(1, L, L, CZ))
    roofs = {}
    for label, fn, mib in (("add_2r1w", lambda: ttnn.add(z0, z1), 3 * pair),
                           ("clone_1r1w",
                            lambda: ttnn.clone(z0, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                            2 * pair),
                           ("mul_2r1w", lambda: ttnn.multiply(z0, z1), 3 * pair)):
        ms, allv = batched(fn)
        roofs[label] = {"ms": round(ms, 4), "MiB": round(mib, 1),
                        "GBps": round(mib * MiB / (ms * 1e-3) / GB, 1), "all": allv}
        print(f"  roof {label:12s} {ms:7.3f} ms  {roofs[label]['GBps']:7.1f} GB/s", flush=True)
    ttnn.deallocate(z0)
    ttnn.deallocate(z1)
    ROOF = max(v["GBps"] for v in roofs.values())
    R["roofs"], R["dram_roof_GBps"] = roofs, ROOF
    print(f"  MEASURED DRAM roof: {ROOF} GB/s", flush=True)

    # ---------------- the real object at the real shapes -------------------------------------
    from tt_bio.tenstorrent import WeightScope
    torch.manual_seed(0)
    tsd = WeightScope({
        "norm_in.weight": torch.randn(CZ), "norm_in.bias": torch.randn(CZ),
        "norm_out.weight": torch.randn(CZ), "norm_out.bias": torch.randn(CZ),
        "g_in.weight": torch.randn(2 * LATENT, CZ) * 0.02,
        "p_in.weight": torch.randn(2 * LATENT, CZ) * 0.02,
        "g_out.weight": torch.randn(CZ, LATENT) * 0.02,
        "p_out.weight": torch.randn(CZ, LATENT) * 0.02})
    tm_s = T.TriangleMultiplication(False, tsd, CK, gated_move=True)
    tm_e = T.TriangleMultiplication(True, tsd, CK, gated_move=True)
    x = f(torch.randn(1, L, L, CZ) * 0.5)
    for t in (tm_s, tm_e):
        t.prewarm(L, 1)

    chunk_size = T._trimul_chunk_size(L, tm_e._hidden, 1)
    n_pairs = tm_e._hidden // chunk_size
    MC = T._triangle_mul_memory_config(L)
    group = T._trimul_inproj_group(L, chunk_size, 1, n_pairs)
    R["shape"] = {"chunk": chunk_size, "n_pairs": n_pairs, "group": group,
                  "groups": n_pairs // group, "dram": MC.buffer_type == ttnn.BufferType.DRAM}
    print(f"  shape {R['shape']}", flush=True)

    # ---------------- the whole-op wall, batched ---------------------------------------------
    walls = {}
    for name, tm in (("start", tm_s), ("end", tm_e)):
        ms, allv = batched(lambda: tm(x, None))
        walls[name] = {"ms": round(ms, 4), "all": allv}
        print(f"  WHOLE tri_mul_{name:5s} {ms:8.3f} ms   {allv}", flush=True)
    R["whole_op_wall"] = walls
    WALL = (walls["start"]["ms"] + walls["end"]["ms"]) / 2

    # ---------------- op by op, batched, at the taped shapes ---------------------------------
    # The op list, counts and shapes below are the ones `trimul_reprice_c0.json` taped out of the
    # shipped `__call__`; nothing here is guessed.
    def rec(label, fn, mult, mib, flop=None, keep=False):
        ms, allv = batched(fn)
        r = {"ms": round(ms, 4), "x": mult, "MiB_per_call": round(mib, 1),
             "GBps": round(mib * MiB / (ms * 1e-3) / GB, 1),
             "floor_ms_bytes": round(mib * MiB / (ROOF * GB) * 1e3, 4), "all": allv}
        if flop:
            r["GFLOP"] = round(flop / 1e9, 1)
            r["TFLOPps"] = round(flop / (ms * 1e-3) / 1e12, 1)
        R["ops"][label] = r
        print(f"  {label:34s} x{mult} {ms:8.3f} ms  {r['GBps']:7.1f} GB/s  "
              f"floor {r['floor_ms_bytes']:6.3f}"
              + (f"  {r['TFLOPps']:6.1f} TF/s" if flop else ""), flush=True)
        return ms

    ln = lambda t, w, b: ttnn.layer_norm(t, weight=w, bias=b, epsilon=1e-5,
                                         compute_kernel_config=CK)
    tm = tm_e
    rec("1_layer_norm_in", lambda: ln(x, tm.in_norm_weight, tm.in_norm_bias), 2, 2 * pair)
    x_norm = ln(x, tm.in_norm_weight, tm.in_norm_bias)

    w_gp = tm._gp_in_chunks(chunk_size, group)[0]
    nw = int(w_gp.shape[-1])
    inproj_mib = pair + pair * nw / CZ
    rec(f"2_inproj_minimal_mm_N{nw}",
        lambda: ttnn.experimental.minimal_matmul(x_norm, w_gp, memory_config=MC,
                                                 dtype=ttnn.bfloat16, compute_kernel_config=CK),
        1, inproj_mib, 2 * L * L * CZ * nw)
    gp = ttnn.experimental.minimal_matmul(x_norm, w_gp, memory_config=MC, dtype=ttnn.bfloat16,
                                          compute_kernel_config=CK)
    sc = int(gp.shape[-1]) // 4
    assert RB.eligible_gated(gp, sc, MC), "E6 declines here -- the replay is not the shipped path"
    rec("3_E6_gated_move",
        lambda: RB.reblock_permute_gated(gp, 2 * sc, 0, sc, memory_config=MC), 2, 3 * pair)
    a_ch = RB.reblock_permute_gated(gp, 2 * sc, 0, sc, memory_config=MC)
    b_ch = RB.reblock_permute_gated(gp, 3 * sc, sc, sc, memory_config=MC)
    rec("4_transpose_inner", lambda: ttnn.transpose(a_ch, -2, -1, memory_config=MC), 1, 2 * pair)
    a_t = ttnn.transpose(a_ch, -2, -1, memory_config=MC)

    pc = T._triangle_mul_program_config((L + 31) // 32)
    rec("5_channel_matmul",
        lambda: ttnn.matmul(a_t, b_ch, compute_kernel_config=CK, memory_config=MC,
                            program_config=pc, dtype=ttnn.bfloat16),
        1, 3 * pair, 2 * CZ * L * L * L)
    x_ch = ttnn.matmul(a_t, b_ch, compute_kernel_config=CK, memory_config=MC,
                       program_config=pc, dtype=ttnn.bfloat16)
    rec("6_channel_move_back", lambda: T._channel_move_back(x_ch, MC), 1, 2 * pair)
    x_mix = T._channel_move_back(x_ch, MC)
    rec("7_layer_norm_out", lambda: ln(x_mix, tm.out_norm_weight, tm.out_norm_bias), 2 - 1,
        2 * pair)
    x_ln = ln(x_mix, tm.out_norm_weight, tm.out_norm_bias)
    rec("8_out_proj", lambda: T._trimul_out_proj(x_ln, tm.out_p_weight, CK), 2, 2 * pair,
        2 * L * L * CZ * CZ)
    p_out = T._trimul_out_proj(x_ln, tm.out_p_weight, CK)
    g_out = T._trimul_out_proj(x_norm, tm.g_out_weight, CK)
    # the shipped call is multiply_ (in place into p_out); timed non-inplace so the batch does not
    # mutate its own operand. Same traffic: 2 reads + 1 write.
    rec("9_gate_multiply", lambda: ttnn.multiply(
        p_out, g_out, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]), 1, 3 * pair)

    # ---------------- reconstruction + floor --------------------------------------------------
    # `1_layer_norm_in` is x2 in the tape (norm_in and norm_out are the same op at the same shape);
    # `7_layer_norm_out` is therefore counted once here to avoid double counting.
    tot = sum(v["ms"] * v["x"] for v in R["ops"].values())
    floor = sum(v["floor_ms_bytes"] * v["x"] for v in R["ops"].values())
    R["reconstruction"] = {
        "sum_of_batched_ops_ms": round(tot, 4),
        "whole_op_wall_ms_mean": round(WALL, 4),
        "ratio": round(tot / WALL, 4),
        "floor_ms_all_ops_at_dram_roof": round(floor, 4),
        "headroom_ms_per_call": round(WALL - floor, 4)}
    print(f"\n  sum of batched ops {tot:.3f} ms vs whole-op wall {WALL:.3f} ms "
          f"ratio {tot / WALL:.4f}", flush=True)
    print(f"  byte floor at {ROOF} GB/s: {floor:.3f} ms  ->  headroom "
          f"{WALL - floor:.3f} ms/call", flush=True)
    for k, v in sorted(R["ops"].items(), key=lambda kv: -kv[1]["ms"] * kv[1]["x"]):
        print(f"    {k:34s} x{v['x']} {v['ms'] * v['x']:7.3f} ms  floor "
              f"{v['floor_ms_bytes'] * v['x']:6.3f}  headroom "
              f"{(v['ms'] - v['floor_ms_bytes']) * v['x']:6.3f}", flush=True)

    a.out.write_text(json.dumps(R, indent=1))
    print("\nwrote " + str(a.out), flush=True)


if __name__ == "__main__":
    main()
