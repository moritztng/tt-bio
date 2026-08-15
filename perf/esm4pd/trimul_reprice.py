#!/usr/bin/env python3
"""Re-price ESMFold2's TriangleMultiplication op by op, on current code.

The floor in `state/esmfold2-to-4x-per-dollar.md` §4 rests entirely on a per-op table measured on
qb2 card 2 at ttnn 0.68.0 BEFORE two levers landed (the gated-move work-split re-key, and the
L1-resident fc1). §6 step 1 of that document asks for exactly one thing: re-measure that table on
current code, on this card, and redo the floor.

Two properties this harness has that the predecessor's `perf/esm4x/trimul_ops.py` did not:

1. **It tapes the real `__call__`**, wrapping the ops the shipped code actually issues, instead of
   re-spelling the op sequence by hand. A hand transcription drifts every time the model code
   changes, and the model code has changed twice since the table was written.
2. **The wall it scales against is measured batched** -- `n` back-to-back calls with ONE
   `synchronize_device` at the end -- while the tape syncs every op. The ratio between the two IS
   the oversync inflation (`tt-bio-isolated-op-timing-oversync-inflates-cost`), reported rather
   than assumed away, and the per-op costs are published both raw and scaled by it.

The DRAM roof is measured in this same session on this card, never carried in
(`roofline-roof-must-be-measured-not-asserted`).
"""
import argparse, collections, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RB

MB = 2 ** 20
GB = 10 ** 9


def shp(t):
    try:
        return "x".join(str(int(d)) for d in t.shape)
    except Exception:
        return "?"


def buf(kw):
    mc = kw.get("memory_config")
    if mc is None:
        return "-"
    return "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3, help="repeats of the whole wall+tape cycle")
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
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else "?",
         "grid": [g.x, g.y], "L": L, "c_z": CZ, "n": a.n, "warm": a.warm, "reps": a.reps}
    pair_mb = L * L * CZ * 2 / MB
    R["pair_mb"] = round(pair_mb, 2)
    print(f"grid {g.x}x{g.y}  pair tensor {pair_mb:.1f} MB", flush=True)

    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    # ---------------- the DRAM roof, measured here, in this session --------------------------
    def batched(fn, n, warm):
        """ms per call of fn, n calls back to back with ONE sync. The honest per-call cost."""
        outs = [fn() for _ in range(warm)]
        ttnn.synchronize_device(dev)
        for o in outs:
            if isinstance(o, ttnn.Tensor):
                ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(n)]
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3 / n
        for o in outs:
            if isinstance(o, ttnn.Tensor):
                ttnn.deallocate(o)
        return ms

    z0, z1 = f(torch.randn(1, L, L, CZ)), f(torch.randn(1, L, L, CZ))
    roofs = {}
    for label, fn, mbs in (
        ("add_2r1w", lambda: ttnn.add(z0, z1), 3 * pair_mb),
        ("clone_1r1w", lambda: ttnn.clone(z0, memory_config=ttnn.DRAM_MEMORY_CONFIG),
         2 * pair_mb),
        ("mul_2r1w", lambda: ttnn.multiply(z0, z1), 3 * pair_mb),
    ):
        ms = st.median([batched(fn, a.n, a.warm) for _ in range(a.reps)])
        gbs = mbs * MB / (ms * 1e-3) / GB
        roofs[label] = {"ms": round(ms, 4), "MB": round(mbs, 1), "GBps": round(gbs, 1)}
        print(f"  roof {label:12s} {ms:8.3f} ms  {mbs:7.1f} MB  {gbs:7.1f} GB/s", flush=True)
    ttnn.deallocate(z0)
    ttnn.deallocate(z1)
    R["roofs"] = roofs
    DRAM_ROOF = max(v["GBps"] for v in roofs.values())
    R["dram_roof_GBps"] = DRAM_ROOF
    print(f"  MEASURED DRAM roof this session: {DRAM_ROOF} GB/s", flush=True)

    # ---------------- the real object, at the real shapes ------------------------------------
    from tt_bio.tenstorrent import WeightScope
    torch.manual_seed(0)
    tsd = WeightScope({
        "norm_in.weight": torch.randn(CZ), "norm_in.bias": torch.randn(CZ),
        "norm_out.weight": torch.randn(CZ), "norm_out.bias": torch.randn(CZ),
        "g_in.weight": torch.randn(2 * LATENT, CZ) * 0.02,
        "p_in.weight": torch.randn(2 * LATENT, CZ) * 0.02,
        "g_out.weight": torch.randn(CZ, LATENT) * 0.02,
        "p_out.weight": torch.randn(CZ, LATENT) * 0.02})
    tms = {"start": T.TriangleMultiplication(False, tsd, CK, gated_move=True),
           "end": T.TriangleMultiplication(True, tsd, CK, gated_move=True)}
    x = f(torch.randn(1, L, L, CZ) * 0.5)

    tm = tms["start"]
    chunk_size = T._trimul_chunk_size(L, tm._hidden, 1)
    n_pairs = tm._hidden // chunk_size
    mc = T._triangle_mul_memory_config(L)
    large = mc.buffer_type == ttnn.BufferType.DRAM
    group = T._trimul_inproj_group(L, chunk_size, 1, n_pairs) if large else 1
    R["shape"] = {"hidden": tm._hidden, "chunk_size": chunk_size, "n_pairs": n_pairs,
                  "group": group, "groups": n_pairs // group, "dram": bool(large),
                  "host_concat": bool(large and T._host_concat(x)),
                  "SEQ_LEN_MORE_CHUNKING": T.SEQ_LEN_MORE_CHUNKING,
                  "row_norm": bool(L * L * CZ * 2 > T.TRIMUL_IN_NORM_ROWBLOCK_BYTES)}
    print(f"  shape gate: {R['shape']}", flush=True)
    for t in tms.values():
        t.prewarm(L, 1)

    # ---------------- the tape: every op the shipped path issues -----------------------------
    ROWS, ON = [], [False]

    def wrap(mod, name, tagger):
        orig = getattr(mod, name)

        def w(*args, **kw):
            if not ON[0]:
                return orig(*args, **kw)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = orig(*args, **kw)
            ttnn.synchronize_device(dev)
            ROWS.append((tagger(args, kw), (time.perf_counter() - t0) * 1e3))
            return out
        setattr(mod, name, w)
        return orig

    originals = []
    for mod, name, tagger in (
        (ttnn, "matmul", lambda a_, k: f"matmul[{shp(a_[0])}@{shp(a_[1])}]->{buf(k)}"),
        (ttnn, "linear", lambda a_, k: f"linear[{shp(a_[0])}@{shp(a_[1])}]->{buf(k)}"),
        (ttnn.experimental, "minimal_matmul",
         lambda a_, k: f"minimal_matmul[{shp(a_[0])}@{shp(a_[1])}]->{buf(k)}"),
        (ttnn, "permute", lambda a_, k: f"permute{tuple(a_[1])}[{shp(a_[0])}]->{buf(k)}"),
        (ttnn, "transpose", lambda a_, k: f"transpose({a_[1]},{a_[2]})[{shp(a_[0])}]->{buf(k)}"),
        (ttnn, "layer_norm", lambda a_, k: f"layer_norm[{shp(a_[0])}]"),
        (ttnn, "multiply", lambda a_, k: f"multiply[{shp(a_[0])}]"),
        (ttnn, "multiply_", lambda a_, k: f"multiply_[{shp(a_[0])}]"),
        (ttnn, "chunk", lambda a_, k: f"chunk[{shp(a_[0])}]"),
        (ttnn, "concat", lambda a_, k: f"concat[{len(a_[0])}x{shp(a_[0][0])}]"),
        (ttnn, "clone", lambda a_, k: f"clone[{shp(a_[0])}]->{buf(k)}"),
        (ttnn, "reallocate", lambda a_, k: f"reallocate[{shp(a_[0])}]"),
        (ttnn, "add", lambda a_, k: f"add[{shp(a_[0])}]"),
        # the two hand-written kernels: generic_op, so no ttnn entry point sees them
        (RB, "reblock_permute_gated",
         lambda a_, k: f"E6_gated_move[{shp(a_[0])}]->{buf(k)}"),
        (RB, "reblock_permute", lambda a_, k: f"REBLOCK_fwd[{shp(a_[0])}]->{buf(k)}"),
        (RB, "reblock_permute_back", lambda a_, k: f"REBLOCK_back[{shp(a_[0])}]->{buf(k)}"),
        (T, "_channel_move_back", lambda a_, k: f"_channel_move_back[{shp(a_[0])}]"),
    ):
        originals.append((mod, name, wrap(mod, name, tagger)))

    def wall(which):
        """Batched per-call wall of the REAL op: n calls, one sync. No tape."""
        return st.median([batched(lambda: tms[which](x, None), a.n, a.warm)
                          for _ in range(a.reps)])

    def tape(which):
        for arr in (RB.STATS, RB.STATS_BACK, RB.STATS_GATED):
            arr[0] = arr[1] = 0
        ON[0] = True
        del ROWS[:]
        out = tms[which](x, None)
        ON[0] = False
        ttnn.deallocate(out)
        agg = collections.OrderedDict()
        for tag, ms in ROWS:
            r = agg.setdefault(tag, {"tag": tag, "n": 0, "ms": 0.0})
            r["n"] += 1
            r["ms"] += ms
        return agg, {"gated": list(RB.STATS_GATED), "fwd": list(RB.STATS),
                     "back": list(RB.STATS_BACK)}

    results = {}
    for which in ("start", "end"):
        w = wall(which)
        agg, stats = tape(which)
        # `_channel_move_back` brackets REBLOCK_back: count the outer, keep the inner for info.
        inner = {k: v for k, v in agg.items() if k.startswith("REBLOCK_back")}
        outer = {k: v for k, v in agg.items() if not k.startswith("REBLOCK_back")}
        taped = sum(v["ms"] for v in outer.values())
        scale = w / taped
        rows = sorted(outer.values(), key=lambda r: -r["ms"])
        for r in rows:
            r["ms"] = round(r["ms"], 4)
            r["ms_scaled"] = round(r["ms"] * scale, 4)
            r["share"] = round(r["ms"] / taped, 4)
        results[which] = {
            "wall_ms_batched": round(w, 4), "taped_sum_ms": round(taped, 4),
            "oversync_ratio": round(taped / w, 4), "scale": round(scale, 4),
            "reblock_stats_served_declined": stats,
            "reblock_back_inner_ms": {k: round(v["ms"], 4) for k, v in inner.items()},
            "ops": rows}
        print(f"\n=== tri_mul_{which}: batched wall {w:.3f} ms | taped sum {taped:.3f} ms "
              f"| oversync {taped / w:.4f}x | reblock {stats} ===",
              flush=True)
        for r in rows:
            print(f"  {r['tag']:52s} x{r['n']:<3d} {r['ms']:8.3f} ms "
                  f"-> {r['ms_scaled']:8.3f} scaled  {100 * r['share']:5.1f} %", flush=True)
        if RB.REJECTS:
            print("  reblock refusals:", dict(RB.REJECTS), flush=True)

    for mod, name, orig in originals:
        setattr(mod, name, orig)
    R["trimul"] = results
    R["mean_wall_ms"] = round(
        (results["start"]["wall_ms_batched"] + results["end"]["wall_ms_batched"]) / 2, 4)
    a.out.write_text(json.dumps(R, indent=1))
    print("\nwrote " + str(a.out), flush=True)


if __name__ == "__main__":
    main()
