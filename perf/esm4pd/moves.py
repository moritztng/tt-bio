#!/usr/bin/env python3
"""P-A: this card's demonstrated permute rate, and whether the back move has a cheaper spelling.

`trimul_ops2.py` measured `_channel_move_back` at 187.2 GB/s for 256 MiB of traffic while
`ttnn.clone` moved the same 256 MiB at 401.8 GB/s in the same session. Before that 2.15x can be
called a floor, it has to be shown to be the index move's own rate on THIS card and not a kernel
that lost a grid. §4's floor imports 185.5 and 268.6 GB/s from qb2's 11x10 grid; this box is 13x10.

Every arm is timed batched -- `n` back-to-back calls, one `synchronize_device` -- and every
alternative is checked with `torch.equal` against the shipped spelling, because "a permute is a
pure index move so it must be bit-exact" is an argument, not a measurement.

Predictions and kill gate: `perf/esm4pd/PREDICTIONS2.md`, committed before this ran.
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
    L, CZ = a.L, 256
    pair = L * L * CZ * 2 / MiB
    DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "L": L, "pair_MiB": round(pair, 1), "arms": {}}
    f = lambda t, **kw: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                        dtype=ttnn.bfloat16, **kw)
    print(f"grid {g.x}x{g.y}  pair {pair:.1f} MiB", flush=True)

    def batched(fn):
        got = []
        for _ in range(a.reps):
            outs = [fn() for _ in range(a.warm)]
            ttnn.synchronize_device(dev)
            for o in outs:
                ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            outs = [fn() for _ in range(a.n)]
            ttnn.synchronize_device(dev)
            got.append((time.perf_counter() - t0) * 1e3 / a.n)
            for o in outs:
                ttnn.deallocate(o)
        return st.median(got), [round(v, 4) for v in got]

    def rec(label, fn, mib, ref=None, equal=None):
        ms, allv = batched(fn)
        r = {"ms": round(ms, 4), "MiB": round(mib, 1),
             "GBps": round(mib * MiB / (ms * 1e-3) / GB, 1), "all": allv}
        if ref is not None:
            r["vs_shipped"] = round(ref / ms, 4)
        if equal is not None:
            r["torch_equal"] = bool(equal)
        R["arms"][label] = r
        print(f"  {label:44s} {ms:8.4f} ms  {r['GBps']:7.1f} GB/s"
              + (f"  {ref / ms:6.3f}x" if ref is not None else "")
              + (f"  equal={equal}" if equal is not None else ""), flush=True)
        return ms

    # ---- same-bytes controls: what 256 MiB and 384 MiB of pure traffic cost on this card ----
    src = f(torch.randn(1, L, L, CZ), memory_config=DRAM)
    rec("CTRL_clone_pair_256MiB", lambda: ttnn.clone(src, memory_config=DRAM), 2 * pair)
    src2 = f(torch.randn(1, L, L, CZ), memory_config=DRAM)
    rec("CTRL_add_pair_384MiB", lambda: ttnn.add(src, src2), 3 * pair)
    ttnn.deallocate(src2)

    # ---- the FORWARD move (0,3,1,2): plain, and the E6 gated one ----------------------------
    print("\n=== forward move (0,3,1,2), [1,L,L,C] -> [1,C,L,L] ===", flush=True)
    ok_fwd = RB.eligible(src, DRAM)
    R["eligible_fwd"] = bool(ok_fwd)
    base = rec("fwd_stock_ttnn.permute", lambda: ttnn.permute(src, (0, 3, 1, 2),
                                                              memory_config=DRAM), 2 * pair)
    if ok_fwd:
        ref_t = ttnn.to_torch(ttnn.permute(src, (0, 3, 1, 2), memory_config=DRAM))
        got = RB.reblock_permute(src, DRAM)
        eq = torch.equal(ttnn.to_torch(got), ref_t)
        ttnn.deallocate(got)
        del ref_t
        rec("fwd_REBLOCK_kernel", lambda: RB.reblock_permute(src, DRAM), 2 * pair, base, eq)
    else:
        print("  (reblock forward NOT eligible at this shape)", flush=True)

    # ---- the BACK move (0,2,3,1): the shipped kernel and every stock spelling ----------------
    print("\n=== back move (0,2,3,1), [1,C,L,L] -> [1,L,L,C] ===", flush=True)
    ch = f(torch.randn(1, CZ, L, L), memory_config=DRAM)
    R["eligible_back"] = bool(RB.eligible_back(ch, DRAM))
    shipped = rec("back_SHIPPED__channel_move_back",
                  lambda: T._channel_move_back(ch, DRAM), 2 * pair)
    ref_t = ttnn.to_torch(T._channel_move_back(ch, DRAM))

    def two_transpose():
        t1 = ttnn.transpose(ch, 1, 2, memory_config=DRAM)
        t2 = ttnn.transpose(t1, 2, 3, memory_config=DRAM)
        ttnn.deallocate(t1)
        return t2

    for label, fn, mib in (
        ("back_stock_ttnn.permute(0,2,3,1)",
         lambda: ttnn.permute(ch, (0, 2, 3, 1), memory_config=DRAM), 2 * pair),
        ("back_two_transpose_1,2_then_2,3", two_transpose, 4 * pair),
        ("back_reshape_transpose_-2-1_first",
         lambda: ttnn.permute(ttnn.transpose(ch, -2, -1, memory_config=DRAM), (0, 3, 1, 2),
                              memory_config=DRAM), 4 * pair),
    ):
        try:
            r = fn()
            eq = torch.equal(ttnn.to_torch(r), ref_t)
            ttnn.deallocate(r)
            rec(label, fn, mib, shipped, eq)
        except Exception as e:
            R["arms"][label] = {"ok": False, "err": repr(e)[:300]}
            print(f"  {label:44s} FAILED {repr(e)[:110]}", flush=True)

    # Does the back kernel's work split change with the grid? Price it on a forced-11x10 program
    # config the same way the fold would see it, to separate "this card's rate" from "this grid".
    R["split_plan"] = {}
    try:
        plan = RB._split_plan(dev, CZ)
        R["split_plan"]["units_C%d" % CZ] = str(plan)
        print(f"\n  _split_plan(dev, {CZ}) = {plan}", flush=True)
    except Exception as e:
        R["split_plan"]["err"] = repr(e)[:200]
        print(f"\n  _split_plan unavailable: {repr(e)[:120]}", flush=True)

    del ref_t
    ttnn.deallocate(ch)
    ttnn.deallocate(src)

    # ---- the verdict against the kill gate ---------------------------------------------------
    alts = {k: v for k, v in R["arms"].items()
            if k.startswith("back_") and v.get("torch_equal") and "vs_shipped" in v}
    best = max(alts.items(), key=lambda kv: kv[1]["vs_shipped"], default=None)
    R["verdict"] = {
        "shipped_back_ms": round(shipped, 4),
        "best_bitexact_alt": best[0] if best else None,
        "best_speedup": best[1]["vs_shipped"] if best else None,
        "kill_gate": 1.20,
        "decision": ("GO" if best and best[1]["vs_shipped"] >= 1.20 else "NO-GO")}
    print(f"\n  VERDICT: {R['verdict']}", flush=True)
    a.out.write_text(json.dumps(R, indent=1))
    print("wrote " + str(a.out), flush=True)


if __name__ == "__main__":
    main()
