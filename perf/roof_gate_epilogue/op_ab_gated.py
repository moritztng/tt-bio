#!/usr/bin/env python3
"""Price the gate epilogue at the OP level: what it deletes against what it costs.

The block A/B reads a small loss, which can mean two different things, and the block cannot tell
them apart: either the `multiply_` it deletes was already free there, or the epilogue costs the
SDPA more than the multiply cost. These four arms separate them.

    mul        the `multiply_(o, g, SIGMOID)` program the fold deletes, on its own
    sdpa       the fused SDPA as it ships, on its own
    pair       sdpa + mul back to back -- what the fold runs today
    gated      sdpa with `gate=g` -- what the fold runs with the lever on

`pair / gated` is the lever. `gated - sdpa` is what the epilogue costs inside the kernel, and
`mul` is what it buys. Arms interleave per rep and `sdpa` is run twice as its own A/A floor.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import triatt_sdpa as TS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--out", default="perf/roof_gate_epilogue/op_ab_gated_512_qb2_c2.json")
    a = ap.parse_args()
    S, H, D = a.n, a.heads, a.head_dim
    scale = D ** -0.5

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    torch.manual_seed(0)
    f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = f(torch.randn(S, H, S, D) * 0.5)
    k = f(torch.randn(S, H, S, D) * 0.5)
    v = f(torch.randn(S, H, S, D) * 0.5)
    b = f(torch.randn(1, H, S, S) * 2.0)
    g = f(torch.randn(S, H, S, D) * 2.0)

    k_chunk = T._tri_att_k_chunks(S, S)[-1]
    fits = list(T._tri_att_q_chunks(S, S))
    # the rung the fold takes, found once
    q_chunk = None
    for qc in fits:
        o = TS.sdpa(q, k, v, b, scale, qc, k_chunk)
        if o is not None:
            q_chunk = qc
            ttnn.deallocate(o)
            break
    assert q_chunk is not None, "the fused kernel declined every rung"

    def run_sdpa():
        return TS.sdpa(q, k, v, b, scale, q_chunk, k_chunk)

    def run_gated():
        return TS.sdpa(q, k, v, b, scale, q_chunk, k_chunk, gate=g)

    def timed(fn):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) * 1e3
        if out is not None:
            ttnn.deallocate(out)
        return dt

    o_hold = run_sdpa()          # a live o for the mul arm, re-made every call below

    def arm_mul():
        o = run_sdpa()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) * 1e3
        ttnn.deallocate(o)
        return dt

    def arm_pair():
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = run_sdpa()
        o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) * 1e3
        ttnn.deallocate(o)
        return dt

    ttnn.deallocate(o_hold)
    ARMS = {"mul": arm_mul, "sdpa": lambda: timed(run_sdpa), "sdpa2": lambda: timed(run_sdpa),
            "pair": arm_pair, "gated": lambda: timed(run_gated)}
    order = list(ARMS)
    t = {n: [] for n in order}
    for r in range(a.warm + a.reps):
        for n in order:
            dt = ARMS[n]()
            if r >= a.warm:
                t[n].append(dt)

    med = {n: st.median(t[n]) for n in order}
    spread = {n: (max(t[n]) - min(t[n])) / med[n] for n in order}
    res = {"n": S, "heads": H, "head_dim": D, "reps": a.reps, "warm": a.warm,
           "arch": str(dev.arch()), "grid": list(grid), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "loadavg": os.getloadavg(),
           "q_chunk": q_chunk, "k_chunk": k_chunk,
           "median_ms": med, "spread_frac": spread, "all_ms": t,
           "gate_stats": list(TS.GATE_STATS),
           "gate_rejects": {str(kk): vv for kk, vv in TS.GATE_REJECTS.items()},
           "derived": {
               "aa_floor_frac": abs(med["sdpa"] / med["sdpa2"] - 1.0),
               "lever_ratio_pair_over_gated": med["pair"] / med["gated"],
               "epilogue_cost_ms": med["gated"] - med["sdpa"],
               "mul_deleted_ms": med["mul"],
               "net_ms": med["mul"] - (med["gated"] - med["sdpa"]),
           }}
    print(json.dumps({"median_ms": med, "spread_frac": spread,
                      "derived": res["derived"], "gate_stats": res["gate_stats"],
                      "gate_rejects": res["gate_rejects"]}, indent=1), flush=True)
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
