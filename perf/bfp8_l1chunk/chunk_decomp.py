#!/usr/bin/env python3
"""Decompose bfp8-at-its-best-chunk into the dtype step and the chunk step.

chunk_ab.py measured bf16 (q512,k256) against bfp8 (q512,k512) at 1.2577x, which disagrees in
sign with bfp8-sdpa-unlock's 0.8071x for the operand narrowing alone at a fixed chunk. Both can be
true only if the chunk step is large, so this measures all three points in ONE session:

    A  bf16 (512, 256)   bf16's best legal config, the reference
    B  bfp8 (512, 256)   the dtype step alone, the sibling's comparison
    C  bfp8 (512, 512)   the dtype step plus the chunk bf16 cannot fit
    N  bf16 (512, 512)   NEGATIVE CONTROL -- must be refused by L1, or the premise is wrong
    A' bf16 (512, 256)   A/A twin

Arms interleaved rep by rep with the interior order reversed on odd reps.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, socket, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_cspec = importlib.util.spec_from_file_location(
    "_clk", REPO / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock" / "clk.py")
CLK = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(CLK)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    import tt_bio.triatt_sdpa as TS
    from tt_bio.tenstorrent import get_device
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), _TB.__file__

    dev = get_device()
    S, H, D = a.seq, a.heads, a.head_dim
    scale = D ** -0.5

    def mk(dtype):
        g = torch.Generator().manual_seed(0)
        t = [ttnn.from_torch(torch.randn(S, H, S, D, generator=g), dtype=dtype,
                             layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
        t.append(ttnn.from_torch(torch.randn(1, H, S, S, generator=g), dtype=dtype,
                                 layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return t

    T = {"bf16": mk(ttnn.bfloat16), "bfp8": mk(ttnn.bfloat8_b)}
    ARMS = {"A_bf16_k256": ("bf16", 512, 256), "B_bfp8_k256": ("bfp8", 512, 256),
            "C_bfp8_k512": ("bfp8", 512, 512), "Aa_bf16_k256": ("bf16", 512, 256)}
    NEG = ("bf16", 512, 512)

    def once(spec):
        q, k, v, b = T[spec[0]]
        return TS.sdpa(q, k, v, b, scale, spec[1], spec[2])

    # negative control first, before any memo could mask it: bf16 at k512 must be refused
    TS._PM_OVER_L1.clear(); TS.REJECTS.clear()
    neg_out = once(NEG)
    neg = {"served": neg_out is not None,
           "rejects": {str(kk): vv for kk, vv in TS.REJECTS.items()},
           "l1_error": {str(kk): vv for kk, vv in TS.PM_L1_ERRORS.items()}}
    if neg_out is not None:
        ttnn.deallocate(neg_out)
    print("NEGATIVE CONTROL bf16 (512,512) served =", neg["served"], neg["rejects"])

    def run(name):
        spec = ARMS[name]
        served = TS.STATS[0]
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.iters):
            o = once(spec)
            assert o is not None, f"{name} declined -- a fallback is a failure, not a cost"
            ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / a.iters, TS.STATS[0] - served

    for name in ARMS:
        run(name)                                   # warm: each arm compiles its own program

    nodes = CLK.nodes_open_by_this_process()
    held = CLK.force(a.mhz, nodes)
    sampler = CLK.Sampler(nodes[0])
    draws = {k: [] for k in ARMS}
    serves = {k: 0 for k in ARMS}
    # ABBA over the whole arm set, so every arm gets two draws symmetric about the rep midpoint
    # and a linear drift in the box cancels per arm rather than landing on whichever arm ran last.
    # The A/A twin sits ADJACENT to its own arm in both halves: the previous design put it at the
    # opposite end of the rep and read a 5.288 % floor on a session whose real floor was 0.1 %.
    names = ["A_bf16_k256", "Aa_bf16_k256", "B_bfp8_k256", "C_bfp8_k512"]
    per_rep = {k: [] for k in ARMS}
    for rep in range(a.reps):
        half = {}
        for name in names + names[::-1]:
            dt, n = run(name)
            half.setdefault(name, []).append(dt)
            serves[name] += n
        for name in names:
            per_rep[name].append(sum(half[name]) / len(half[name]))
            draws[name].extend(half[name])
    clk_stats = sampler.stop()
    CLK.release()

    # paired per rep, then the median of the per-rep ratios -- not a ratio of pooled medians
    med = {k: st.median(v) for k, v in per_rep.items()}
    base = med["A_bf16_k256"]
    paired = {k: st.median([r / x for r, x in zip(per_rep["A_bf16_k256"], per_rep[k])])
              for k in ARMS}
    res = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": list(TT.COMPUTE_GRID_MAIN), "shape": {"S": S, "H": H, "D": D},
        "reps": a.reps, "iters": a.iters,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "aiclk_requested_mhz": a.mhz, "aiclk_nodes_held": held,
        "aiclk_sampled_during_run": clk_stats,
        "negative_control_bf16_k512": neg,
        "median_ms": {k: round(v * 1e3, 4) for k, v in med.items()},
        "aa_floor_pct": round(100 * abs(paired["Aa_bf16_k256"] - 1.0), 4),
        "speedup_vs_A_paired": {k: round(v, 4) for k, v in paired.items()},
        "dtype_step_A_over_B": round(paired["B_bfp8_k256"], 4),
        "chunk_step_B_over_C": round(paired["C_bfp8_k512"] / paired["B_bfp8_k256"], 4),
        "per_rep_ms": {k: [round(x * 1e3, 4) for x in v] for k, v in per_rep.items()},
        "served_per_arm": serves,
        "draws_ms": {k: [round(x * 1e3, 4) for x in v] for k, v in draws.items()},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "draws_ms"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
