"""ROOF phase A, link 1 (EPILOGUE): the trimul tail's `g_out` -> `multiply_` DRAM round trip.

`perf/roof_orchestrator/FUSION_PAIRS.md` rank 5, 268.4 MB of one Pairformer block. The trace
`perf/b2z2_byte_floor/out/trace_512_wh_c10.json.gz` names the buffer: at 512 aa the tail runs

    p_out = _pair_proj_linear(x_norm_out, Wp, l1_out=True)      -> 67.11 MB in L1, SERVED
    g_out = _pair_proj_linear(x_norm_in,  Wg, l1_out=True)      -> REFUSED, 67.11 MB in DRAM
    multiply_(p_out, g_out, SIGMOID on b)                       -> in place into p_out's L1

Only `g_out` round-trips, because `p_out` already holds the one L1 slot a 512 aa pair tensor fits
in. So the epilogue cannot be bought by moving a destination; it needs one kernel over both
projections, which is `tt_bio/trimul_tail.py` (F1), with its product left in L1.

Arms, interleaved in one process on one device open:
    A   shipped default: the three ops above
    B   F1 at the (4,4) key with an L1 product          <- the lever
    C   F1 at the (4,4) key with a DRAM product          (what F1 ships as; isolates the
                                                          destination from the kernel)
    A2  A again, the A/A floor
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as F1


def sync(dev):
    ttnn.synchronize_device(dev)


def timed(dev, fn, n):
    fn()
    sync(dev)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        sync(dev)
        out.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return out


def arm_a(x, xin, wp, wg, ckc):
    p = T._trimul_out_proj(x, wp, ckc)
    g = T._trimul_out_proj(xin, wg, ckc)
    return ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])


def arm_f1(x, xin, wp, wg, ckc, grid, l1):
    from tt_bio import mm_generic as MG
    return F1.fused_tail(x, xin, wp, wg, MG.ckc_args(ckc), grid,
                         out_memory_config=ttnn.L1_MEMORY_CONFIG if l1 else None)


def bytes_of(fn):
    """DRAM bytes one arm moves, deduped on buffer ADDRESS. `real_traffic.counts` is the
    census's own counter, reused so this table cannot drift from the one it extends."""
    from real_traffic import counts

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    r = fn()
    ttnn.synchronize_device(r.device())
    ttnn.deallocate(r)
    c = counts({"nodes": ttnn.graph.end_graph_capture()})
    return {"real_MB": round(c["real_MB"], 3), "w_MB": round(c["real_w_MB"], 3),
            "r_MB": round(c["real_r_MB"], 3), "n_ops": c["n_ops"],
            "per_op": [[round(t / 1e6, 3), n] for t, n, _, _ in c["per_op"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--cz", type=int, default=128)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="perf/roof_trimul_out/op_ab.json")
    a = ap.parse_args()

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    grid = tuple(T.COMPUTE_GRID_MAIN)
    ckc = T.trunk_compute_kernel_config(compute_kernel_config())
    N, C = a.n, a.cz
    torch.manual_seed(0)
    hx = torch.randn(1, N, N, C, dtype=torch.float32) * 0.5
    hxin = torch.randn(1, N, N, C, dtype=torch.float32) * 0.5
    hwp = torch.randn(C, C, dtype=torch.float32) * (C ** -0.5)
    hwg = torch.randn(C, C, dtype=torch.float32) * (C ** -0.5)
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x, xin, wp, wg = tt(hx), tt(hxin), tt(hwp), tt(hwg)

    res = {"n": N, "cz": C, "grid": list(grid), "arch": str(dev.arch()),
           "reps": a.reps, "f1_keys_default": sorted(map(list, F1.F1_BLOCK_KEYS))}

    # Eligibility first: an arm that never fires is not an arm.
    F1.set_f1_cz128(True)
    res["f1_eligible"] = F1.eligible(x, xin, wp, wg)
    print("F1 eligible:", res["f1_eligible"], "grid", grid, "arch", res["arch"], flush=True)
    if res["f1_eligible"] is not None:
        json.dump(res, open(a.out, "w"), indent=1)
        return

    fns = {
        "A": lambda: arm_a(x, xin, wp, wg, ckc),
        "B": lambda: arm_f1(x, xin, wp, wg, ckc, grid, True),
        "C": lambda: arm_f1(x, xin, wp, wg, ckc, grid, False),
        "A2": lambda: arm_a(x, xin, wp, wg, ckc),
    }

    # PARITY, before timing: torch.equal of each arm against A.
    ref = fns["A"]()
    sync(dev)
    hr = ttnn.to_torch(ref)
    ttnn.deallocate(ref)
    res["where"] = {}
    res["parity"] = {}
    for k in ("B", "C", "A2"):
        r = fns[k]()
        sync(dev)
        res["where"][k] = str(r.memory_config().buffer_type)
        h = ttnn.to_torch(r)
        ttnn.deallocate(r)
        d = (h.float() - hr.float()).abs()
        res["parity"][k] = {"equal": bool(torch.equal(h, hr)),
                            "max_abs": float(d.max()), "n_diff": int((d > 0).sum()),
                            "n_elem": int(d.numel())}
        print(k, res["where"][k], res["parity"][k], flush=True)
    res["f1_out_l1_stats"] = list(F1.OUT_L1_STATS)

    # BYTES, buffer-address deduped, before timing perturbs anything.
    res["bytes"] = {}
    for k in ("A", "B", "C"):
        try:
            res["bytes"][k] = bytes_of(fns[k])
        except Exception as e:                                             # noqa: BLE001
            res["bytes"][k] = {"error": repr(e)[:200]}
        print("bytes", k, res["bytes"][k], flush=True)

    # TIMING, interleaved A/B/C/A2 round-robin so no arm owns the warm-up.
    samples = {k: [] for k in fns}
    for k in fns:                       # one warm pass each, discarded
        timed(dev, fns[k], 1)
    for _ in range(a.reps):
        for k in fns:
            samples[k] += timed(dev, fns[k], 1)
    res["ms"] = {k: [round(v * 1e3, 4) for v in v_] for k, v_ in samples.items()}
    med = {k: statistics.median(v) for k, v in samples.items()}
    res["median_ms"] = {k: round(v * 1e3, 4) for k, v in med.items()}
    res["ratio_vs_A"] = {k: round(med["A"] / med[k], 4) for k in ("B", "C", "A2")}
    res["aa_floor_pct"] = round(100.0 * (med["A2"] - med["A"]) / med["A"], 3)
    print(json.dumps({k: res[k] for k in ("median_ms", "ratio_vs_A", "aa_floor_pct")}, indent=1),
          flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
