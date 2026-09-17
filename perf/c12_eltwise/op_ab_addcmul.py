#!/usr/bin/env python3
"""Price the addcmul fold at the OP level, on the exact shapes the executed graph uses.

The trace (perf/c12_eltwise/runs/pairs_512.json) leaves three DRAM-real candidates, all on census
keys the fold runs at 52-61 % of the 442.9 GB/s DRAM roof. A byte prediction is only honest at a
key that is bandwidth-bound; at 58 % of the roof the class may be paying a per-call floor instead,
and then deleting a third of its bytes buys much less than a third of its time. This measures the
conversion instead of asserting it.

Instrument. One `synchronize_device` per CALL measures the host round trip, not the device: it
reads 49 us on a shape the census of record puts at 9.15 us, so the fused arm would win on
program count, which is exactly the framing this campaign refuses. Every arm here therefore
issues `--loop` calls inside ONE sync window and divides. `mul_only` is the calibration arm: its
us/call must land on the census's own number for the same key, or the instrument is not measuring
what the census measured and no ratio below it means anything.

    mul_only    multiply_(a, s, SIGMOID)                   calibration against the census key
    pair_ip     multiply_(a, s, SIGMOID) ; add_(., b)      what tenstorrent.py:9447-9448 runs
    fused_ip    addcmul(b, a, sigmoid(s))                  the candidate, sigmoid hoisted
    pair_oop    multiply(s, x) ; add(a, .)                 what tenstorrent.py:9622-9630 runs
    fused_oop   addcmul(a, s, x)                           the candidate, no activation needed

Hoisting the sigmoid is free at both sites: the trace shows the scale operand takes 17 distinct
buffers across 12000 calls at 9447 and 9 across 6000 at 9534, so it is memoised and the unary runs
per memo, not per call.

Arms interleave rep by rep in one session, and each pair arm runs twice per rep as that session's
own A/A floor. A ratio smaller than its own A/A floor is not a result. The clock is sampled DURING
the run from the host telemetry log and reported with every number.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T

TEL = "/home/ttuser/qbcard/cardtel.tsv"
DRAM_ROOF_GBS = 442.8767243360721


def clock_during(t0, t1, card):
    """AICLK samples for `card` whose timestamp lies inside [t0, t1]."""
    try:
        head, vals = None, []
        with open(TEL) as f:
            for line in f:
                if line.startswith("#epoch"):
                    head = line.rstrip("\n").split("\t")
                    continue
                if line.startswith("#") or head is None:
                    continue
                p = line.rstrip("\n").split("\t")
                if not (t0 <= float(p[0]) <= t1):
                    continue
                i = head.index("c%d_tt_aiclk" % card)
                if i < len(p) and p[i].strip():
                    vals.append(int(p[i]))
        if not vals:
            return {"samples": 0}
        return {"samples": len(vals), "min_MHz": min(vals), "max_MHz": max(vals),
                "median_MHz": st.median(vals)}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--loop", type=int, default=64)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "2")))
    ap.add_argument("--out", default="perf/c12_eltwise/runs/op_ab_addcmul.json")
    a = ap.parse_args()

    dev = T.get_device()
    torch.manual_seed(0)
    SIG = [ttnn.UnaryOpType.SIGMOID]

    def f(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    shapes = [("1x512x768", (1, 512, 768)), ("1x140x32x128", (1, 140, 32, 128))]
    results = {}
    t_start = time.time()

    for label, shp in shapes:
        nb = 1
        for d in shp:
            nb *= d
        nb *= 2
        a_src = f(torch.randn(*shp) * 0.7)
        s = f(torch.randn(*shp) * 1.3)
        b = f(torch.randn(*shp) * 0.4)
        x = f(torch.randn(*shp) * 0.9)
        sig_s = ttnn.sigmoid(s)          # hoisted, memoised in the fold; outside every window
        c = ttnn.clone(a_src)            # the in-place arms' victim, reused across the loop

        def w_mul_only(n):
            for _ in range(n):
                ttnn.multiply_(c, s, input_tensor_b_activations=SIG)

        def w_pair_ip(n):
            for _ in range(n):
                r = ttnn.multiply_(c, s, input_tensor_b_activations=SIG)
                ttnn.add_(r, b)

        def w_fused_ip(n):
            for _ in range(n):
                ttnn.addcmul(b, c, sig_s, output_tensor=c)

        def w_pair_oop(n):
            for _ in range(n):
                p = ttnn.multiply(s, x)
                r = ttnn.add(a_src, p)
                ttnn.deallocate(p)
                ttnn.deallocate(r)

        def w_fused_oop(n):
            for _ in range(n):
                r = ttnn.addcmul(a_src, s, x)
                ttnn.deallocate(r)

        def window(fn):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            fn(a.loop)
            ttnn.synchronize_device(dev)
            return (time.perf_counter() - t0) * 1e6 / a.loop     # us per call

        arms = [("mul_only", w_mul_only),
                ("pair_ip", w_pair_ip), ("aa_ip", w_pair_ip), ("fused_ip", w_fused_ip),
                ("pair_oop", w_pair_oop), ("aa_oop", w_pair_oop), ("fused_oop", w_fused_oop)]
        acc = {n: [] for n, _ in arms}
        for rep in range(a.warm + a.reps):
            for n, fn in arms:
                dt = window(fn)
                if rep >= a.warm:
                    acc[n].append(dt)

        med = {n: st.median(v) for n, v in acc.items()}

        # accuracy: one clean call of each form, compared on host
        c2 = ttnn.clone(a_src)
        r_pair = ttnn.add_(ttnn.multiply_(c2, s, input_tensor_b_activations=SIG), b)
        h_pair_ip = ttnn.to_torch(r_pair).float()
        ttnn.deallocate(r_pair)
        c3 = ttnn.clone(a_src)
        r_fused = ttnn.addcmul(b, c3, sig_s)
        h_fused_ip = ttnn.to_torch(r_fused).float()
        ttnn.deallocate(r_fused)
        ttnn.deallocate(c3)
        p_o = ttnn.multiply(s, x)
        r_po = ttnn.add(a_src, p_o)
        h_pair_oop = ttnn.to_torch(r_po).float()
        ttnn.deallocate(p_o)
        ttnn.deallocate(r_po)
        r_fo = ttnn.addcmul(a_src, s, x)
        h_fused_oop = ttnn.to_torch(r_fo).float()
        ttnn.deallocate(r_fo)

        d_ip = (h_fused_ip - h_pair_ip).abs()
        d_oop = (h_fused_oop - h_pair_oop).abs()
        results[label] = {
            "shape": list(shp), "bytes_per_tensor": nb, "loop": a.loop, "reps": a.reps,
            "us_per_call": med,
            "mul_only_GBs": 3 * nb / (med["mul_only"] * 1e-6) / 1e9,
            "aa_floor_ip": med["aa_ip"] / med["pair_ip"],
            "aa_floor_oop": med["aa_oop"] / med["pair_oop"],
            "ratio_ip": med["fused_ip"] / med["pair_ip"],
            "ratio_oop": med["fused_oop"] / med["pair_oop"],
            "us_saved_ip": med["pair_ip"] - med["fused_ip"],
            "us_saved_oop": med["pair_oop"] - med["fused_oop"],
            "us_saved_ip_at_roof": 2 * nb / (DRAM_ROOF_GBS * 1e9) * 1e6,
            "maxabs_ip": float(d_ip.max()), "maxabs_oop": float(d_oop.max()),
            "rel_ip": float(d_ip.max() / h_pair_ip.abs().max()),
            "rel_oop": float(d_oop.max() / h_pair_oop.abs().max()),
            "bitexact_ip": bool(d_ip.max() == 0), "bitexact_oop": bool(d_oop.max() == 0),
        }
        for t in (a_src, s, b, x, sig_s, c, c2):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        print("%-14s %s" % (label, json.dumps(med)), flush=True)

    t_end = time.time()
    clk = clock_during(t_start, t_end, a.card)
    out = {"card": a.card, "clock_during": clk, "t_start": t_start, "t_end": t_end,
           "loadavg": os.getloadavg(), "dram_roof_GBs": DRAM_ROOF_GBS, "results": results}
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))

    print("\nclock during: %s   loadavg %s" % (json.dumps(clk), out["loadavg"]))
    for label, r in results.items():
        print("\n%s  (%d B/tensor)  mul_only %.2f us/call = %.1f GB/s on the census byte model"
              % (label, r["bytes_per_tensor"], r["us_per_call"]["mul_only"], r["mul_only_GBs"]))
        print("  %-12s %9s %9s %8s %8s %10s %10s"
              % ("form", "pair_us", "fused_us", "ratio", "A/A", "us_saved", "maxabs"))
        print("  %-12s %9.2f %9.2f %8.4f %8.4f %10.2f %10.2e"
              % ("in-place", r["us_per_call"]["pair_ip"], r["us_per_call"]["fused_ip"],
                 r["ratio_ip"], r["aa_floor_ip"], r["us_saved_ip"], r["maxabs_ip"]))
        print("  %-12s %9.2f %9.2f %8.4f %8.4f %10.2f %10.2e"
              % ("out-of-place", r["us_per_call"]["pair_oop"], r["us_per_call"]["fused_oop"],
                 r["ratio_oop"], r["aa_floor_oop"], r["us_saved_oop"], r["maxabs_oop"]))
        print("  2N at the %.1f GB/s roof would be %.2f us/call"
              % (DRAM_ROOF_GBS, r["us_saved_ip_at_roof"]))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
