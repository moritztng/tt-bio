#!/usr/bin/env python3
"""The op ladder over the shapes the fold actually routes, not two hand-picked ones.

`mm1dcheck.py --time` prices one synthetic shape at two K values.  That was the right thing before
the fold integration existed; now the router has named the 23 signatures that really take this
path and how many calls each carries, so the ladder should price those, at the block config the
chooser really picks, and weight them by their real call counts.

Three arms, interleaved, same as the fold A/B and for the same reason:

  native   ttnn.linear on its own auto path -- what ships
  generic  the transcription, split OFF
  split    the transcription, split ON

plus a native A/A twin in the same session, because a ratio is only readable against this
session's own floor.  `native_aa` is measured last so it sees the same warmed state as the rest.

Every point is an n-ladder (n = 1, 2, 4, 8, 16) fitted to time = L + c*n, so the per-call cost c
is separated from the ~0.05 ms host bracket floor L that a `sync; call; sync` charges.  Reading an
intercept without the ladder is how a fixed cost gets billed as op time.

Each ladder point is the MINIMUM of its repetitions, not the median.  Contention only ever adds
time to a bracket -- a descheduled python thread, a stolen core, an evicted cache line all make a
call look slower and none makes it look faster -- so over enough repetitions the minimum converges
on the uncontended cost while the median tracks the load.  That is what makes this readable on a
host somebody else is folding on, and it is the whole reason the estimator changed.

It is not taken on trust.  `native_aa` is a second native arm, interleaved into every round exactly
like the three real ones, so the session measures its own floor with the same instrument, the same
warmth and the same neighbours as the ratio it reports.  `--contended` runs on a loud host and lets
that floor decide: `usable` is true only if the worst per-signature A/A floor is inside `--aa-bar`.
A session whose own twin moves more than the bar cannot price a lever and says so instead of
returning a plausible number.  `--anyway` is the weaker mode that only proves the harness executes.

The AICLK is forced to the target and sampled DURING the window either way; a number without a
clock is not a measurement.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import clk  # noqa: E402
import ttnn  # noqa: E402

QUIET = ROOT / "perf/c12_orchestrator/pair_guard/host_quiet.py"


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    sxy = sum((v - mx) * (u - my) for v, u in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b, max(abs(u - (my - b * mx + b * v)) for v, u in zip(xs, ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(HERE / "mm1dfold.jsonl"))
    ap.add_argument("--tag", default="sc_splitwide", help="route record to take signatures from")
    ap.add_argument("--top", type=int, default=6, help="signatures to price, by call count")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--anyway", action="store_true")
    ap.add_argument("--contended", action="store_true",
                    help="run on a host that is not quiet and let this session's own A/A "
                         "floor decide whether it counts; only honest with --stat min")
    ap.add_argument("--stat", choices=("min", "median"), default="min")
    ap.add_argument("--aa-bar", type=float, default=0.5,
                    help="worst per-signature A/A floor, in percent, a contended session "
                         "may carry and still count as a measurement")
    ap.add_argument("--out", default=str(HERE / "mm1dopladder.json"))
    a = ap.parse_args()

    quiet = subprocess.run([sys.executable, str(QUIET)], capture_output=True, text=True)
    host_quiet = quiet.returncode == 0
    if not host_quiet and not (a.anyway or a.contended):
        print(quiet.stdout.strip()[-400:])
        print("REFUSING: host not quiet.  Re-run when it is, --contended to let the A/A "
              "floor decide, or --anyway for a bare harness check.")
        return 3

    import torch
    from tt_bio import tenstorrent as T
    from tt_bio.mm1d_generic import systolic_1d_config, generic_mm1d

    recs = [json.loads(l) for l in open(a.jsonl)]
    picked = [x for x in recs if x["tag"] == a.tag]
    if not picked:
        print("no record tagged %r in %s" % (a.tag, a.jsonl))
        return 2
    sigs = picked[-1]["route_shapes"][:a.top]

    device = T.get_device()
    G = T.CORE_GRID_MAIN
    grid = (G.x, G.y)
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    CKC = (ttnn.MathFidelity.HiFi4, False, True, True)
    DR = ttnn.DRAM_MEMORY_CONFIG
    torch.manual_seed(0)

    held = clk.force(a.clock, clk.nodes_open_by_this_process())
    t0 = time.time()
    while time.time() - t0 < 10.0 and not all(clk.aiclk(n) >= a.clock - 5 for n in held):
        time.sleep(0.01)
    reached = {n: clk.aiclk(n) for n in held}
    if any(v < a.clock - 5 for v in reached.values()) and not a.anyway:
        print("REFUSING to measure: clock did not hold, %r" % (reached,))
        return 2
    print("nodes %r forced to %d MHz (host_quiet=%s)" % (held, a.clock, host_quiet), flush=True)
    sampler = clk.Sampler(held[0])

    cases, failed = [], []
    for s in sigs:
        ash, bsh = s["a"], s["b"]
        x = ttnn.from_torch(torch.randn(*ash) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DR)
        w = ttnn.from_torch(torch.randn(*bsh) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DR)
        pc, mc0 = systolic_1d_config(x, w, grid, True, ttnn.bfloat16)
        if pc is None or mc0:
            print("  skipping %s x %s: no 1D config" % (ash, bsh))
            continue
        step = "native"
        try:
            ref = ttnn.linear(x, w, compute_kernel_config=KC, memory_config=DR,
                              dtype=ttnn.bfloat16, core_grid=G)
            ttnn.synchronize_device(device)
            spec = ttnn.TensorSpec(ttnn.Shape([int(d) for d in ref.shape]), ttnn.bfloat16,
                                   ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM)
            tr = ttnn.to_torch(ref)
            ttnn.deallocate(ref)
            og, os_ = (ttnn.allocate_tensor_on_device(spec, device),
                       ttnn.allocate_tensor_on_device(spec, device))
            step = "generic"
            eq_g = bool(torch.equal(tr, ttnn.to_torch(
                generic_mm1d(device, x, w, og, pc, CKC))))
            step = "split"
            eq_s = bool(torch.equal(tr, ttnn.to_torch(
                generic_mm1d(device, x, w, os_, pc, CKC, writer_on_in0=True))))
        except Exception as e:
            # Which arm refused matters: the wheel's own op failing is a shape this ladder may
            # not price at all, while the transcription failing is a coverage defect in mine.
            head = str(e).strip().splitlines()
            failed.append({"a": ash, "b": bsh, "n": s["n"], "arm": step,
                           "err": head[0][:300] if head else repr(e)[:300]})
            print("  FAILED TO BUILD %s x %s at the %s arm: %s"
                  % (ash, bsh, step, (head[0] if head else repr(e))[:200]), flush=True)
            continue
        cases.append({"a": ash, "b": bsh, "n": s["n"], "pc": pc, "x": x, "w": w,
                      "og": og, "os": os_, "generic_bitexact": eq_g, "split_bitexact": eq_s})
        print("  %s x %s  n=%-6d cfg=%s  bit-exact generic %s split %s"
              % (ash, bsh, s["n"], pc[1:], eq_g, eq_s), flush=True)

    def native(c):
        return ttnn.linear(c["x"], c["w"], compute_kernel_config=KC, memory_config=DR,
                           dtype=ttnn.bfloat16, core_grid=G)

    def generic(c):
        return generic_mm1d(device, c["x"], c["w"], c["og"], c["pc"], CKC)

    def split(c):
        return generic_mm1d(device, c["x"], c["w"], c["os"], c["pc"], CKC, writer_on_in0=True)

    ARMS = {"native": native, "generic": generic, "split": split}

    AGG = min if a.stat == "min" else st.median

    def ladder(fn, c, ns=(1, 2, 4, 8, 16)):
        pts = []
        for n in ns:
            ts = []
            for i in range(a.reps + a.warm):
                ttnn.synchronize_device(device)
                s0 = time.perf_counter()
                rs = [fn(c) for _ in range(n)]
                ttnn.synchronize_device(device)
                e0 = time.perf_counter()
                for r in rs:
                    if r is not c["og"] and r is not c["os"]:
                        ttnn.deallocate(r)
                if i >= a.warm:
                    ts.append((e0 - s0) * 1e3)
            pts.append((n, AGG(ts)))
        L, cc, res = fit([p[0] for p in pts], [p[1] for p in pts])
        return {"pts": pts, "L": L, "c": cc, "resid": res}

    for c in cases:
        for fn in ARMS.values():
            for _ in range(3):
                fn(c)
        ttnn.synchronize_device(device)

    if not cases:
        print("no signature built; nothing to price")
        return 2

    plan = []
    for _ in range(a.rounds):
        plan += list(ARMS.items()) + [("native_aa", native)]

    R = {"clock_target": a.clock, "host_quiet": host_quiet, "usable": None,
         "stat": a.stat, "aa_bar_pct": a.aa_bar, "contended": bool(a.contended),
         "tag": a.tag, "arms": {}, "failed_signatures": failed}
    for name, fn in plan:
        for i, c in enumerate(cases):
            key = "%s_%d" % (name, i)
            res = ladder(fn, c)
            R["arms"].setdefault(key, []).append(res)
            print("  %-14s %-24s c = %.5f ms  L = %.5f  resid %.5f"
                  % (name, "%s" % (c["a"],), res["c"], res["L"], res["resid"]), flush=True)

    R["clock"] = sampler.stop()
    print("AICLK during: %r" % (R["clock"],), flush=True)

    best = lambda k: AGG([r["c"] for r in R["arms"][k]])
    tot = {"native": 0.0, "generic": 0.0, "split": 0.0}
    R["cases"] = []
    for i, c in enumerate(cases):
        nv, gn, sp = best("native_%d" % i), best("generic_%d" % i), best("split_%d" % i)
        aa = best("native_aa_%d" % i)
        floor = 100 * abs(nv - aa) / aa
        for k, v in (("native", nv), ("generic", gn), ("split", sp)):
            tot[k] += v * c["n"]
        R["cases"].append({"a": c["a"], "b": c["b"], "n": c["n"], "pc": list(c["pc"][1:]),
                           "native_ms": nv, "generic_ms": gn, "split_ms": sp,
                           "generic_over_native": gn / nv, "native_over_split": nv / sp,
                           "generic_over_split": gn / sp, "aa_floor_pct": floor,
                           "generic_bitexact": c["generic_bitexact"],
                           "split_bitexact": c["split_bitexact"]})
        print("%-24s n=%-6d native %.5f generic %.5f split %.5f ms | gen/nat %.4fx  "
              "nat/split %.4fx  gen/split %.4fx  A/A %.3f %%"
              % ("%s" % (c["a"],), c["n"], nv, gn, sp, gn / nv, nv / sp, gn / sp, floor),
              flush=True)

    floors = [c["aa_floor_pct"] for c in R["cases"]]
    R["aa_floor_max_pct"] = max(floors) if floors else None
    R["usable"] = bool(floors) and R["aa_floor_max_pct"] <= a.aa_bar
    R["weighted_ms_per_fold"] = tot
    print("\nweighted by real call counts, these %d signatures cost per fold:" % len(cases))
    print("  native %.2f ms   generic %.2f ms   split %.2f ms"
          % (tot["native"], tot["generic"], tot["split"]))
    if tot["split"]:
        print("  split saves %.2f ms against native and %.2f ms against the transcription"
              % (tot["native"] - tot["split"], tot["generic"] - tot["split"]))
    print("\nA/A floor worst %.3f %% against a %.3f %% bar; host_quiet=%s; stat=%s -> "
          "usable=%s" % (R["aa_floor_max_pct"], a.aa_bar, host_quiet, a.stat, R["usable"]))
    if not R["usable"]:
        print("This session cannot price the lever: its own A/A twin moves more than the "
              "bar.  No ratio from it may be quoted.")
    Path(a.out).write_text(json.dumps(
        {k: v for k, v in R.items() if k != "_"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
