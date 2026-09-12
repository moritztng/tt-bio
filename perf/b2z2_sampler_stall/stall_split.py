#!/usr/bin/env python3
"""The diffusion step's inner stall split, and what shape its wait has.

`b2z2-sampler-ceiling-map` established that `DEVICE COMPUTE CB WAIT FRONT` and
`DEVICE COMPUTE CB RESERVE BACK` are blank in all 1066 rows of every committed diffusion-step
capture on both architectures, so the campaign's floor statement -- "57.0 % of the math thread's
resident time is blocked on input tiles" -- has only ever been a Pairformer-block number. This
reads a step capture taken with `--enable-sum-profiling` armed and fills those columns in.

Three questions, in the order `b2z2-sharded-sampler` needs them:

  1. the split: input-tile wait / output-room wait / compute, as FRACTIONS of math-thread
     residency, overall and per op code;
  2. whether the wait is byte-shaped, the way `CONTEXT §2-CORRECTION-B` found the trunk's is
     (bytes R2 0.79/0.55 against the tile-count model's -0.56/-0.57);
  3. the programs whose math thread is never resident at all -- the ones that only move bytes.

Both stall accumulators are SUMMED OVER THE CORES that ran the op while TRISC1 residency is a
duration, so every op is divided by its own `CORE COUNT` before anything is added, exactly as
`perf/b2z2_profiler/cb_split.py` does. Fractions are the deliverable, not seconds: 1066 tiny
programs per call is well outside the 1.0079x perturbation measured on a 272-op block.

Usage: stall_split.py --csv OPS.csv[.gz] --meta step_prof_*.json --out OUT.json --label WH-c2
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "b2z2_tile_census"))
from census_tiles import arrivals, ksrc, operands, shape, tiles, traffic  # noqa: E402
from census_tiles import lstsq, r2, spearman  # noqa: E402

FENCE_DIM, FENCE_N = 32, 3
WAIT = "DEVICE COMPUTE CB WAIT FRONT [ns]"
RES = "DEVICE COMPUTE CB RESERVE BACK [ns]"
TRISC1 = "DEVICE TRISC1 KERNEL DURATION [ns]"
KERNEL = "DEVICE KERNEL DURATION [ns]"
CORES = "CORE COUNT"
NS_PER_TILE = {"WH": 71.3, "BH": 20.82}     # b2z2-datum-rate-floor; WH is the derived figure


def f(row, key, default=0.0):
    try:
        return float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def is_fence(row):
    if not row.get("OP CODE", "").startswith("Unary"):
        return False
    return tuple(int(f(row, f"INPUT_0_{d}", 0)) for d in ("Y", "X")) in (
        (FENCE_DIM, FENCE_DIM), (0, 0))


def find_region(rows):
    runs, i = [], 0
    while i < len(rows):
        if is_fence(rows[i]):
            j = i
            while j < len(rows) and is_fence(rows[j]):
                j += 1
            if j - i >= FENCE_N:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < 2:
        raise SystemExit(f"expected >=2 fence runs, found {len(runs)}")
    return rows[runs[-2][1]:runs[-1][0]]


def open_csv(p: Path):
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arch", default="WH", choices=("WH", "BH"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open_csv(a.csv)))
    meta = json.loads(a.meta.read_text())
    reps = int(meta["env"]["reps"])
    region = find_region(rows)
    if len(region) % reps:
        raise SystemExit(f"{len(region)} ops in the region is not divisible by {reps} reps")
    n_ops = len(region) // reps
    chunks = [region[i * n_ops:(i + 1) * n_ops] for i in range(reps)]

    ns_per_cycle = st.median([f(r, "DEVICE FW DURATION [ns]") /
                              (f(r, "DEVICE FW END CYCLE") - f(r, "DEVICE FW START CYCLE"))
                              for r in region
                              if f(r, "DEVICE FW END CYCLE") > f(r, "DEVICE FW START CYCLE")])

    per_rep, by_code = [], defaultdict(lambda: defaultdict(float))
    bad = 0
    for ch in chunks:
        span = (f(ch[-1], "DEVICE FW END CYCLE") - f(ch[0], "DEVICE FW START CYCLE")) * ns_per_cycle
        t1 = w = rb = kern = 0.0
        dead_kern = dead_n = 0.0
        for r in ch:
            n = max(f(r, CORES), 1.0)
            t, wi, wo = f(r, TRISC1), f(r, WAIT) / n, f(r, RES) / n
            k = f(r, KERNEL)
            t1, w, rb, kern = t1 + t, w + wi, rb + wo, kern + k
            if t <= 0:
                dead_kern += k
                dead_n += 1
            if wi + wo > t and t > 0:
                bad += 1
            c = by_code[r.get("OP CODE", "?")]
            c["trisc1"] += t
            c["wait_in"] += wi
            c["wait_out"] += wo
            c["kernel"] += k
            c["n"] += 1
            if t <= 0:
                c["dead_kernel"] += k
                c["dead_n"] += 1
        per_rep.append({"span_ms": span / 1e6, "kernel_ms": kern / 1e6, "trisc1_ms": t1 / 1e6,
                        "wait_in_ms": w / 1e6, "wait_out_ms": rb / 1e6,
                        "compute_ms": (t1 - w - rb) / 1e6,
                        "zero_residency_kernel_ms": dead_kern / 1e6,
                        "zero_residency_programs": dead_n})

    med = {k: st.median([r[k] for r in per_rep]) for k in per_rep[0]}
    span, t1m = med["span_ms"], med["trisc1_ms"]

    # --- site-level shape of the wait: bytes vs tiles vs programs ------------------------
    sites = defaultdict(lambda: defaultdict(float))
    for r in chunks[0]:
        cores = max(int(f(r, CORES)), 1)
        arr, mac, _pack = arrivals(r, cores)
        tr = traffic(r)
        out = shape(r, "OUTPUT_0")
        key = (r["OP CODE"], ksrc(r),
               tuple("x".join(map(str, s)) for _, s, _, _, _ in operands(r)),
               "x".join(map(str, out)) if out else "", cores)
        s = sites[key]
        s["n"] += 1
        s["arrivals_pc"] += arr
        s["macs_pc"] += mac
        for tk, tv in tr.items():
            s[tk] += tv
        s["wait_ns"] += f(r, WAIT) / cores
        s["resv_ns"] += f(r, RES) / cores
        s["trisc1_ns"] += f(r, TRISC1)
        s["kernel_ns"] += f(r, KERNEL)
    S = [dict(v, op=k[0], kernel=k[1], inputs=list(k[2]), output=k[3], cores=k[4])
         for k, v in sites.items() if v["trisc1_ns"] > 0]
    y = [s["wait_ns"] for s in S]
    fits = {
        "tile arrivals only": lambda s: [s["arrivals_pc"]],
        "tile-pair MACs only": lambda s: [s["macs_pc"]],
        "DRAM bytes only": lambda s: [s["dram_rd"]],
        "DRAM + L1 bytes": lambda s: [s["dram_rd"], s["l1_rd"]],
        "DRAM + L1 + per-program": lambda s: [s["dram_rd"], s["l1_rd"], s["n"]],
        "DRAM + L1 + tile arrivals": lambda s: [s["dram_rd"], s["l1_rd"], s["arrivals_pc"]],
    }
    fit_out, coef = {}, None
    have_wait = sum(y) > 0        # the committed BH step capture has both CB columns blank
    for name, fn in (fits.items() if have_wait else ()):
        A = [fn(s) for s in S]
        x = lstsq(A, y)
        fit_out[name] = {"r2": r2(A, y, x), "coef": x}
        if name == "DRAM + L1 bytes":
            coef = x
    rank_out = {name: spearman(y, [key(s) for s in S]) for name, key in (() if not have_wait else (
        ("bytes delivered (DRAM + L1 read)", lambda s: s["dram_rd"] + s["l1_rd"]),
        ("DRAM bytes read", lambda s: s["dram_rd"]),
        ("CB tile arrivals per core", lambda s: s["arrivals_pc"]),
        ("tile-pair MACs per core", lambda s: s["macs_pc"]),
        ("programs in the site", lambda s: s["n"])))}

    arr_tot = sum(s["arrivals_pc"] for s in S)
    out = {
        "label": a.label, "arch": a.arch, "csv": str(a.csv),
        "programs_per_step": n_ops, "reps": reps, "ns_per_cycle": ns_per_cycle,
        "per_rep_ms": per_rep, "median_ms": med,
        "synced_wall_ms_per_call": meta.get("synced_wall_ms_per_call"),
        "trisc1_pct": {k: 100.0 * med[k] / t1m
                       for k in ("wait_in_ms", "wait_out_ms", "compute_ms")},
        "span_pct": {k: 100.0 * med[k] / span
                     for k in ("trisc1_ms", "kernel_ms", "wait_in_ms", "wait_out_ms",
                               "compute_ms")},
        "non_resident_ms": span - t1m,
        "gap_ms": span - med["kernel_ms"],
        "ops_where_stall_exceeds_residency": bad // reps,
        "movement_free_multiplier": span / (span - med["wait_in_ms"] - med["wait_out_ms"]),
        "have_wait": have_wait,
        "by_op_code": {k: {"n": int(v["n"] // reps),
                           "trisc1_ms": v["trisc1"] / reps / 1e6,
                           "wait_in_ms": v["wait_in"] / reps / 1e6,
                           "wait_out_ms": v["wait_out"] / reps / 1e6,
                           "compute_ms": (v["trisc1"] - v["wait_in"] - v["wait_out"]) / reps / 1e6,
                           "kernel_ms": v["kernel"] / reps / 1e6,
                           "dead_kernel_ms": v["dead_kernel"] / reps / 1e6,
                           "dead_n": int(v["dead_n"] // reps)}
                       for k, v in sorted(by_code.items(), key=lambda kv: -kv[1]["kernel"])},
        "site_fits": fit_out, "site_spearman": rank_out, "n_sites": len(S),
        "arrivals_pc": arr_tot,
        "datum_rate_ms": arr_tot * NS_PER_TILE[a.arch] / 1e6,
        "bytes_MB": {k: sum(s[k] for s in S) / 1e6
                     for k in ("dram_rd", "dram_wr", "l1_rd", "l1_wr")},
        "sites": sorted(S, key=lambda s: -s["wait_ns"])[:25],
    }
    a.out.write_text(json.dumps(out, indent=1))

    P = print
    P(f"\n=== {a.label} ({a.arch}): {n_ops} programs/step, span {span:.4f} ms, "
      f"{reps} reps, spread "
      f"{100*(max(r['span_ms'] for r in per_rep)/min(r['span_ms'] for r in per_rep)-1):.2f} %")
    P(f"  kernel time            {med['kernel_ms']:9.4f} ms  "
      f"{100*med['kernel_ms']/span:5.1f} % of span")
    P(f"  TRISC1 resident        {t1m:9.4f} ms  {100*t1m/span:5.1f} % of span")
    for k, lbl in (("wait_in_ms", "CB wait-front (input) "),
                   ("wait_out_ms", "CB reserve-back (out) "),
                   ("compute_ms", "not stalled on a CB   ")):
        P(f"    {lbl} {med[k]:9.4f} ms  {out['trisc1_pct'][k]:5.1f} % of TRISC1"
          f"   {100*med[k]/span:5.1f} % of span")
    P(f"  non-resident           {out['non_resident_ms']:9.4f} ms  "
      f"{100*out['non_resident_ms']/span:5.1f} % of span")
    P(f"  gap (span - kernels)   {out['gap_ms']:9.4f} ms")
    P(f"  identity check: in+out+compute+non-resident = "
      f"{med['wait_in_ms']+med['wait_out_ms']+med['compute_ms']+out['non_resident_ms']:.4f} ms "
      f"vs span {span:.4f} ms")
    P(f"  movement-free multiplier on the step span: "
      f"{out['movement_free_multiplier']:.4f}x")
    P(f"  zero-residency programs {int(med['zero_residency_programs']):4d}  "
      f"{med['zero_residency_kernel_ms']:.4f} ms  "
      f"{100*med['zero_residency_kernel_ms']/med['kernel_ms']:.1f} % of kernel time")
    P(f"  divisor error bar: {out['ops_where_stall_exceeds_residency']}/{n_ops} ops have "
      f"per-core stall > per-core TRISC1")
    P("\n  by op code (ms/step, per core, ordered by device kernel time):")
    P(f"    {'op code':32s} {'n':>4s} {'kernel':>8s} {'TRISC1':>8s} {'in':>8s} {'in%':>6s} "
      f"{'out':>7s} {'compute':>8s}")
    for code, v in list(out["by_op_code"].items())[:14]:
        P(f"    {code[:32]:32s} {v['n']:4d} {v['kernel_ms']:8.3f} {v['trisc1_ms']:8.3f} "
          f"{v['wait_in_ms']:8.3f} "
          f"{100*v['wait_in_ms']/max(v['trisc1_ms'],1e-9):5.1f}% {v['wait_out_ms']:7.3f} "
          f"{v['compute_ms']:8.3f}")
    P(f"\n  What predicts the per-site wait? {len(S)} sites that run a compute kernel:")
    for name, v in fit_out.items():
        P(f"    R2 = {v['r2']:8.4f}   {name:28s} " + "  ".join(f"{c:.5g}" for c in v["coef"]))
    if coef and coef[0] > 0 and coef[1] > 0:
        P(f"    two-bandwidth fit: DRAM {1/coef[0]:,.1f} GB/s, "
          f"L1-interleaved {1/coef[1]:,.1f} GB/s")
    P("\n  Spearman rank correlation against the measured per-site wait:")
    for name, v in rank_out.items():
        P(f"    {v:+.3f}   {name}")
    P(f"\n  CB tile arrivals per core-averaged step {arr_tot:,.0f}"
      f"   x {NS_PER_TILE[a.arch]} ns = {out['datum_rate_ms']:.4f} ms"
      + (f" = {100*out['datum_rate_ms']/med['wait_in_ms']:.1f} % of the input wait"
         if med['wait_in_ms'] else "   (CB columns blank in this capture)"))
    b = out["bytes_MB"]
    P(f"  bytes: DRAM read {b['dram_rd']:.1f} MB, L1 read {b['l1_rd']:.1f} MB, "
      f"DRAM write {b['dram_wr']:.1f} MB, L1 write {b['l1_wr']:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
