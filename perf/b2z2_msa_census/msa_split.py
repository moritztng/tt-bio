#!/usr/bin/env python3
"""The MSA layer's stall split and the shape of its wait, whole and per sub-unit.

`stall_split.py` answers both questions for a whole block. It cannot answer them per sub-unit,
and on this block that is the question: the four sub-units of an `MSALayer` are not one regime.
`msa_probe.py --mark` puts one 32x32 `ttnn.exp` in front of each of them, so the device program
stream carries its own boundaries and a segment can be cut out of the capture without a device
sync per sub-unit. This reads those segments.

Everything under the segmentation is `stall_split.py`'s arithmetic and `census_tiles.py`'s
readers, unchanged: per-core division of both stall accumulators by the op's own `CORE COUNT`,
the same three candidate models (tile arrivals / bytes / bytes + a per-program constant), the
same R2 and Spearman. What is added is the cut and a per-segment repeat of the fit.

Usage: msa_split.py --csv OPS.csv[.gz] --meta prof_*.json --out OUT.json --label WH-c1
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
from census_tiles import arrivals, ksrc, operands, shape, traffic  # noqa: E402
from census_tiles import lstsq, r2, spearman  # noqa: E402

FENCE_DIM, FENCE_N = 32, 3
WAIT = "DEVICE COMPUTE CB WAIT FRONT [ns]"
RES = "DEVICE COMPUTE CB RESERVE BACK [ns]"
TRISC1 = "DEVICE TRISC1 KERNEL DURATION [ns]"
KERNEL = "DEVICE KERNEL DURATION [ns]"
CORES = "CORE COUNT"
NS_PER_TILE = {"WH": 71.3, "BH": 20.82}
SUBUNITS = ("pair_weighted_averaging", "msa_transition", "outer_product_mean", "pairformer_layer")


def f(row, key, default=0.0):
    try:
        return float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def is_marker(row):
    """A 32x32 unary: either a window fence (in a run of >= FENCE_N) or a sub-unit marker."""
    if not row.get("OP CODE", "").startswith("Unary"):
        return False
    return tuple(int(f(row, f"INPUT_0_{d}", 0)) for d in ("Y", "X")) in ((FENCE_DIM, FENCE_DIM),
                                                                        (0, 0))


def runs_of_markers(rows):
    out, i = [], 0
    while i < len(rows):
        if is_marker(rows[i]):
            j = i
            while j < len(rows) and is_marker(rows[j]):
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def find_region(rows):
    """The window between the last two fences, keeping the first call's leading marker.

    A fence is FENCE_N 32x32 unaries and the replayed call opens with a 32x32 sub-unit marker of
    its own, so the opening fence and the first call's first marker read as ONE run of FENCE_N+1
    and `rows[run_end:]` silently eats that marker -- which is what makes the window 2345 rather
    than 3 x 782. Take the opening fence to be exactly its first FENCE_N rows instead.
    """
    fences = [r for r in runs_of_markers(rows) if r[1] - r[0] >= FENCE_N]
    if len(fences) < 2:
        raise SystemExit(f"expected >=2 fence runs, found {len(fences)}")
    return rows[fences[-2][0] + FENCE_N:fences[-1][0]]


def split_segments(chunk):
    """One rep -> {sub-unit: [rows]}, cut at the four single-marker boundaries."""
    marks = [a for a, b in runs_of_markers(chunk) if b - a == 1]
    if len(marks) != len(SUBUNITS):
        raise SystemExit(f"expected {len(SUBUNITS)} sub-unit markers in a rep, found {len(marks)}")
    bounds = list(zip(marks, marks[1:] + [len(chunk)]))
    return {n: chunk[a + 1:b] for n, (a, b) in zip(SUBUNITS, bounds)}


def split(rows, arch):
    """The stall split and the three-model fit over one list of program rows."""
    t1 = w = rb = kern = 0.0
    dead_k = dead_n = 0
    sites = defaultdict(lambda: defaultdict(float))
    for r in rows:
        n = max(f(r, CORES), 1.0)
        t, wi, wo, k = f(r, TRISC1), f(r, WAIT) / n, f(r, RES) / n, f(r, KERNEL)
        t1, w, rb, kern = t1 + t, w + wi, rb + wo, kern + k
        if t <= 0:
            dead_k += k
            dead_n += 1
            continue
        cores = max(int(n), 1)
        arr, mac, _pack = arrivals(r, cores)
        out = shape(r, "OUTPUT_0")
        key = (r["OP CODE"], ksrc(r),
               tuple("x".join(map(str, s)) for _, s, _, _, _ in operands(r)),
               "x".join(map(str, out)) if out else "", cores)
        s = sites[key]
        s["n"] += 1
        s["arrivals_pc"] += arr
        s["macs_pc"] += mac
        for tk, tv in traffic(r).items():
            s[tk] += tv
        s["wait_ns"] += wi
        s["kernel_ns"] += k
        s["trisc1_ns"] += t
    S = [dict(v, op=k[0], kernel_src=k[1], inputs=list(k[2]), output=k[3], cores=k[4])
         for k, v in sites.items()]
    y = [s["wait_ns"] for s in S]
    fits, decomp = {}, {}
    models = {
        "tile arrivals only": lambda s: [s["arrivals_pc"]],
        "DRAM bytes only": lambda s: [s["dram_rd"]],
        "DRAM + L1 bytes": lambda s: [s["dram_rd"], s["l1_rd"]],
        "DRAM + L1 + per-program": lambda s: [s["dram_rd"], s["l1_rd"], s["n"]],
    }
    if len(S) >= 4 and sum(y) > 0:
        for name, fn in models.items():
            A = [fn(s) for s in S]
            try:
                x = lstsq(A, y)
            except ZeroDivisionError:
                continue
            fits[name] = {"r2": r2(A, y, x), "coef": x}
        c = fits.get("DRAM + L1 + per-program", {}).get("coef")
        if c:
            decomp = {"DRAM bytes": c[0] * sum(s["dram_rd"] for s in S) / 1e6,
                      "L1 bytes": c[1] * sum(s["l1_rd"] for s in S) / 1e6,
                      "per-program": c[2] * sum(s["n"] for s in S) / 1e6,
                      "per_program_ns": c[2]}
    rank = {}
    if sum(y) > 0 and len(S) >= 4:
        rank = {n: spearman(y, [k(s) for s in S]) for n, k in (
            ("bytes delivered (DRAM + L1 read)", lambda s: s["dram_rd"] + s["l1_rd"]),
            ("DRAM bytes read", lambda s: s["dram_rd"]),
            ("CB tile arrivals per core", lambda s: s["arrivals_pc"]),
            ("programs in the site", lambda s: s["n"]))}
    n_compute = sum(int(s["n"]) for s in S)
    return {
        "programs": len(rows), "compute_programs": n_compute,
        "zero_residency_programs": dead_n, "zero_residency_kernel_ms": dead_k / 1e6,
        "kernel_ms": kern / 1e6, "trisc1_ms": t1 / 1e6,
        "wait_in_ms": w / 1e6, "wait_out_ms": rb / 1e6, "compute_ms": (t1 - w - rb) / 1e6,
        "mean_compute_program_us": (kern - dead_k) / 1e3 / max(n_compute, 1),
        "arrivals_pc": sum(s["arrivals_pc"] for s in S),
        "datum_rate_ms": sum(s["arrivals_pc"] for s in S) * NS_PER_TILE[arch] / 1e6,
        "bytes_MB": {k: sum(s[k] for s in S) / 1e6
                     for k in ("dram_rd", "dram_wr", "l1_rd", "l1_wr")},
        "n_sites": len(S), "fits": fits, "spearman": rank, "decomposition_ms": decomp,
        "top_sites": sorted(S, key=lambda s: -s["kernel_ns"])[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arch", default="WH", choices=("WH", "BH"))
    a = ap.parse_args()

    rows = list(csv.DictReader(gzip.open(a.csv, "rt") if a.csv.suffix == ".gz" else open(a.csv)))
    meta = json.loads(a.meta.read_text())
    reps = int(meta["env"]["reps"])
    region = find_region(rows)
    if len(region) % reps:
        raise SystemExit(f"{len(region)} programs in the region is not divisible by {reps} reps")
    k = len(region) // reps
    chunks = [region[i * k:(i + 1) * k] for i in range(reps)]
    codes = [r["OP CODE"] for r in chunks[0]]
    for c in chunks[1:]:
        if [r["OP CODE"] for r in c] != codes:
            raise SystemExit("the reps are not the same program sequence")

    # median over reps, program position by program position, so one slow rep cannot move a site
    med_rows = []
    for i in range(k):
        cand = [c[i] for c in chunks]
        pick = sorted(range(reps), key=lambda j: f(cand[j], KERNEL))[reps // 2]
        med_rows.append(cand[pick])

    out = {"label": a.label, "arch": a.arch, "csv": str(a.csv), "reps": reps,
           "programs_per_call": k, "padded_rows": meta.get("padded_rows"),
           "tokens": meta.get("tokens"), "commit": meta["env"].get("commit"),
           "armed_ms_per_call": meta.get("armed_ms_per_call"),
           "whole": split(med_rows, a.arch),
           "subunits": {n: split(rs, a.arch)
                        for n, rs in split_segments(med_rows).items()}}
    a.out.write_text(json.dumps(out, indent=1))

    P = print
    W = out["whole"]
    P(f"\n=== {a.label} ({a.arch}) MSALayer: {k} device programs/call, "
      f"{out['padded_rows']} padded rows, {out['tokens']} tokens")
    P(f"  kernel time          {W['kernel_ms']:9.4f} ms")
    P(f"  TRISC1 resident      {W['trisc1_ms']:9.4f} ms")
    for key, lbl in (("wait_in_ms", "blocked on input tiles"),
                     ("wait_out_ms", "blocked on output room"),
                     ("compute_ms", "not stalled on a CB   ")):
        P(f"    {lbl} {W[key]:9.4f} ms  {100*W[key]/max(W['trisc1_ms'],1e-9):5.1f} % of TRISC1")
    P(f"  compute programs     {W['compute_programs']:6d}, mean "
      f"{W['mean_compute_program_us']:.1f} us")
    P(f"  byte-only programs   {W['zero_residency_programs']:6d}, "
      f"{W['zero_residency_kernel_ms']:.3f} ms")
    P("\n  What predicts the per-site input wait?")
    for n, v in W["fits"].items():
        P(f"    R2 = {v['r2']:8.4f}   {n:26s} " + "  ".join(f"{c:.5g}" for c in v["coef"]))
    P("\n  Spearman:")
    for n, v in W["spearman"].items():
        P(f"    {v:+.3f}   {n}")
    d = W["decomposition_ms"]
    if d:
        P(f"\n  The {W['wait_in_ms']:.3f} ms of input wait, in the 3-term fit "
          f"(per-program {d['per_program_ns']/1e3:.2f} us):")
        for key in ("DRAM bytes", "L1 bytes", "per-program"):
            P(f"    {d[key]:8.3f} ms  {100*d[key]/W['wait_in_ms']:5.1f} %   {key}")

    P(f"\n  Per sub-unit ({'ms' } of the {W['kernel_ms']:.3f} ms of kernel time):")
    P(f"    {'sub-unit':26s} {'progs':>6s} {'kernel':>9s} {'share':>7s} {'mean us':>8s} "
      f"{'in-wait':>8s} {'in %':>6s} {'MB':>8s} {'per-prog':>9s} {'bytes %':>8s}")
    for n, s in out["subunits"].items():
        dd = s["decomposition_ms"]
        b = s["bytes_MB"]
        tot = (dd.get("DRAM bytes", 0) + dd.get("L1 bytes", 0) + dd.get("per-program", 0)) or 1
        P(f"    {n:26s} {s['compute_programs']:6d} {s['kernel_ms']:9.3f} "
          f"{100*s['kernel_ms']/W['kernel_ms']:6.1f}% {s['mean_compute_program_us']:8.1f} "
          f"{s['wait_in_ms']:8.3f} {100*s['wait_in_ms']/max(s['trisc1_ms'],1e-9):5.1f}% "
          f"{b['dram_rd']+b['l1_rd']:8.1f} "
          f"{(dd.get('per_program_ns',0)/1e3):8.2f}u "
          f"{100*(dd.get('DRAM bytes',0)+dd.get('L1 bytes',0))/tot:7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
