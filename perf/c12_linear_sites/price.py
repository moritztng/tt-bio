#!/usr/bin/env python3
"""Price each sibling set as a RATE change, from the census's own measured N-families.

Mechanism, per set: n projections that read one activation become one matmul of the summed N. The
arithmetic is identical; what changes is reuse -- the shared activation is read once instead of n
times, and the wider N gives the kernel more output per K-pass. So the prediction is a RATE, and
the census already measured how rate moves with N at fixed (batch, M, K):

    (1,512,768)    N=768 41.20  N=1536 53.72  N=3072 65.66 TFLOP/s
    (1,512,384)    N=384 18.63  N=1536 39.22
    (1,512,1536)   N=384 30.36  N=768  50.74
    (140,32,128)   N=128  9.10  N=256  11.86
    (512,512,128)  N=8     1.41 N=16    2.82  N=128  9.12

Where the fused N is itself a measured key, its measured rate is used. Where it is not, the rate is
a log-log fit in N over that family and is FLAGGED as extrapolated. Where the family has one point,
the set is REFUSED a rate rather than given a borrowed one.

Two costs are charged against every predicted win:
  * the SLICE TAX. Each member's consumer needs its own tensor, and ttnn has no offset view of a
    TILE-layout tensor, so recovering n halves costs n slices. Each is charged the larger of the
    session's own 5.5968 us/call launch floor and its bytes at the measured 442.9 GB/s DRAM roof.
  * an L1 activation. A set whose shared activation lives in L1 (read from the fold's own capture,
    not assumed) re-reads L1, not DRAM, so the byte half of the reuse argument does not apply to it
    and only the rate half can.
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import log
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOCK_MHZ = 1350.0
LAUNCH_US = 5.596765862639961     # c10-fold-census sweep2 budget.json, dispatch.session_floor
DRAM_GBs = 442.9


def families(cen):
    fam = defaultdict(dict)
    for k in cen.values():
        out = k["out"]
        b = 1
        for d in out[:-2]:
            b *= d
        fam[(b, out[-2], k["K"])][out[-1]] = k["TFLOPs"]
    return fam


def rate_at(fam, key, n):
    """The rate at the fused N, and where it came from.

    Extrapolation is capped at 2x the family's widest measured N. Two or three points fix a local
    slope, not a scaling law (`c10-two-points-cannot-measure-a-scaling-exponent`), and the fit is
    plainly wrong far out: carried to N=36864 it reads 135 TFLOP/s, above anything this chip has
    ever been measured at on these shapes. Beyond 2x, the widest MEASURED rate is used instead and
    labelled `capped`, which understates the win rather than inventing one."""
    pts = sorted(fam.get(key, {}).items())
    if n in dict(pts):
        return dict(pts)[n], "measured"
    if len(pts) < 2:
        return None, "refused (one point in family)"
    (n0, r0), (n1, r1) = pts[-2], pts[-1]
    s = log(r1 / r0) / log(n1 / n0)
    if n > 2 * n1:
        return r1, "capped at measured N=%d (fit would need %.1fx extrapolation)" % (n1, n / n1)
    return r1 * (n / n1) ** s, "extrapolated (slope %.3f from N=%d,%d)" % (s, n0, n1)


def main() -> int:
    sites = json.loads((HERE / "sites_512.json").read_text())
    budget = json.loads((HERE / "census_budget_sweep2.json").read_text())
    cen = {k["key"]: k for k in budget["keys"] if k["arm"] == "linear"}
    fam = families(cen)

    out = []
    for s in sites["sets"]:
        ms = s["members"]
        n = len(ms)
        b = 1
        for d in s["act_shape"][:-2]:
            b *= d
        M, K = s["act_shape"][-2], s["K"]
        flop = 0.0
        for m in ms:
            c = cen[m["key"]]
            flop += (s["linear_calls"] / n) * c["TFLOP"] * 1e12 / c["calls"]
        n_fused = sum(m["N"] for m in ms)
        r_now = flop / s["fold_s"] / 1e12
        r_new, how = rate_at(fam, (b, M, K), n_fused)
        # byte floor: same achieved GB/s, minus the re-read. Only DRAM re-reads count.
        gbs = s["GBs"]
        byte_win = (s["reread_B"] / (gbs * 1e9)) if s["act_mem"] == "DRAM" else 0.0
        s_new = flop / (r_new * 1e12) if r_new else None
        rate_win = (s["fold_s"] - s_new) if s_new else None
        # SLICE TAX. Recovering member j from the fused output copies that member's own output:
        # a read and a write of batch*M*N_j*2 B. Charged at the larger of the session's launch
        # floor and those bytes at the measured DRAM roof. This is where the geometry decides:
        # a set whose shared activation is bigger than one member's output (PairWeightedAveraging,
        # act 67.1 MB against a 33.6 MB member) can pay it; a set of wide outputs off a narrow
        # activation (every AdaLN and transition set) cannot.
        # An L1-resident set gets no tax number and no byte win: both its shared read and its
        # slices are L1 traffic, which the 442.9 GB/s DRAM roof does not price at all.
        inst = s["linear_calls"] / n
        slice_s = (sum(inst * max(LAUNCH_US * 1e-6, 2 * b * M * m["N"] * 2 / (DRAM_GBs * 1e9))
                       for m in ms) if s["act_mem"] == "DRAM" else None)
        out.append({**s, "n": n, "N_fused": n_fused, "flop": flop, "r_now": r_now,
                    "r_new": r_new, "how": how, "s_new": s_new, "rate_win_s": rate_win,
                    "byte_win_s": byte_win, "slice_tax_s": slice_s,
                    "net_s": (rate_win - slice_s)
                    if (rate_win is not None and slice_s is not None) else None})

    print("== MECHANISM: reuse, priced as a rate. Clock 1350 MHz; cycles = s x 1350 Mc/s ==")
    hdr = ("set", "n", "calls", "now_s", "TF/s", "N_fus", "pred TF/s", "pred_s", "rate_win",
           "byte_win", "slice_tax", "net_s", "net_Mc")
    print("%-46s %2s %7s %7s %6s %6s %9s %7s %8s %8s %9s %8s %7s" % hdr)
    for e in sorted(out, key=lambda e: -(e["rate_win_s"] or 0)):
        name = "%s/%s %s K%d" % (e["capture"][:4], e["owner"],
                                 "x".join(map(str, e["act_shape"])), e["K"])
        print("%-46s %2d %7.0f %7.4f %6.2f %6d %9s %7s %8s %8.4f %9s %8s %7s" %
              (name[:46], e["n"], e["linear_calls"], e["fold_s"], e["r_now"], e["N_fused"],
               ("%.2f" % e["r_new"]) if e["r_new"] else "-",
               ("%.4f" % e["s_new"]) if e["s_new"] else "-",
               ("%.4f" % e["rate_win_s"]) if e["rate_win_s"] is not None else "-",
               e["byte_win_s"],
               ("%.4f" % e["slice_tax_s"]) if e["slice_tax_s"] is not None else "L1",
               ("%.4f" % e["net_s"]) if e["net_s"] is not None else "-",
               ("%.1f" % (e["net_s"] * CLOCK_MHZ)) if e["net_s"] is not None else "-"))
        print("        act %s, %s; rate source: %s" % (e["act_mem"], e["act_dtype"], e["how"]))
    tot = sum(e["fold_s"] for e in out)
    calls = sum(e["linear_calls"] for e in out)
    rw = sum(e["rate_win_s"] or 0 for e in out)
    bw = sum(e["byte_win_s"] for e in out)
    st = sum(e["slice_tax_s"] or 0 for e in out)
    rw_dram = sum((e["rate_win_s"] or 0) for e in out if e["act_mem"] == "DRAM")
    # residual: keys with calls in no sibling set
    inset = defaultdict(float)
    for e in out:
        for m in e["members"]:
            inset[m["key"]] += e["linear_calls"] / e["n"]
    print("\n== COVERAGE residual: calls in NO sibling set ==")
    res_c = res_s = 0.0
    for k, c in sorted(cen.items(), key=lambda kv: -kv[1]["fold_s"]):
        left = c["calls"] - inset.get(k, 0.0)
        if left <= 0.5:
            continue
        s_left = left * c["us_per_call"] * 1e-6
        res_c += left
        res_s += s_left
        print("  %-42s %8.0f calls  %7.4f s  %8.1f Mc" % (k, left, s_left, s_left * CLOCK_MHZ))
    print("  residual %.0f calls, %.4f s (%.1f %% of the class)" %
          (res_c, res_s, 100 * res_s / 4.6604))
    print("\nCOVERAGE: %d sets, %.0f of 108608 calls (%.1f %%), %.4f of 4.6604 s (%.1f %%)"
          % (len(out), calls, 100 * calls / 108608, tot, 100 * tot / 4.6604))
    print("rate route %.4f s (%.1f Mc) before tax, of which DRAM-resident %.4f s; slice tax on "
          "the DRAM sets %.4f s, net %.4f s" % (rw, rw * CLOCK_MHZ, rw_dram, st, rw_dram - st))
    print("byte-only floor, DRAM sets only: %.4f s (%.1f Mc)" % (bw, bw * CLOCK_MHZ))
    xf = sites.get("cross_frame_sets", [])
    print("\n== CROSS-FRAME sets, priced the same way ==")
    xtot = 0.0
    for e in xf:
        b = 1
        for d in e["act_shape"][:-2]:
            b *= d
        M, K = e["act_shape"][-2], e["K"]
        flop = sum((e["calls"] / e["n"]) * cen[m["key"]]["TFLOP"] * 1e12 / cen[m["key"]]["calls"]
                   for m in e["members"])
        r_new, how = rate_at(fam, (b, M, K), e["N_fused"])
        s_new = flop / (r_new * 1e12) if r_new else None
        win = (e["fold_s"] - s_new) if s_new else None
        tax = e["calls"] * max(LAUNCH_US * 1e-6,
                               (e["reread_B"] / max(e["n"] - 1, 1) / e["calls"]) / (DRAM_GBs * 1e9))
        print("  %-40s n=%-3d %7.0f calls  now %7.4f s  pred %s  win %s  tax %.4f  net %s"
              % (",".join(e["owners"])[:40], e["n"], e["calls"], e["fold_s"],
                 ("%.4f" % s_new) if s_new else "-", ("%.4f" % win) if win else "-", tax,
                 ("%.4f" % (win - tax)) if win else "-"))
        print("        rate source: %s; act %s" % (how, e["act_mem"]))
        xtot += (win or 0)
    print("  cross-frame rate route %.4f s (%.1f Mc) before tax" % (xtot, xtot * CLOCK_MHZ))
    xc = sum(e["calls"] for e in xf)
    xs = sum(e["fold_s"] for e in xf)
    print("  COVERAGE with cross-frame: %.0f of 108608 calls (%.1f %%), %.4f of 4.6604 s "
          "(%.1f %%); with no sibling at all %.0f calls, %.4f s"
          % (calls + xc, 100 * (calls + xc) / 108608, tot + xs, 100 * (tot + xs) / 4.6604,
             res_c - xc, res_s - xs))

    # The one variant worth naming: ConditionedTransitionBlock's swish and gate halves come from
    # ONE checkpoint tensor that tenstorrent.py:9474 splits with torch.chunk, so re-fusing just
    # those two lands on a MEASURED key (N=3072 in the same family) and needs no new weight.
    ctb = [e for e in out if e["owner"] == "ConditionedTransitionBlock"
           and e["act_shape"] == [1, 512, 768]][0]
    flop2 = ctb["flop"] * 2 / 3
    r_now2 = cen["linear|out=1x512x1536|K=768"]["TFLOPs"]
    r_new2 = cen["linear|out=1x512x3072|K=768"]["TFLOPs"]
    inst = ctb["linear_calls"] / ctb["n"]
    print("\n== VARIANT: CTB swish+gate only, N=1536+1536 -> 3072, both rates MEASURED ==")
    print("  %.4f TFLOP  %.2f -> %.2f TFLOP/s  %.4f -> %.4f s  win %.4f s (%.1f Mc)"
          % (flop2 / 1e12, r_now2, r_new2, flop2 / (r_now2 * 1e12), flop2 / (r_new2 * 1e12),
             flop2 / (r_now2 * 1e12) - flop2 / (r_new2 * 1e12),
             (flop2 / (r_now2 * 1e12) - flop2 / (r_new2 * 1e12)) * CLOCK_MHZ))
    print("  slice tax with ttnn slices: %.4f s; with a gated consumer kernel (precedent "
          "reader_reblock_permute_gated.cpp): 0" % (2 * inst * LAUNCH_US * 1e-6))

    json.dump(out, open(HERE / "priced_512.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
