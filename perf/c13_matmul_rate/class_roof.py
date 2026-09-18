#!/usr/bin/env python3
"""The fold's matmul class priced against a PER-SHAPE roofline, from the executed call census.

The campaign's 10.0 s frontier asks the class to reach 35.49 TFLOP/s, and quotes 21.38 TFLOP/s
against a 122.29 TFLOP/s dense cube as "17.5 %, so there is 5.7x of headroom". That comparison
is only valid for shapes whose arithmetic intensity is at or above the crossover. It is not:
the two keys carrying the thin-K mass sit at 101 FLOP/byte against a crossover of 276, so the
dense cube is not their ceiling and the gap to it is not headroom. This script replaces the
single cube denominator with each key's own binding roof and reports what the class could reach
if every kernel were perfect.

Nothing here opens a device or reads a counter. Every byte count is a closed form over the
shape -- in0 once, the weight once, the result once -- so the byte-counter defect class that
derailed four campaigns cannot occur. The census's own recorded `B` field IS defective on
exactly these keys (it reads 0.1311 MB/call on 1x16x512x512 K=128 against a tile-exact
10.6168 MB/call, dropping the 4-D operand) and is deliberately not read.

The parser is the second thing rewritten here. perf/c13_matmul_rate/keys.py assumed the census
key prints in0 before the weight, which is false for 6 of 18 linear keys, and it dropped every
key carrying a bias. Both errors under-attribute FLOPs to the thin-K shapes: they cost key A
more than half its mass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_shapes(field):
    """'1x16x512x128,128x512,512' -> [[1,16,512,128],[128,512],[512]] (empty on any junk)."""
    out = []
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append([int(x) for x in part.split("x")])
        except ValueError:
            return []
    return out


def parse_key(key):
    """-> (batch, M, K, N, has_bias) for a linear/matmul census key, or None.

    Identifies in0 as the operand whose second-to-last dim is the output's M, rather than
    assuming argument order. 1-D operands are biases and are not matmul operands.
    """
    if "|out=" not in key or "|in=" not in key:
        return None
    head, rest = key.split("|out=", 1)
    out_f, in_f = rest.split("|in=", 1)
    out_s = parse_shapes(out_f)
    ins = parse_shapes(in_f)
    if len(out_s) != 1 or not ins:
        return None
    o = out_s[0]
    if len(o) < 2:
        return None
    m, n = o[-2], o[-1]
    batch = 1
    for x in o[:-2]:
        batch *= x
    ops = [s for s in ins if len(s) >= 2]
    has_bias = any(len(s) == 1 for s in ins)
    if len(ops) < 2:
        return None
    in0 = [s for s in ops if s[-2] == m]
    wgt = [s for s in ops if s[-1] == n and s not in in0]
    if not in0 or not wgt:
        # M == N, or a transposed print: fall back to the shared inner dim
        for a in ops:
            for b in ops:
                if a is b:
                    continue
                if a[-1] == b[-2] and b[-1] == n and a[-2] == m:
                    in0, wgt = [a], [b]
        if not in0 or not wgt:
            return None
    k = in0[0][-1]
    if wgt[0][-2] != k:
        return None
    return batch, m, k, n, has_bias


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path,
                    default=Path("perf/roof_launch/op_census_512.json"))
    ap.add_argument("--cube", type=float, required=True,
                    help="in-session dense cube TFLOP/s, the arithmetic roof")
    ap.add_argument("--dram", type=float, required=True,
                    help="in-session DRAM roof GB/s, the traffic roof")
    ap.add_argument("--target", type=float, default=35.49,
                    help="class TFLOP/s the 10.0 s frontier asks for")
    ap.add_argument("--mhz", type=float, default=1350.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    census = json.loads(a.census.read_text())["top_shapes"]
    cube_f, dram_b = a.cube * 1e12, a.dram * 1e9
    crossover = cube_f / dram_b

    merged, unparsed = {}, []
    for key, v in census.items():
        op = key.split("|", 1)[0]
        if op not in ("ttnn.linear", "ttnn.matmul"):
            continue
        p = parse_key(key)
        if p is None:
            unparsed.append((key, v["calls"]))
            continue
        batch, m, k, n, bias = p
        # canonical: the same arithmetic under any argument print order
        sig = (batch, m, k, n)
        e = merged.setdefault(sig, {"calls": 0.0, "bias": False, "keys": []})
        e["calls"] += v["calls"]
        e["bias"] = e["bias"] or bias
        e["keys"].append(key)

    rows = []
    for (batch, m, k, n), e in merged.items():
        calls = e["calls"]
        f_call = 2.0 * batch * m * k * n
        b_call = 2.0 * (batch * m * k + k * n + batch * m * n) + (2.0 * n if e["bias"] else 0.0)
        ai = f_call / b_call
        t_traffic = calls * b_call / dram_b
        t_arith = calls * f_call / cube_f
        rows.append({
            "batch": batch, "m": m, "k": k, "n": n, "calls": calls,
            "tflop": calls * f_call / 1e12, "gb": calls * b_call / 1e9, "ai": ai,
            "t_traffic_s": t_traffic, "t_arith_s": t_arith,
            "t_roof_s": max(t_traffic, t_arith),
            "binding": "traffic" if t_traffic >= t_arith else "arithmetic",
            "shape_roof_tflops": f_call / max(b_call / dram_b, f_call / cube_f) / 1e12,
            "keys": e["keys"],
        })
    rows.sort(key=lambda r: -r["tflop"])

    tot_f = sum(r["tflop"] for r in rows)
    tot_roof = sum(r["t_roof_s"] for r in rows)
    tot_traffic = sum(r["t_traffic_s"] for r in rows)
    tot_arith = sum(r["t_arith_s"] for r in rows)

    print("ROOFS  arithmetic %.2f TFLOP/s, traffic %.1f GB/s, crossover AI %.1f FLOP/byte"
          % (a.cube, a.dram, crossover))
    print("KEYS   %d canonical shapes from %d census keys, %d keys unparsed (%.0f calls)"
          % (len(rows), sum(len(r["keys"]) for r in rows), len(unparsed),
             sum(c for _, c in unparsed)))
    print("\n%9s %7s %26s %8s %7s %9s %9s %9s %s"
          % ("calls", "TFLOP", "b,M,K,N", "AI", "roofTFs", "t_traf s", "t_arith s", "t_roof s",
             "binds"))
    for r in rows:
        print("%9.0f %7.3f %26s %8.1f %7.1f %9.4f %9.4f %9.4f %s"
              % (r["calls"], r["tflop"],
                 "%dx%dx%dx%d" % (r["batch"], r["m"], r["k"], r["n"]),
                 r["ai"], r["shape_roof_tflops"], r["t_traffic_s"], r["t_arith_s"],
                 r["t_roof_s"], r["binding"]))

    print("\nCLASS  %.3f TFLOP over %.0f calls" % (tot_f, sum(r["calls"] for r in rows)))
    print("  if every kernel hit its own binding roof: %.4f s = %.2f TFLOP/s"
          % (tot_roof, tot_f / tot_roof))
    print("    of which traffic-bound %.4f s, arithmetic-bound %.4f s"
          % (sum(r["t_roof_s"] for r in rows if r["binding"] == "traffic"),
             sum(r["t_roof_s"] for r in rows if r["binding"] == "arithmetic")))
    print("  the same FLOPs at the DENSE CUBE rate would be %.4f s = %.2f TFLOP/s "
          "(physically unreachable: it ignores traffic on %d of %d shapes)"
          % (tot_arith, tot_f / tot_arith,
             sum(1 for r in rows if r["binding"] == "traffic"), len(rows)))
    print("  the frontier asks for %.2f TFLOP/s = %.4f s" % (a.target, tot_f / a.target))
    verdict = ("REACHABLE in principle" if tot_f / tot_roof >= a.target
               else "NOT REACHABLE at these shapes on this part")
    print("  %.2f TFLOP/s target vs %.2f TFLOP/s roofline ceiling -> %s"
          % (a.target, tot_f / tot_roof, verdict))
    print("  headroom a perfect kernel could still buy: %.4f s (%.4f s -> %.4f s)"
          % (0.0, tot_roof, tot_roof))
    n_traffic = sum(1 for r in rows if r["binding"] == "traffic")
    print("\nBINDING  %d of %d shapes are traffic-bound, carrying %.1f %% of the class FLOPs"
          % (n_traffic, len(rows),
             100.0 * sum(r["tflop"] for r in rows if r["binding"] == "traffic") / tot_f))
    thin = [r for r in rows if r["ai"] < crossover]
    print("         %d shapes sit below the crossover AI, carrying %.1f %% of class FLOPs "
          "and %.4f s of the %.4f s roofline floor"
          % (len(thin), 100.0 * sum(r["tflop"] for r in thin) / tot_f,
             sum(r["t_roof_s"] for r in thin), tot_roof))
    if unparsed:
        print("\nUNPARSED (excluded from every total above, never silently dropped):")
        for key, c in sorted(unparsed, key=lambda x: -x[1]):
            print("  %8.0f calls  %s" % (c, key))
    if a.out:
        a.out.write_text(json.dumps(
            {"cube_tflops": a.cube, "dram_gbs": a.dram, "crossover_ai": crossover,
             "target_tflops": a.target, "class_tflop": tot_f,
             "class_roof_s": tot_roof, "class_roof_tflops": tot_f / tot_roof,
             "class_arith_only_s": tot_arith, "class_traffic_only_s": tot_traffic,
             "verdict": verdict, "rows": rows,
             "unparsed": [{"key": k, "calls": c} for k, c in unparsed]}, indent=1))
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
