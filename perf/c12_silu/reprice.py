#!/usr/bin/env python3
"""Re-price the fused-silu tax per ELEMENT, because the inherited basis had the wrong shape.

c12-kblock-unlock swept `Transition.swiglu` fc1 at a=[1,16,512,128] and priced the fold at
8,960 calls x 0.0887 ms = 0.7947 s. An instrumented 512 aa fold (sites_512.json) says that shape
never runs: the shipped pair-track chunk height is 47 rows, the site fires on five shapes, and
there are 3,762 fused-silu calls in the fold, not 8,960. A per-call price is therefore not a
transferable unit and a per-element one is, because the epilogue is an SFPU pass over the output.

    reprice.py --op perf/c12_silu/op_ab_s1.json --sites perf/c12_silu/sites_512.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def out_elements(a_shape: str, b_shape: str) -> int:
    """Elements of a linear's output: the a dims with the last replaced by b's last."""
    a = [int(d) for d in a_shape.split("x")]
    b = [int(d) for d in b_shape.split("x")]
    assert a[-1] == b[0], f"contraction mismatch {a_shape} x {b_shape}"
    return math.prod(a[:-1]) * b[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", type=Path, required=True)
    ap.add_argument("--sites", type=Path, required=True)
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    op = json.loads(a.op.read_text())
    sites = json.loads(a.sites.read_text())
    d, c = op["derived"], op["counts"]

    # the arm's own output volume, from the same counts block the known-answer control checked
    arm_els = c["silu_elements"]
    tax_ps = d["tax_ms_per_call"] / 1e3 / arm_els * 1e12
    silu_ps = d["silu_ms"] / 1e3 / arm_els * 1e12
    win_ps = tax_ps - silu_ps

    rows = []
    for r in sites["silu_rows"]:
        els = out_elements(r["a_shape"], r["b_shape"])
        rows.append({**{k: r[k] for k in ("site", "a_shape", "b_shape", "calls")},
                     "out_elements_per_call": els, "out_elements_total": els * r["calls"]})
    rows.sort(key=lambda r: -r["out_elements_total"])
    total = sum(r["out_elements_total"] for r in rows)
    calls = sum(r["calls"] for r in rows)

    res = {
        "doc": __doc__,
        "clock_mhz": a.mhz,
        "arm": {"shape_measured": f"{c['tiles_out']} out tiles, {arm_els} elements",
                "tax_ms_per_call": d["tax_ms_per_call"], "silu_ms_per_call": d["silu_ms"],
                "tax_ps_per_element": round(tax_ps, 3),
                "silu_ps_per_element": round(silu_ps, 3),
                "win_ps_per_element": round(win_ps, 3)},
        "fold_512aa": {
            "fused_silu_calls": calls, "out_elements": total,
            "tax_s": round(total * tax_ps / 1e12, 4),
            "win_s": round(total * win_ps / 1e12, 4),
            "win_mcycles": round(total * win_ps / 1e12 * a.mhz, 1),
            "rows": rows},
        "inherited_basis_refuted": {
            "claimed_calls": 8960, "executed_calls": calls,
            "claimed_shape": "1x16x512x128", "executed_dominant_shape": rows[0]["a_shape"],
            "claimed_tax_s": 0.7947, "repriced_tax_s": round(total * tax_ps / 1e12, 4),
            "note": "the two tax figures agree to 1.5 % only because 2,800 calls of 47 rows is "
                    "nearly the same output volume as 8,960 of 16. The per-call price and the "
                    "call count are both wrong; the product happens to survive."},
    }
    print(json.dumps(res["arm"], indent=1))
    print(json.dumps({k: v for k, v in res["fold_512aa"].items() if k != "rows"}, indent=1))
    for r in rows:
        print("  %-22s %-16s x %-12s calls=%5d  els=%14d" % (
            r["site"], r["a_shape"], r["b_shape"], r["calls"], r["out_elements_total"]))
    print(json.dumps(res["inherited_basis_refuted"], indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
