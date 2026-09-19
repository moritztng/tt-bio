#!/usr/bin/env python3
"""Price the whole derived-config surface: measured marginal x measured call count.

Two things this does that a ratio table cannot. It reports the ABSOLUTE second, because every
ratio inherits a denominator and the campaign has had two stale walls in circulation
(BULLETIN-01 §15); and it converts standalone marginals to in-fold seconds with the measured
in-fold factor rather than assuming they are the same number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# trix-scaffold-attribute measured 1.1919 ms/call in-fold against a 1.4490 ms standalone base on
# the same tree and the same card: in-fold work costs 0.8226 of its standalone marginal. Applied
# here as the campaign's own conversion rather than assuming standalone == in-fold.
IN_FOLD_FACTOR = 0.8226
FOLD_S = 48.62  # trix-scaffold-attribute, 512 aa protenix-v2 cdk2x2_512, qb2 card 1, 1350 MHz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, default=HERE / "census512.json")
    ap.add_argument("--ladders", nargs="+", type=Path,
                    default=[HERE / "ladder_sw1.json", HERE / "ladder_fold.json",
                             HERE / "ladder_ab.json"])
    ap.add_argument("--out", type=Path, default=HERE / "priced.json")
    a = ap.parse_args()

    cen = json.loads(a.census.read_text())
    counts: dict[tuple, int] = {}
    for r in cen["rows"]:
        if r["cfg"] != "derived":
            continue
        d0 = [int(x) for x in r["in0"].split("|")[0].split("x")]
        d1 = [int(x) for x in r["in1"].split("|")[0].split("x")]
        b = 1
        for x in d0[:-2]:
            b *= x
        counts[(b, d0[-2], d0[-1], d1[-1])] = counts.get((b, d0[-2], d0[-1], d1[-1]), 0) + r["n"]

    shapes, seen = [], set()
    for p in a.ladders:
        if not p.is_file():
            continue
        for s in json.loads(p.read_text())["shapes"]:
            if s["shape"] in seen:
                continue
            seen.add(s["shape"])
            shapes.append(s)

    print(f"fold {cen['fold_s']} s census / {FOLD_S} s reference, in-fold factor {IN_FOLD_FACTOR}")
    print(f"{'shape':5s} {'calls':>7s} {'derived':>9s} {'best obh':>9s} {'x':>7s} "
          f"{'best ibw':>9s} {'x':>7s} {'A/A':>6s}  {'obh s':>7s} {'ibw s':>7s}")
    tot_obh = tot_ibw = 0.0
    rows = []
    for s in shapes:
        n = counts.get((s["b"], s["M"], s["K"], s["N"]), 0)
        by = {r["tag"]: r for r in s["rows"]}
        base = by["derived"]["ms"]
        obh = min((r for r in s["rows"] if r["tag"].startswith("obh")),
                  key=lambda r: r["ms"], default=None)
        ibw = min((r for r in s["rows"] if r["tag"].startswith("ibw")),
                  key=lambda r: r["ms"], default=None)
        aa = s["aa_pct"]
        # a gain inside the A/A floor is not a gain
        g_obh = max(0.0, base - obh["ms"]) if obh and 100 * (base - obh["ms"]) / base > aa else 0.0
        g_ibw = max(0.0, base - ibw["ms"]) if ibw and 100 * (base - ibw["ms"]) / base > aa else 0.0
        s_obh, s_ibw = n * g_obh / 1e3 * IN_FOLD_FACTOR, n * g_ibw / 1e3 * IN_FOLD_FACTOR
        tot_obh += s_obh
        tot_ibw += s_ibw
        print(f"{s['shape']:5s} {n:7d} {base:9.5f} "
              f"{(obh['tag'] if obh else '-'):>9s} {(base / obh['ms'] if obh else 1):7.4f} "
              f"{(ibw['tag'] if ibw else '-'):>9s} {(base / ibw['ms'] if ibw else 1):7.4f} "
              f"{aa:5.2f}%  {s_obh:7.4f} {s_ibw:7.4f}")
        rows.append({"shape": s["shape"], "calls": n, "derived_ms": base,
                     "best_obh": obh["tag"] if obh else None,
                     "obh_x": round(base / obh["ms"], 4) if obh else 1.0,
                     "best_ibw": ibw["tag"] if ibw else None,
                     "ibw_x": round(base / ibw["ms"], 4) if ibw else 1.0,
                     "aa_pct": aa, "infold_s_obh": round(s_obh, 4),
                     "infold_s_ibw": round(s_ibw, 4)})
    print(f"\nout_block_h over the whole surface: {tot_obh:.4f} s in-fold "
          f"= {FOLD_S / (FOLD_S - tot_obh):.5f}x on a {FOLD_S} s fold")
    print(f"in0_block_w over the whole surface: {tot_ibw:.4f} s in-fold "
          f"= {FOLD_S / (FOLD_S - tot_ibw):.5f}x")
    print(f"both, as an upper bound that assumes they do not interact: "
          f"{tot_obh + tot_ibw:.4f} s = {FOLD_S / (FOLD_S - tot_obh - tot_ibw):.5f}x")
    a.out.write_text(json.dumps({"in_fold_factor": IN_FOLD_FACTOR, "fold_s": FOLD_S,
                                 "rows": rows, "total_infold_s_obh": round(tot_obh, 4),
                                 "total_infold_s_ibw": round(tot_ibw, 4)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
