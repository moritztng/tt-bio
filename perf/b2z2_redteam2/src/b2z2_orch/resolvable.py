#!/usr/bin/env python3
"""How big does a Boltz-2 512 aa fold lever have to be before the cell can see it?

Every fold-level lever this campaign has shipped has dissolved into the fold's noise, and each time
the response has been to argue about the ratio. The prior question is what the instrument can
resolve at all, and it is answerable from data already committed: `b2z-perfpage-recell` ran 28 folds
on qb2 card 0 under benchlock at loadavg 1.92, interleaved base/ship, and recorded every rep.

Two floors come out of it and they are different numbers:

  UNPAIRED  what you can see comparing two arms' medians, which is what most of this campaign did.
  PAIRED    what you can see differencing rep i of one arm against rep i of the other in the same
            process. The interleaved design earns this and it is several times tighter.

Run:  python3 perf/b2z2_orch/resolvable.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "perf" / "b2z-perfpage" / "recell_512_qb2c0.json"

# Two-sided t, 95 %, by degrees of freedom. Table rather than scipy: this repo does not depend on
# scipy and a power estimate does not justify adding one.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145, 16: 2.120, 19: 2.093,
       24: 2.064, 29: 2.045, 39: 2.023, 49: 2.010}


def t95(df: int) -> float:
    if df <= 0:
        return float("inf")
    ks = sorted(T95)
    for k in ks:
        if df <= k:
            return T95[k]
    return 1.96


def published_median(arm: str) -> float:
    """What the committed file reports, which is NOT the warm median -- see main()."""
    return float(json.loads(SRC.read_text())["median_512_s"][arm])


def load() -> tuple[list[float], list[float]]:
    runs = json.loads(SRC.read_text())["runs"]
    # Filter on TARGET as well as warmup. The harness interleaves cdk2x2_298 into the same file,
    # and a 298 aa fold is roughly a third of a 512 aa one, so mixing them reports a ~50 % "spread"
    # that is two populations rather than one instrument's noise.
    warm = [r for r in runs if not r.get("warmup") and r["target"] == "cdk2x2_512"]
    base = [r["fold_s"] for r in warm if r["arm"] == "base"]
    ship = [r["fold_s"] for r in warm if r["arm"] == "ship"]
    return base, ship


def mde(sd: float, n: int, paired: bool) -> float:
    """Minimum detectable difference in seconds at 95 % confidence, n reps per arm."""
    if paired:
        return t95(n - 1) * sd / math.sqrt(n)
    return t95(2 * n - 2) * sd * math.sqrt(2.0 / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    base, ship = load()
    n = min(len(base), len(ship))
    pairs = [b - s for b, s in zip(base[:n], ship[:n])]
    cell = st.median(ship)

    sd_un = st.stdev(base + ship)
    sd_pair = st.stdev(pairs)
    wins = sum(1 for d in pairs if d > 0)

    print(f"source  {SRC.relative_to(ROOT)}  ({n} warm pairs, qb2 card 0, benchlock, loadavg 1.92)")
    print(f"base    median {st.median(base):.3f} s   spread {(max(base)-min(base))/st.median(base)*100:.2f} %")
    print(f"ship    median {cell:.3f} s   spread {(max(ship)-min(ship))/cell*100:.2f} %")
    print(f"paired  mean diff {st.mean(pairs):+.3f} s, sd {sd_pair:.3f} s, ship faster in {wins}/{n}")
    print(f"        warm-only ratio {st.median(base)/cell:.5f}x")
    pb, ps = published_median("base"), published_median("ship")
    print(f"\nNOTE: the committed median_512_s is {ps:.3f} s, not the warm median {cell:.3f} s.")
    wb = [r["fold_s"] for r in json.loads(SRC.read_text())["runs"]
          if r.get("warmup") and r["target"] == "cdk2x2_512"]
    print(f"      It is the median of SEVEN folds per arm including the warmup rep "
          f"({' / '.join(f'{v:.3f}' for v in wb)} s), which the campaign protocol discards.")
    print(f"      Warm-only the arms are {st.median(base):.3f} / {cell:.3f} s")
    print(f"      = {st.median(base)/cell:.5f}x against the committed {pb/ps:.5f}x. The median is")
    print(f"      robust so the published cell moves only {abs(ps-cell):.3f} s ({abs(ps-cell)/ps*100:.2f} %),")
    print( "      which changes no conclusion -- but the next re-cell should drop the warmup rep.\n")
    print("Minimum detectable fold lever at 95 % confidence, as a ratio on the published cell:\n")
    print(f"  {'reps/arm':>9}  {'unpaired':>18}  {'paired, interleaved':>22}")
    rows = {}
    for k in (3, 5, 6, 10, 15, 20, 30):
        u, p = mde(sd_un, k, False), mde(sd_pair, k, True)
        rows[k] = {"unpaired_s": u, "unpaired_x": cell / (cell - u),
                   "paired_s": p, "paired_x": cell / (cell - p)}
        print(f"  {k:>9}  {u:>7.3f} s = {cell/(cell-u):.4f}x  {p:>9.3f} s = {cell/(cell-p):.4f}x")

    print("\nWhat that means for the levers on the table:")
    for name, sec in (("shipped three (measured 0.077 s)", 0.077),
                      ("unfused silu (measured 0.449 s)", 0.449),
                      ("HOST, if 1.06704x is real", cell - cell / 1.06704),
                      ("atom key window, BH projection", 0.366)):
        need_p = next((k for k in (3, 5, 6, 10, 15, 20, 30) if rows[k]["paired_s"] <= sec), None)
        need_u = next((k for k in (3, 5, 6, 10, 15, 20, 30) if rows[k]["unpaired_s"] <= sec), None)
        pp = f"{need_p} paired reps" if need_p else "NOT resolvable at 30 paired reps"
        uu = f"{need_u} unpaired" if need_u else "not resolvable at 30 unpaired"
        print(f"  {sec:5.3f} s  {name:<34} {pp}  ({uu})")

    if a.json:
        a.json.write_text(json.dumps(
            {"cell_s": cell, "n_pairs": n, "sd_paired_s": sd_pair, "sd_unpaired_s": sd_un,
             "paired_mean_diff_s": st.mean(pairs), "ship_faster_pairs": wins,
             "mde": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
