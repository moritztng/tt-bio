#!/usr/bin/env python3
"""How much of the fold routes to ttnn's 1D matmul factory, where the 2D factory is untested.

`c12-kblock-unlock` found the mechanism and cited it at source: `create_matmul_program_config`
picks between the 1D systolic and 2D factories on an output height/width ratio of 8 and nothing
else (`matmul_program_config.cpp:28`), then sets the K block inside the chosen factory. It
validated the rule on real keys -- the pair keys at N=512 have ratio 16.0 and get 1D, the SAME
operands at N=1024 have ratio 8.0 and get 2D -- and measured 1.2592x on one ratio-16 key by
crossing to 2D, 1.2434x of it surviving into the production path.

A sweep confined to the derived family cannot see the better factory, and every C12 sweep before
kblock's was so confined. So: which keys route 1D, and what are they worth?

ratio = Mt_total / Nt, Mt_total = prod(all dims but the last) * M / 32, Nt = N / 32.
ratio > 8 -> 1D.

THE EXTRAPOLATION IS THE HARD PART and this script gives two, because the optimistic one is wrong:
  * uniform: apply kblock's best single ratio to every 1D second. Overstates badly -- its own four
    keys came in at 1.2434, 1.1852, 1.0627 and 0.9935, the last inside its A/A floor.
  * realized: kblock recovered +0.2052 s from keys carrying 1.9449 s of census seconds, a 10.55 %
    recovery rate on the production path. Applying that to the 1D seconds it has NOT touched is
    the defensible estimate.
Both sit on census seconds, which passes 6/9/13 showed overstate L1-resident keys by ~2x, so the
in-fold prize is plausibly half of either. `c12-profiled-fold` is what settles the base.
"""
import json
import subprocess

CENSUS = "origin/wk/c10-fold-census:perf/c10_fold_census/runs/sweep2/replay.json"
KBLOCK_GAIN_S = 0.2052        # production-path A/B, c12-kblock-unlock
KBLOCK_BASE_S = 1.9449        # census seconds of the four keys it priced
BEST_RATIO = 1.2592           # its best single cross-family ratio


def main():
    d = json.loads(subprocess.run(["git", "show", CENSUS], capture_output=True,
                                  check=True, text=True).stdout)
    rows = [r for r in d["rows"]
            if r["arm"] in ("linear", "matmul") and not str(r["label"]).endswith("@grid110")]
    tot = {"1D": 0.0, "2D": 0.0}
    keys = []
    for r in rows:
        dims = list(r["out"])
        if len(dims) < 2:
            continue
        lead = 1
        for x in dims[:-2]:
            lead *= x
        mt, nt = lead * dims[-2] / 32.0, dims[-1] / 32.0
        if not nt:
            continue
        q = mt / nt
        s = (r.get("s_per_call_qualified") or 0) * (r["calls"] or 0)
        fac = "1D" if q > 8 else "2D"
        tot[fac] += s
        keys.append((s, r["arm"], "x".join(map(str, dims)), r["K"], mt, nt, q, fac))

    print(f"{'key':46}{'Mt_tot':>8}{'Nt':>6}{'ratio':>9}{'fac':>5}{'census s':>10}")
    for s, a, o, K, mt, nt, q, fac in sorted(keys, key=lambda x: -x[0])[:12]:
        print(f"{a+'|'+o+'|K='+str(K):46}{mt:8.0f}{nt:6.0f}{q:9.2f}{fac:>5}{s:10.4f}")

    one, two = tot["1D"], tot["2D"]
    rate = KBLOCK_GAIN_S / KBLOCK_BASE_S
    untouched = one - KBLOCK_BASE_S
    print(f"\nrouted 1D (ratio > 8), 2D untested : {one:8.4f} s of census seconds")
    print(f"routed 2D (ratio <= 8)             : {two:8.4f} s")
    print(f"\nkblock already priced {KBLOCK_BASE_S:.4f} s of the 1D set and recovered "
          f"{KBLOCK_GAIN_S:.4f} s -> {100*rate:.2f} % on the production path")
    print(f"1D seconds it has NOT touched      : {untouched:8.4f} s")
    print(f"  uniform at its best {BEST_RATIO}x   : {untouched*(1-1/BEST_RATIO):8.4f} s  "
          f"<- OVERSTATES, its own keys spanned 1.2434 to 0.9935")
    print(f"  at its realized {100*rate:.2f} % rate    : {untouched*rate:8.4f} s  <- the defensible one")
    print(f"\nHalve either for the ~2x census overprice on L1-resident keys; c12-profiled-fold")
    print(f"settles the base. Very high ratios (2048, 16384) are NOT the regime kblock measured at")
    print(f"ratio 16 and must not inherit its number -- a very tall thin output may belong in 1D.")


if __name__ == "__main__":
    main()
