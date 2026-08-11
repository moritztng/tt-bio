#!/usr/bin/env python3
"""Does the stop-rule floor's panel equal the 256->512 gain's panel, per model?

The sanity gate (check_curve_sanity_n512.py) hard-fails when
`within_fold_common.n_targets != gain_ci["256->512"].common_targets`, because comparing a
gain to a floor computed on a different target set is a panel confound. Pass 37 asserted
the equality three times on opendde-abag and pre-registered it, but the equality is not a
theorem: the two sets are built by different rules.

    floor panel   targets whose LARGEST pool has >= D samples, D = deepest MGRID rung
                  with >= 20 such targets                       (deep_stats, cset)
    gain  panel   targets with a pool key at BOTH rung 256 and rung 512   (main, `both`)

A target with a 512 pool but no 256-rung key -- protenix-v2's 9d73, whose rung-256 chunks
0-2 are a documented cure-resistant exclusion -- is in the first and not the second. So is
a target whose 512 rung is short of 512 samples, in reverse. This probe prints both sets
and their symmetric difference, per model, WITHOUT running a bootstrap, so the exec tier
learns the answer in seconds instead of at the last gate with the answer in hand.

Read-only. Costs one pass over labels.json/results.json; no numpy resampling.
"""
import importlib.util, sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "abag_xm_deepn_analysis.py"
_spec = importlib.util.spec_from_file_location("_deepn", _SRC)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

MGRID = (16, 32, 50, 64, 100, 128, 200, 256, 400, 512)
POWER_MIN = 50          # main(): a rung joins the adjacency spine at >= 50 targets
COMMON_MIN = 20         # deep_stats(): a depth qualifies as D at >= 20 targets


def panels(model):
    pools = A.tiera_pools(model) | A.deepn_pools(model) | A.overlay_pools(model) \
        | A.galaxy_pools(model)
    pools.update(A.galaxy64_pools(model))
    import os
    if os.environ.get("DEEPN_N16_ARK") == "1":
        pools.update(A.n16ark_pools(model))

    rung_nt = {}
    for _t, n in pools:
        rung_nt[n] = rung_nt.get(n, 0) + 1
    powered = [n for n, nt in rung_nt.items() if nt >= COMMON_MIN]
    cap = max(powered) if powered else None

    biggest = {}
    for (t, n), d in pools.items():
        if cap is not None and n > cap:
            continue
        if t not in biggest or n > biggest[t][0]:
            biggest[t] = (n, d["pool"])

    depth_ok = [m for m in MGRID
                if sum(1 for _n, p in biggest.values() if len(p) >= m) >= COMMON_MIN]
    D = max(depth_ok) if depth_ok else None
    floor_panel = {t for t in biggest if D is not None and len(biggest[t][1]) >= D}

    pts = A.curve_points(pools)
    ns = sorted(pts)
    ns_powered = [n for n in ns if pts[n]["n_targets"] >= POWER_MIN]
    pair, gain_panel = None, set()
    for lo, hi in zip(ns_powered, ns_powered[1:]):
        if (lo, hi) == (256, 512):
            pair = "256->512"
            gain_panel = {t for t, _n in pools if (t, lo) in pools and (t, hi) in pools}
    return dict(cap=cap, D=D, ns_powered=ns_powered, pair=pair,
                floor_panel=floor_panel, gain_panel=gain_panel,
                rung_nt=rung_nt, biggest=biggest)


def main():
    models = sys.argv[1:] or sorted(A.MODELS)
    bad = 0
    for model in models:
        r = panels(model)
        fp, gp = r["floor_panel"], r["gain_panel"]
        print(f"\n=== {model} ===")
        print(f"  cap={r['cap']}  D={r['D']}  powered spine={r['ns_powered']}")
        print(f"  rung 256 nt={r['rung_nt'].get(256)}  rung 512 nt={r['rung_nt'].get(512)}")
        if r["pair"] is None:
            print("  256->512 is NOT an adjacent powered pair yet -- the gain does not "
                  "exist. Either rung 512 is below 50 targets or a rung interposes.")
            bad += 1
            continue
        print(f"  floor panel {len(fp)}   gain panel {len(gp)}   "
              f"{'EQUAL' if fp == gp else 'MISMATCH -- the sanity gate will FAIL'}")
        if fp != gp:
            bad += 1
            only_f = sorted(fp - gp)
            only_g = sorted(gp - fp)
            if only_f:
                print(f"  in floor, not in gain ({len(only_f)}): {' '.join(only_f)}")
                print("    -> has >= D samples in its largest pool but no rung-256 key "
                      "(a documented rung exclusion), or its 512 pool is keyed elsewhere")
            if only_g:
                print(f"  in gain, not in floor ({len(only_g)}): {' '.join(only_g)}")
                print("    -> keyed at both rungs but its largest pool is short of D "
                      "samples (an incomplete 512 rung: unfolded or unlabelled chunks)")
    print(f"\n{len(models) - bad}/{len(models)} model(s) panel-equal")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
