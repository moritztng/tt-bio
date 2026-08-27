#!/usr/bin/env python3
"""Solve the two-term recycle model for protenix-v1 against protenix-v2, at 512 aa.

    fold(512, n) = F + n * C          C = one trunk cycle, F = everything else

Four arms over-determine it, which is the point: two protenix-v2 arms give C256 and F,
two protenix-v1 arms give C128 and F_v1 independently, and the c_z factor k = C128/C256
is the verdict. F_v1 is NOT fitted to rescue k -- it falls out of v1's own two arms and
is compared to F_v2 as a check on the model itself.

    python3 scripts/protenix_v1_port/decompose_512.py perf/pxv1
"""
import json
import sys
from pathlib import Path


def arm(root: Path, name: str) -> dict:
    d = json.loads((root / name).read_text())
    w = d["warm_times_s"]
    spread = (max(w) - min(w)) / (sum(w) / len(w)) * 100.0
    return {"median": d["warm_median_s"], "min": min(w), "max": max(w), "n": len(w),
            "spread_pct": spread, "cycles": d["recycling_steps"], "model": d["model"],
            "tokens": d["n_tokens"], "msa": d["cold_metrics"]["msa"]}


def main(root):
    root = Path(root)
    arms = {p.name: arm(root, p.name) for p in sorted(root.glob("v[12]_512_*.json"))}
    for n, a in arms.items():
        print(f"{n:28s} {a['model']:11s} {a['cycles']:2d}cyc  median {a['median']:7.3f}s  "
              f"[{a['min']:.3f}, {a['max']:.3f}]  spread {a['spread_pct']:.2f}%  "
              f"msa={a['msa']}")

    def pick(model, cycles, required=True):
        hits = [a for a in arms.values() if a["model"] == model and a["cycles"] == cycles]
        if not hits:
            if required:
                raise SystemExit(f"missing arm: {model} at {cycles} cycles")
            return None, 0
        # median of the per-process medians, so an independent-process replicate counts once
        meds = sorted(h["median"] for h in hits)
        return meds[len(meds) // 2], len(hits)

    v2_10, n_v2_10 = pick("protenix-v2", 10)
    v2_2, n_v2_2 = pick("protenix-v2", 2)
    v1_4, n_v1_4 = pick("protenix-v1", 4)
    v1_2, n_v1_2 = pick("protenix-v1", 2, required=False)

    c256 = (v2_10 - v2_2) / 8.0
    f_v2 = v2_10 - 10 * c256

    print()
    print(f"protenix-v2: 10cyc {v2_10:.3f}s (x{n_v2_10})   2cyc {v2_2:.3f}s (x{n_v2_2})")
    print(f"C256 = {c256:.4f} s/cycle    F_v2 = {f_v2:.3f} s   trunk share "
          f"{10 * c256 / v2_10 * 100:.1f}%")
    print()
    print("--- the registered prediction, scored ---")
    for kk in (0.48, 0.55, 0.75):
        pred = f_v2 + 4 * kk * c256
        print(f"  k={kk:.2f} -> predicted {pred:7.3f}s   ratio {pred / v2_10:.4f}"
              f"   measured-minus-predicted {v1_4 - pred:+7.3f}s")
    print(f"  MEASURED v1@4cyc {v1_4:.3f}s (x{n_v1_4} process)   "
          f"ratio v1/v2 = {v1_4 / v2_10:.4f}")

    print()
    if v1_2 is None:
        # Under-determined: one v1 arm cannot separate k from F_v1. Bound it instead of
        # fitting it. F_v1 <= F_v2 on theory (v1's confidence head and diffusion
        # conditioning scale with c_z, and v1 has no template track at all), and every
        # euro of F_v1 that is below F_v2 is a euro that moves into the trunk term, so
        # the F_v1 = F_v2 substitution is the MINIMUM k consistent with the measurement.
        k_lo = (v1_4 - f_v2) / 4.0 / c256
        print("v1 2cyc arm MISSING -- k is bounded, not measured.")
        print(f"  assuming F_v1 = F_v2 = {f_v2:.3f}s:  C128 = {(v1_4 - f_v2) / 4:.4f} s/cycle"
              f"   k = {k_lo:.4f}  <- LOWER BOUND on k")
        print(f"  F_v1 < F_v2 is the expectation, and it pushes k UP, so the true k is above "
              f"{k_lo:.3f}.")
        print(f"  The verdict does not need k: the prediction was registered as a FOLD TIME "
              f"({f_v2 + 4 * 0.55 * c256:.2f}s central, band "
              f"[{f_v2 + 4 * 0.48 * c256:.2f}, {f_v2 + 4 * 0.75 * c256:.2f}]) and the fold "
              f"was measured at {v1_4:.3f}s.")
        return 0

    c128 = (v1_4 - v1_2) / 2.0
    f_v1 = v1_4 - 4 * c128
    k = c128 / c256
    print(f"protenix-v1:  4cyc {v1_4:.3f}s (x{n_v1_4})   2cyc {v1_2:.3f}s (x{n_v1_2})")
    print(f"C128 = {c128:.4f} s/cycle    F_v1 = {f_v1:.3f} s   trunk share "
          f"{4 * c128 / v1_4 * 100:.1f}%")
    print(f"k = C128/C256 = {k:.4f}          band [0.48, 0.75] -> "
          f"{'IN BAND' if 0.48 <= k <= 0.75 else 'OUT OF BAND'}")
    print(f"F_v1 / F_v2 = {f_v1 / f_v2:.4f}  (the two-term model wants this near 1, a little "
          f"below: v1's confidence head and diffusion conditioning scale with c_z)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "perf/pxv1"))
