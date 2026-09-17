#!/usr/bin/env python3
"""CPU known-answer controls for the two-clock decomposition. No device, no fold.

Every check has an answer known before it runs. The negative controls at the end break the thing
the checks actually read, not something adjacent to it: a relabelled clock, an item whose slow-arm
reading is inflated, a clock sample set scored against the wrong target, and a region tree with a
hole in it must each be caught by the same code path that produces the table.
"""
from __future__ import annotations
import json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "perf" / "b2x_host_residual"))
import fit
import evidence

PASS, FAIL = [], []


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) and not isinstance(want, bool) \
        else (got == want)
    (PASS if ok else FAIL).append(f"{name}: got {got!r} want {want!r} tol {tol}")
    return ok


def synth(F, C, f):
    return F + C / f


def main() -> int:
    # 1. the gains are the analytic ones and sum to 1, at the row's clocks and at others
    g_hi, g_lo = fit.gains()
    check("gains/hi", g_hi, 2.4545454545, 1e-9)
    check("gains/lo", g_lo, -1.4545454545, 1e-9)
    check("gains/sum", g_hi + g_lo, 1.0, 1e-12)
    for f_hi, f_lo in ((1200.0, 1000.0), (1350.0, 1000.0), (2000.0, 500.0)):
        a, b = fit.gains(f_hi, f_lo)
        check(f"gains/sum@{f_hi:.0f}-{f_lo:.0f}", a + b, 1.0, 1e-12)

    # 2. the fit recovers an exact synthetic (F, C) with zero residual, at both clock pairs
    for F, C in ((3.0, 16000.0), (0.0, 14665.0), (1.95, 10403.4), (-0.4, 900.0)):
        for f_hi, f_lo in ((1350.0, 800.0), (1350.0, 1000.0)):
            Fh, Ch = fit.fixed_and_work(synth(F, C, f_hi), synth(F, C, f_lo), f_hi, f_lo)
            check(f"recover/F({F},{C})@{f_lo:.0f}", Fh, F, 1e-9)
            check(f"recover/C({F},{C})@{f_lo:.0f}", Ch, C, 1e-6)

    # 3. the three pure cases classify without a hand-set threshold
    pure_host = fit.classify([0.500, 0.502, 0.499], [0.501, 0.498, 0.500])
    check("pure-host/verdict", pure_host["verdict"], "CLOCK-IMMUNE")
    check("pure-host/F", pure_host["F_s"], 0.500, 0.01)
    check("pure-host/C", pure_host["C_Mcycles"], 0.0, 10.0)

    t_hi = 4.0
    dev = [t_hi * 1350.0 / 800.0] * 3
    pure_dev = fit.classify([t_hi] * 3, dev)
    check("pure-device/verdict", pure_dev["verdict"], "CLOCK-SCALED")
    check("pure-device/F", pure_dev["F_s"], 0.0, 1e-9)
    check("pure-device/C", pure_dev["C_Mcycles"], t_hi * 1350.0, 1e-6)

    mixed = fit.classify([2.0, 2.01, 1.99], [2.0 + 1.0 * (1350.0 / 800.0 - 1.0)] * 3)
    check("mixed/verdict", mixed["verdict"], "MIXED")
    check("mixed/F", mixed["F_s"], 1.0, 0.05)

    tiny = fit.classify([0.0005, 0.0009, 0.0002], [0.0007, 0.0003, 0.0011])
    check("tiny/verdict", tiny["verdict"], "UNRESOLVED")

    # 4. closure is an identity, not a fit: a partition's F_i sum to the whole's F exactly
    rng = random.Random(0)
    parts = [(rng.uniform(0, 2), rng.uniform(0, 4000)) for _ in range(12)]
    hi = sum(synth(F, C, 1350.0) for F, C in parts)
    lo = sum(synth(F, C, 800.0) for F, C in parts)
    per = {f"i{k}": fit.fixed_and_work(synth(F, C, 1350.0), synth(F, C, 800.0))[0]
           for k, (F, C) in enumerate(parts)}
    whole = fit.fixed_and_work(hi, lo)[0]
    c = fit.closure(per, whole, whole)
    check("closure/identity", c["gap_vs_fold_s"], 0.0, 1e-9)
    check("closure/sum", c["item_F_sum_s"], sum(F for F, _ in parts), 1e-9)

    # 5. scaling reads a ratio and does not invent one from a zero
    sc = fit.scaling({"a": 3.983, "b": 0.0}, {"a": 1.950, "b": 0.0})
    check("scaling/ratio", sc["a"]["ratio"], 2.0425, 1e-3)
    check("scaling/zero-is-none", sc["b"]["ratio"], None)

    # 6. clock coverage is scored against the fold's OWN target, in both directions
    def samples(mhz, n=50, t0=0, step=1_000_000, err_at=None, move_at=None, move_to=None):
        rows = []
        for i in range(n):
            r = {"read_start_ns": t0 + i * step, "read_end_ns": t0 + i * step + 50_000}
            if err_at is not None and i == err_at:
                r["error"] = "OSError"
            else:
                r["MHz"] = move_to if (move_at is not None and i >= move_at) else mhz
                r["W"] = 60.0
            rows.append(r)
        return rows

    iv = {"start_monotonic_ns": 0, "end_monotonic_ns": 50 * 1_000_000}
    check("cov/800@800", evidence.coverage(samples(800), iv, 800)["pass"], True)
    check("cov/800@1350", evidence.coverage(samples(800), iv, 1350)["pass"], False)
    check("cov/1350@1350", evidence.coverage(samples(1350), iv, 1350)["pass"], True)
    check("cov/1350@800", evidence.coverage(samples(1350), iv, 800)["pass"], False)
    check("cov/mid-fold-move", evidence.coverage(samples(1350, move_at=25, move_to=1200), iv,
                                                 1350)["pass"], False)
    check("cov/read-error", evidence.coverage(samples(1350, err_at=10), iv, 1350)["pass"], False)
    check("cov/thin", evidence.coverage(samples(1350, n=2), iv, 1350)["pass"], False)
    gap = samples(1350, n=6, step=12_000_000)
    check("cov/gap", evidence.coverage(gap, {"start_monotonic_ns": 0,
                                             "end_monotonic_ns": 6 * 12_000_000}, 1350)["pass"],
          False)

    # 7. the region tree partitions its parent: exclusive sums and a measured leftover
    import host_residual as HR
    reg = HR.Regions()

    class Fake:
        def outer(self):
            time.sleep(0.05)
            self.inner()
            self.inner()

        def inner(self):
            time.sleep(0.02)

    obj = Fake()
    reg.patch(Fake, "outer", "outer")
    reg.patch(Fake, "inner", "inner")
    t0 = time.perf_counter()
    obj.outer()
    wall = time.perf_counter() - t0
    reg.remove()
    tbl = reg.table()
    check("regions/outer-calls", tbl["outer"]["calls"], 1)
    check("regions/inner-calls", tbl["outer/inner"]["calls"], 2)
    check("regions/outer-excl", tbl["outer"]["excl_s"], 0.05, 0.02)
    check("regions/inner-incl", tbl["outer/inner"]["incl_s"], 0.04, 0.02)
    excl = sum(v["excl_s"] for v in tbl.values())
    # `Regions.table` rounds every column to 5 dp, so the partition identity holds to the sum
    # of those roundings and not to machine epsilon. Three rows, so 2e-5 is the honest tolerance.
    check("regions/exclusive-partition", excl, tbl["outer"]["incl_s"], 2e-5)
    check("regions/unattributed", wall - tbl["outer"]["incl_s"], 0.0, 0.01)
    check("regions/unpatched", callable(Fake.outer) and Fake.outer.__name__, "outer")

    # --- negative controls: break what the checks read -----------------------------------
    # (a) a relabelled clock. Feeding the arms in the wrong order turns a host item into a
    #     negative fixed term, which is impossible, so it cannot pass unnoticed.
    # A swap does not make F negative -- for a device item it makes F larger than the item's
    # own fast-clock seconds, which is the impossible direction. The bound 0 <= F <= t_hi catches
    # both, so it is the invariant the fit enforces and the one tested here.
    swapped = fit.classify(dev, [t_hi] * 3)
    check("neg/relabelled-clock-verdict", swapped["verdict"], "IMPOSSIBLE")
    check("neg/relabelled-clock-exceeds-fast-arm", swapped["F_s"] > swapped["t_hi_s"], True)
    slow_arm_faster = fit.classify([2.0] * 3, [1.0] * 3)
    check("neg/slow-arm-faster", slow_arm_faster["verdict"], "IMPOSSIBLE")
    check("pure-host/physical", pure_host["physical"], True)
    check("pure-device/physical", pure_dev["physical"], True)
    check("mixed/physical", mixed["physical"], True)
    # (b) one item's slow arm inflated by 40 %: its verdict must leave CLOCK-IMMUNE and the
    #     closure gap must move by the fitted amount, not stay put.
    good = fit.classify([0.500] * 3, [0.500] * 3)
    bad = fit.classify([0.500] * 3, [0.700] * 3)
    check("neg/inflated-verdict-moves", (good["verdict"], bad["verdict"]),
          ("CLOCK-IMMUNE", "MIXED"))
    check("neg/inflated-F-moves", bad["F_s"] - good["F_s"], -0.29090909, 1e-6)
    # (c) a region tree with a hole: the leftover row must grow by exactly the hole.
    holed = {k: v for k, v in tbl.items() if k != "outer/inner"}
    check("neg/hole-shows-up",
          sum(v["excl_s"] for v in holed.values()) < excl - 0.03, True)
    # (d) losing one clock arm entirely must produce no fixed term at all, not a fake zero.
    try:
        fit.classify([1.0, 1.0], [])
        check("neg/one-clock-refused", False, True)
    except Exception:
        check("neg/one-clock-refused", True, True)

    log = HERE / "controls.txt"   # not .log: the repo gitignores *.log and a committed control artifact must be committable (c10-qchunk lesson)
    log.write_text("\n".join(["PASS " + x for x in PASS] + ["FAIL " + x for x in FAIL]) + "\n")
    print(f"{len(PASS)} pass, {len(FAIL)} fail -> {log}")
    for x in FAIL:
        print("  FAIL " + x)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
