#!/usr/bin/env python3
"""Unit test for `ab_arms.summarize` -- the function that produces the campaign's number.

No device. Runs anywhere, including on the orchestrator's CPU row, which is the point: the
arithmetic that decides GO/NO-GO should not first be exercised on a card at 03:00.

    python3 perf/b2z-integrate/test_summarize.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_ab_arms", HERE / "ab_arms.py")
AB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AB)

ARMS = ["base", "win", "dead", "quiet", "all"]
ORDER_ARMS = ["win", "dead", "quiet", "all"]
BASE_CIF, WIN_CIF, QUIET_CIF, ALL_CIF = "aa" * 32, "bb" * 32, "cc" * 32, "dd" * 32


def r(arm, t, cif):
    return {"arm": arm, "fold_s": t, "cif_sha256": cif, "plddt": 0.85,
            "stages_s": {"sampler": t / 3, "confidence": 0.5},
            "sampler_ms_per_step": 1000 * t / 3 / 200,
            "prepare_and_trunk_s": t / 2, "diffusion_shape": (512, 4096, 4480),
            "warmup": False}


def build(reps=5):
    """base is 20.0 s dead flat; win 18.0 (1.111x); dead is base exactly; quiet changes the
    answer at the same speed; all is 17.0 (1.176x) against a 3.0 s sum of separate savings."""
    warm = []
    for _ in range(reps):
        warm += [r("base", 20.0, BASE_CIF), r("win", 18.0, WIN_CIF)]
        warm += [r("base", 20.0, BASE_CIF), r("dead", 20.0, BASE_CIF)]
        warm += [r("base", 20.0, BASE_CIF), r("quiet", 20.0, QUIET_CIF)]
        warm += [r("base", 20.0, BASE_CIF), r("all", 17.0, ALL_CIF)]
    return warm


def main() -> int:
    summ, floor = AB.summarize(build(), ARMS, ORDER_ARMS, "all")
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    check(floor["fold_AA_ratio"] == 1.0, f"flat base must give a 1.0 A/A floor, got {floor}")
    check(summ["win"]["paired_speedup"] == round(20.0 / 18.0, 5),
          f"win paired ratio wrong: {summ['win']['paired_speedup']}")
    check(summ["win"]["median_speedup"] == summ["win"]["paired_speedup"],
          "on flat data the paired and the median ratio must agree")
    check("flag" not in summ["win"], f"a 1.11x win must not be flagged: {summ['win'].get('flag')}")
    check(summ["win"]["bit_exact_vs_base"] is False, "win writes a different CIF")

    check(summ["dead"]["flag"].startswith("NO-OBSERVABLE-EFFECT"),
          f"an arm identical to base in time AND digest must be flagged: {summ['dead'].get('flag')}")
    check(summ["quiet"]["flag"].startswith("INSIDE-AA-FLOOR"),
          f"same time, different answer is not 'no effect': {summ['quiet'].get('flag')}")

    add = summ["additivity"]
    check(add["parts"] == ["win", "dead", "quiet"], f"parts wrong: {add['parts']}")
    check(add["sum_of_separate_saved_s"] == 2.0, f"2.0 s of separate savings, got {add}")
    check(add["measured_combined_saved_s"] == 3.0, f"3.0 s measured, got {add}")
    check(add["discount"] == 1.5, f"discount must be measured/summed = 1.5, got {add['discount']}")

    # A noisy base must widen the floor and swallow a small win rather than report it.
    noisy = build()
    for i, x in enumerate(noisy):
        if x["arm"] == "base":
            x["fold_s"] = 20.0 + (0.6 if i % 8 in (0, 2) else -0.6)
        if x["arm"] == "win":
            x["fold_s"] = 19.7        # 1.015x, well inside a 1.06x floor
    s2, f2 = AB.summarize(noisy, ARMS, ORDER_ARMS, None)
    check(f2["fold_AA_ratio"] > 1.05, f"noisy base must show a wide floor, got {f2}")
    check(s2["win"]["flag"].startswith("INSIDE-AA-FLOOR"),
          f"a 1.015x arm under a 1.06x floor is not a result: {s2['win'].get('flag')}")

    for f in fails:
        print("FAIL:", f)
    print("PASS: summarize" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
