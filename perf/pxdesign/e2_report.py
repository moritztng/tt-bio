"""Price E2's legs the way pass 10 priced its own, so the two are comparable line for line.

E2 asks one question: does the SCOPED lever (AF2's own kwarg) deliver what pass 10 measured with
the process-wide env var? So the arithmetic here is deliberately pass 10's -- the stage cells, the
H200 denominator and the bar all come from `p10_report` by import rather than being restated, and
the only thing that differs is which artifacts are read and how the arm is named.

    PYTHONPATH=. python3 perf/pxdesign/e2_report.py

The four checks are the ones stated in advance in the state doc, and they are printed as PASS/FAIL
rather than left for a reader to eyeball:

  1. A/A within 1.0 %            -- the window was clean enough to believe the effect
  2. effect >= 1.25x             -- and at least 20x the A/A, so it is not drift
  3. ONE structure_sha16, and it must be cd80f8e274306706 -- the lever changes the shard's extent,
     not the arithmetic, so the design coordinates must be the SAME ones pass 10 committed
  4. l1_padded_diverged > 0 on the lever arm -- proof the arm was not a no-op. Without this a leg
     that silently ran the incumbent twice reads as a clean A/A and an honest-looking 1.00x.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PERF = ROOT / "perf/pxdesign"

from perf.pxdesign.p10_report import (  # noqa: E402
    DESIGNS, H200, H200_TOTAL, stage_filter, stage_generator,
)

#: Pass 10's committed design coordinates. The lever must not move them.
ANCHOR_SHA16 = "cd80f8e274306706"
P10_INCUMBENT, P10_LEVER = 161.891, 122.980


def legs() -> list[dict]:
    out = []
    for path in sorted(PERF.glob("tt_pxd_e2_fold_848_leg*.json")):
        d = json.loads(path.read_text())
        d["_file"] = path.name
        out.append(d)
    return out


def main() -> int:
    ls = legs()
    if not ls:
        print("no E2 fold artifacts yet", file=sys.stderr)
        return 1

    print("%-32s %4s %9s %8s %9s %9s  digest" % (
        "leg", "arm", "fold_s", "pass_s", "stacks_s", "diverged"))
    for d in ls:
        print("%-32s %4s %9.3f %8.4f %9.4f %9s  %s" % (
            d["label"], d.get("l1_padded_plan", "?"), d["fold_s_warm_median"],
            d["pass_s_warm_median"], d["split_warm_mean_s"]["device_stacks_s"],
            d.get("l1_padded_diverged", "-"),
            ",".join(d.get("structure_sha16_all", ["-"]))))

    inc = [d for d in ls if d.get("l1_padded_plan") == "off"]
    lev = [d for d in ls if d.get("l1_padded_plan") == "on"]
    if not inc or not lev:
        print("need both arms", file=sys.stderr)
        return 1

    inc_s = [d["fold_s_warm_median"] for d in inc]
    lev_s = [d["fold_s_warm_median"] for d in lev]
    inc_mean, lev_mean = sum(inc_s) / len(inc_s), sum(lev_s) / len(lev_s)
    aa = 100 * (max(inc_s) - min(inc_s)) / inc_mean if len(inc_s) > 1 else float("nan")
    effect = inc_mean / lev_mean
    print("\nincumbent %.3f s a design (A/A %.3f %%), lever %.3f s, %.4fx, delta %.3f s"
          % (inc_mean, aa, lev_mean, effect, inc_mean - lev_mean))
    print("pass 10 measured incumbent %.3f / lever %.3f (%.4fx) with the env var"
          % (P10_INCUMBENT, P10_LEVER, P10_INCUMBENT / P10_LEVER))

    digests = sorted({x for d in ls for x in d.get("structure_sha16_all", ())})
    diverged = [d.get("l1_padded_diverged", 0) for d in lev]

    checks = [
        ("A/A within 1.0 %", aa < 1.0, "%.3f %%" % aa),
        ("effect >= 1.25x and >= 20x the A/A", effect >= 1.25 and (effect - 1) * 100 >= 20 * aa,
         "%.4fx vs A/A %.3f %%" % (effect, aa)),
        ("one digest, and it is pass 10's", digests == [ANCHOR_SHA16], str(digests)),
        ("lever arm actually diverged", all(n and n > 0 for n in diverged), str(diverged)),
    ]
    print()
    ok = True
    for name, passed, detail in checks:
        print("%-38s %s  %s" % (name, "PASS" if passed else "FAIL", detail))
        ok &= bool(passed)

    gen, gen_src = stage_generator()
    filt, filt_src = stage_filter()
    print("\nstage table: PXDesign-d %.2f s (%s), Protenix filter %.2f s (%s)"
          % (gen, gen_src, filt, filt_src))
    for name, fold_s in (("incumbent", inc_mean), ("scoped lever", lev_mean)):
        af2 = DESIGNS * fold_s
        total = gen + af2 + filt
        print("  %-13s AF2 %8.2f s (%.2fx)  pipeline %8.2f s  bar %.3fx"
              % (name, af2, af2 / H200["af2ig"], total, total / H200_TOTAL))

    print("\nE2 %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
