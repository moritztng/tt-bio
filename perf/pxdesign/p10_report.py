"""Read pass 10's artifacts and price the bar off them, so the verdict is arithmetic on files.

Three fold legs (incumbent, lever, incumbent) give the A/B and the A/A in one window. The two
stage cells are read from M5's artifacts when they exist and fall back to the numbers the stage
table has carried since pass 1 and pass 2, labelled as such in the output.

    PYTHONPATH=. python3 perf/pxdesign/p10_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "perf/pxdesign"

#: H200 device seconds a stage at the matched cell (laczc768_ext_n8_msa, 8 designs).
H200 = {"generator": 208.705, "af2ig": 145.377, "filter": 43.360}
H200_TOTAL = sum(H200.values())
DESIGNS = 8

#: The stage numbers the table carried before this pass, with the pass that measured them.
STALE = {"generator": (226.634, "p1"), "filter": (160.895, "p2")}


def folds() -> list[dict]:
    out = []
    for path in sorted(PERF.glob("tt_pxd_p10_fold_848_leg*.json")):
        d = json.loads(path.read_text())
        d["_file"] = path.name
        out.append(d)
    return out


def stage_generator():
    p = PERF / "tt_pxd_p10_generator_848.json"
    if not p.exists():
        return STALE["generator"]
    cells = json.loads(p.read_text())["cells"]
    hit = [c for c in cells if c["cell"] == "laczc768" and c["multiplicity"] == DESIGNS]
    return (hit[0]["extrap_s"], "p10") if hit else STALE["generator"]


def stage_filter():
    probe, design = PERF / "tt_pxd_p10_filter_probe_768.json", PERF / "tt_pxd_p10_filter_848.json"
    if not (probe.exists() and design.exists()):
        return STALE["filter"]
    a, b = json.loads(probe.read_text()), json.loads(design.read_text())
    if "warm_median" not in a or "warm_median" not in b:
        return STALE["filter"]
    return a["warm_median"]["total"] + DESIGNS * b["warm_median"]["total"], "p10"


def main() -> int:
    legs = folds()
    if not legs:
        print("no fold artifacts yet", file=sys.stderr)
        return 1

    head = ("leg", "arm", "fold_s", "pass_s", "stacks_s")
    print("%-34s %3s %9s %8s %9s  digest" % head)
    for d in legs:
        arm = 1 if "l1padded1" in d["label"] else 0
        print("%-34s %3d %9.3f %8.4f %9.4f  %s" % (
            d["label"], arm, d["fold_s_warm_median"], d["pass_s_warm_median"],
            d["split_warm_mean_s"]["device_stacks_s"],
            ",".join(d.get("structure_sha16_all", ["-"]))))

    inc = [d for d in legs if "l1padded0" in d["label"]]
    lev = [d for d in legs if "l1padded1" in d["label"]]
    if not inc or not lev:
        print("need both arms", file=sys.stderr)
        return 1

    inc_s = [d["fold_s_warm_median"] for d in inc]
    lev_s = [d["fold_s_warm_median"] for d in lev]
    inc_mean, lev_mean = sum(inc_s) / len(inc_s), sum(lev_s) / len(lev_s)
    aa = 100 * (max(inc_s) - min(inc_s)) / inc_mean if len(inc_s) > 1 else float("nan")
    print("\nincumbent %.3f s a design (A/A %.3f %%), lever %.3f s, %.4fx, delta %.3f s" % (
        inc_mean, aa, lev_mean, inc_mean / lev_mean, inc_mean - lev_mean))

    digests = {tuple(d.get("structure_sha16_all", ())) for d in legs}
    one = len(digests) == 1 and len(next(iter(digests))) == 1
    print("structure digests across all legs: %s %s" % (
        "IDENTICAL" if one else "SPLIT", sorted({x for t in digests for x in t})))
    scal = {json.dumps(d["folds"][-1]["scalars"], sort_keys=True) for d in legs}
    print("confidence scalars across all legs: %s" % ("IDENTICAL" if len(scal) == 1 else "SPLIT"))

    gen, gen_src = stage_generator()
    filt, filt_src = stage_filter()
    print("\nstage table: PXDesign-d %.2f s (%s), Protenix filter %.2f s (%s)" % (
        gen, gen_src, filt, filt_src))
    for name, fold_s in (("incumbent", inc_mean), ("lever", lev_mean)):
        af2 = DESIGNS * fold_s
        total = gen + af2 + filt
        print("  %-10s AF2 %8.2f s (%.2fx)  pipeline %8.2f s  bar %.3fx" % (
            name, af2, af2 / H200["af2ig"], total, total / H200_TOTAL))
    print("  4x bar = %.2f s of ported-device time (H200 %.3f s); AF2 allowance %.3f s a design" % (
        4 * H200_TOTAL, H200_TOTAL, (4 * H200_TOTAL - gen - filt) / DESIGNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
