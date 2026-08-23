#!/usr/bin/env python3
"""Recompute both readings out of the published page data and score them against the six rows
`perf-page-host-device-split` measured.

The page's own JS derives the same two ratios from the same fields, so this catches a mistyped
host figure before it is ever rendered. Every expected value here traces to a measured artifact:
the Tenstorrent host shares to perf/perf-page-host-device-split/tt_<model>_qb2c1.json, the NVIDIA
ones to that directory's hgpu.json and h200/, and RoseTTAFold3's to the pass that measured its cell.
"""

import json
import sys
from pathlib import Path

# Defaults to the repo this file sits in. Pass a root to score a copy of the site tree instead,
# which is how the deployed page is checked: curl the two live files into <root>/site/ and point
# this at <root>.
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]

# model -> (host_tt, host_gpu, device_tt, device_gpu, whole_fold, device_only)
EXPECT = {
    "boltz2":      (0.382, 0.240, 23.122,  7.298, 3.118, 3.168),
    "esmfold2":    (0.056, 0.001, 29.337,  7.256, 4.050, 4.043),
    "protenix-v2": (0.101, 0.619, 50.442, 12.186, 3.947, 4.139),
    "opendde":     (0.125, 6.308, 82.268, 20.640, 3.057, 3.986),
    "openfold3":   (1.839, 0.428, 36.415, 10.263, 3.578, 3.548),
    # a998c57b re-folded RF3 on the fixed numerics and republished the cell at 82.547 s
    # (perf/rf3/page512/bisect_fix_p{1,2}_qb2c2.json, session medians 82.451 and 82.840)
    # without updating this row, so this gate sat red on main until 2026-08-23. The
    # numbers below are that cell: 82.547 - 8.330 host = 74.217 device.
    "rf3":         (8.330, 12.459, 74.217, 7.746, 3.621, 9.581),
}


def device_s(cell):
    sp = cell["split"]
    if "device_s" in sp:
        return sp["device_s"]
    return cell["s_per_fold"] - sp["host_s"] if sp.get("in_cell") else cell["s_per_fold"]


def whole_s(cell):
    sp = cell["split"]
    return cell["s_per_fold"] if sp.get("in_cell") else cell["s_per_fold"] + sp["host_s"]


def main():
    page = json.loads((ROOT / "site" / "data" / "perf-512aa.json").read_text())
    plats = {p["id"]: p for p in page["platforms"]}
    bad = []
    seen = set()
    for m in page["models"]:
        t, g = m["cells"]["p150a"], m["cells"]["h200"]
        if "split" not in t or "split" not in g:
            bad.append(f"{m['id']}: no split block")
            continue
        seen.add(m["id"])
        got = (t["split"]["host_s"], g["split"]["host_s"], round(device_s(t), 3), round(device_s(g), 3),
               round(whole_s(t) / whole_s(g), 3), round(device_s(t) / device_s(g), 3))
        want = EXPECT.get(m["id"])
        if want is None:
            bad.append(f"{m['id']}: unexpected model")
        elif any(abs(a - b) > 0.0011 for a, b in zip(got, want)):
            bad.append(f"{m['id']}: got {got} want {want}")
        else:
            print(f"OK   {m['id']:12s} host {got[0]:8.3f} / {got[1]:7.3f}   "
                  f"device {got[2]:7.3f} / {got[3]:6.3f}   whole {got[4]:.3f}x   device-only {got[5]:.3f}x")
        if not (t.get("measured_on") or plats["p150a"].get("measured_on")):
            bad.append(f"{m['id']} p150a: measured_on absent, so the cell does not say which "
                       "board produced it and inherits nothing from the platform")
        for side, cell in (("p150a", t), ("h200", g)):
            if "in_cell" not in cell["split"]:
                bad.append(f"{m['id']} {side}: split.in_cell absent, so whether the host seconds are "
                           "inside the cell is left to a default")
        if t["split"].get("bound") != "upper":
            bad.append(f"{m['id']}: the Tenstorrent host share is an upper bound and must say so")
    for missing in set(EXPECT) - seen:
        bad.append(f"{missing}: row absent")
    if "network forward" not in page["scope"]["protocol"]:
        bad.append("scope.protocol still does not say what the NVIDIA cells time")
    if not page["scope"].get("split"):
        bad.append("scope.split is missing, so the table renders with no caption")
    for mid in ("protenix-v2", "opendde", "rf3"):
        if not next(m for m in page["models"] if m["id"] == mid).get("note"):
            bad.append(f"{mid}: row note missing")
    if bad:
        print("\nFAIL")
        for b in bad:
            print("  " + b)
        return 1
    print("\nPASS, six rows, both readings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
