#!/usr/bin/env python3
"""Turn the profiled grid sweep into a measured scaling curve per op class, and price it.

Joins the sweep's own program configs to the profiler's rows on (input shapes, per_core_M,
per_core_N), so every point carries the core count the DEVICE reported rather than the one the
config implies. Then, for each class:

  best      the fastest point the op's tile counts can reach on this 11x10 grid
  in situ   what production dispatches today
  recovered (t_situ - t_best) / t_situ x s/fold  -- seconds, from the curve, never a core ratio

The in-situ point doubles as the instrument check: its device kernel time must reproduce the
figure `grid_census.py` read out of the shipped fold.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

DIMRE = re.compile(r"\s*(\d+)")
PCM = re.compile(r"per_core_M=(\d+)")
PCN = re.compile(r"per_core_N=(\d+)")


def _dim(v):
    m = DIMRE.match(str(v or ""))
    return int(m.group(1)) if m else 0


def shape(r, pfx):
    return tuple(_dim(r.get(f"{pfx}_{d}_PAD[LOGICAL]", "")) for d in ("W", "Z", "Y", "X"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    obs = defaultdict(list)
    for r in rows:
        code = r.get("OP CODE", "")
        if not code.startswith(("Matmul", "LayerNorm")):
            continue
        att = r.get("ATTRIBUTES", "") or ""
        m, n = PCM.search(att), PCN.search(att)
        key = (shape(r, "INPUT_0"), shape(r, "INPUT_1"),
               int(m.group(1)) if m else None, int(n.group(1)) if n else None)
        try:
            obs[key].append((int(float(r["CORE COUNT"])),
                             float(r["DEVICE KERNEL DURATION [ns]"])))
        except (ValueError, KeyError):
            pass

    sweep = json.loads(a.sweep.read_text())
    cases = defaultdict(list)
    for p in sweep:
        if "us" not in p:
            continue
        cases[p["case"]].append(p)

    # the shapes each case was built from, mirrored from grid_sweep.CASES
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gs", Path(__file__).parent / "grid_sweep.py")
    gs = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["gs"] = gs
    try:
        spec.loader.exec_module(gs)
    except Exception:                                      # noqa: BLE001  (no ttnn on this host)
        gs = None
    shapes = {c["name"]: (c["in0"], c["in1"] or (0, 0, 0, 0), c["situ"][0], c["census_ns"],
                          c["sfold"], c["n"]) for c in gs.CASES} if gs else {}

    out = {}
    for case, pts in cases.items():
        in0, in1, situ_cores, census_ns, sfold, nprog = shapes[case]
        curve = []
        for p in sorted(pts, key=lambda q: q["cores"]):
            key = (tuple(in0), tuple(in1), p["per_core_M"], p["per_core_N"])
            hits = obs.get(key, [])
            if not hits:
                continue
            # several sweep points can share (pcM, pcN); separate them by the reported core count
            want = [h for h in hits if h[0] == p["cores"]] or hits
            cores_dev = st.mode([h[0] for h in want])
            ns = st.median([h[1] for h in want])
            curve.append({"cores_cfg": p["cores"], "cores_device": cores_dev,
                          "per_core_M": p["per_core_M"], "per_core_N": p["per_core_N"],
                          "quantum": p["quantum"], "kernel_us": ns / 1e3,
                          "wall_us": p["us"]})
        if not curve:
            continue
        # dedupe on the core count the device reported, keeping the fastest program for it
        bydev = {}
        for c in curve:
            if c["cores_device"] not in bydev or c["kernel_us"] < bydev[c["cores_device"]]["kernel_us"]:
                bydev[c["cores_device"]] = c
        curve = sorted(bydev.values(), key=lambda c: c["cores_device"])
        situ = next((c for c in curve if c["cores_device"] == situ_cores), None)
        best = min(curve, key=lambda c: c["kernel_us"])
        one = min((c for c in curve), key=lambda c: c["cores_device"])
        out[case] = {
            "in0": in0, "in1": in1, "programs_per_call": nprog,
            "s_per_fold_at_situ": sfold,
            "census_us": census_ns / 1e3,
            "situ_cores": situ_cores,
            "situ_us": situ["kernel_us"] if situ else None,
            "situ_vs_census": (situ["kernel_us"] / (census_ns / 1e3)) if situ else None,
            "max_cores_reachable": max(c["cores_device"] for c in curve),
            "best_cores": best["cores_device"], "best_us": best["kernel_us"],
            "speedup_best_over_situ": (situ["kernel_us"] / best["kernel_us"]) if situ else None,
            "s_per_fold_recoverable": (sfold * (1 - best["kernel_us"] / situ["kernel_us"]))
                                      if situ else None,
            "lowest_cores": one["cores_device"], "lowest_us": one["kernel_us"],
            "parallel_efficiency_pct": (100.0 * (one["kernel_us"] / best["kernel_us"])
                                        / (best["cores_device"] / one["cores_device"])),
            "curve": curve,
        }
    a.out.write_text(json.dumps(out, indent=1))

    print(f"{'case':<20}{'situ':>6}{'us':>9}{'/census':>8}{'maxC':>6}{'best':>6}{'us':>9}"
          f"{'gain':>7}{'s/fold':>8}{'recov s':>9}{'par.eff':>8}")
    tot = 0.0
    for case, d in sorted(out.items(), key=lambda kv: -(kv[1]["s_per_fold_recoverable"] or 0)):
        tot += d["s_per_fold_recoverable"] or 0
        print(f"{case:<20}{d['situ_cores']:>6}{d['situ_us'] or 0:>9.2f}"
              f"{d['situ_vs_census'] or 0:>8.2f}{d['max_cores_reachable']:>6}"
              f"{d['best_cores']:>6}{d['best_us']:>9.2f}"
              f"{d['speedup_best_over_situ'] or 0:>7.3f}{d['s_per_fold_at_situ']:>8.3f}"
              f"{d['s_per_fold_recoverable'] or 0:>9.3f}{d['parallel_efficiency_pct']:>7.1f}%")
    print(f"{'TOTAL':<20}{'':>6}{'':>9}{'':>8}{'':>6}{'':>6}{'':>9}{'':>7}{'':>8}{tot:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
