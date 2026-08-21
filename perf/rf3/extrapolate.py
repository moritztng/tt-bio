#!/usr/bin/env python3
"""Full-config fold time from a short unit run, and the ladder table it produces.

A full 1024 aa fold at n_recycles=10 / num_steps=50 is ten minutes of card; a unit run at
n_recycles=2 / num_steps=3 gives a warm per-recycle and per-denoiser-call cost and the
one-time phases directly. The method is not assumed: at 512 aa it predicts 81.13 s
against 81.11 s measured, an error of 0.03%, and it predicted the pre-lever 512 aa rung
to 0.2%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
FULL_RECYCLES, FULL_CALLS = 10, 49


def full_from_unit(rec: dict) -> dict:
    m = rec["median_warm"]
    per_recycle = m["recycles"] / rec["n_recycles"]
    per_call = m["diffusion"] / rec["denoiser_calls"]
    total = (m["upload"] + m["feature_init"] + m["distogram"] + m.get("confidence", 0.0)
             + FULL_RECYCLES * per_recycle + FULL_CALLS * per_call)
    return {"aa": rec["aa"], "per_recycle_s": per_recycle, "per_call_s": per_call,
            "one_time_s": m["upload"] + m["feature_init"] + m["distogram"],
            "confidence_s": m.get("confidence", 0.0),
            "recycles_s": FULL_RECYCLES * per_recycle,
            "diffusion_s": FULL_CALLS * per_call, "infer_s": total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("unit", nargs="+", help="win_unit_<aa>.json files")
    args = ap.parse_args()
    rungs = {r["rung_aa"]: r for r in json.loads(
        (REPO / "perf/rf3/gpu_reference.json").read_text())["rungs"] if r["batch"] == 1}
    out = []
    for path in args.unit:
        e = full_from_unit(json.loads(Path(path).read_text()))
        t = rungs.get(e["aa"])
        if t:
            e["tt_target_device_s"] = t["tt_target_device_s"]
            e["ratio_vs_target"] = e["infer_s"] / t["tt_target_device_s"]
            e["gap_vs_h200"] = e["infer_s"] / t["h200_device_s"]
        out.append(e)
    for e in sorted(out, key=lambda x: x["aa"]):
        print(f"{e['aa']:5d} aa  recycles {e['recycles_s']:8.2f}  "
              f"diffusion {e['diffusion_s']:7.2f}  conf {e['confidence_s']:6.2f}  "
              f"one-time {e['one_time_s']:5.2f}  => {e['infer_s']:8.2f} s"
              + (f"  {e['ratio_vs_target']:6.3f}x of target" if "ratio_vs_target" in e
                 else ""))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
