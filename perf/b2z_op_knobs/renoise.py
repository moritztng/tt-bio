#!/usr/bin/env python3
"""Re-measure the instances whose A/A floor came back above 2 in the corrected sweep.

Seven of the twenty-four instances in `resweep_wh_c1.json` ran while whglx was carrying other
swarm rows and returned an A/A floor between 2.16 and 3.31, so their arms were reported but
excluded from every conclusion. One of them, `PairformerLayer#004`, is 6 % of device time and
showed an apparent 2.97x -- if that is real the NO-GO verdict is wrong, so it has to be settled
rather than dropped.

The fix is not more chips, it is more alternation. `resweep_shipped.py` scored an arm against the
median of the two incumbent runs bracketing it, which a host-load excursion lasting longer than one
bracket walks straight through. Here every arm is run as `--pairs` INDEPENDENT (incumbent, arm)
pairs spread across the instance's whole run, and the reported ratio is the median of the per-pair
ratios. A drift that moves both halves of a pair cancels inside the pair.

The A/A floor is measured by the SAME estimator with the arm replaced by another incumbent run, so
the control breaks exactly what the measurement reads (memory
`negative-control-must-break-what-check-reads`). A sound instrument puts it at 1.000.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import op_replay as R                                                     # noqa: E402
from sweep import run_one                                                 # noqa: E402
from resweep_shipped import torch_ref, err_vs                             # noqa: E402

ARMS = ["", "nogrid=1", "fp32acc=0", "fidelity=HiFi2,fp32acc=0",
        "fidelity=LoFi,fp32acc=0", "grid=8x8"]
ARMS_FULL = ARMS + ["fidelity=HiFi2", "packerl1=0", "dstfull=1", "fp32acc=0,dstfull=1",
                    "grid=8x4", "outbuf=L1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "bench_manifest.json"))
    ap.add_argument("--resweep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aa-min", type=float, default=1.20,
                    help="re-run instances whose recorded A/A floor was at least this")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--all-arms", action="store_true", help="the full knob set, not the subset")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--bursts", type=int, default=5)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    R._init_tables(ttnn)

    recs = {r["id"]: r for r in json.load(open(a.manifest))}
    prev = json.load(open(a.resweep))["instances"]
    todo = [i for i in prev if (i.get("aa_floor") or 99) >= a.aa_min]
    arms = ARMS_FULL if a.all_arms else ARMS

    device = R.open_device_retry(ttnn)
    g = device.compute_with_storage_grid_size()
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "grid": f"{g.x}x{g.y}",
                   "arch": str(device.arch()), "reps": a.reps, "bursts": a.bursts,
                   "pairs": a.pairs, "estimator": "median of per-pair (incumbent/arm) ratios"},
           "instances": []}
    print(f"# grid {g.x}x{g.y}  re-running {len(todo)} instances "
          f"x {len(arms)} arms x {a.pairs} pairs", flush=True)

    for p in todo:
        iid = p["id"]
        rec = recs[iid]
        e = {"id": iid, "shapes": p["shapes"], "ms_per_fold": p["ms_per_fold"],
             "unit_path": rec["unit_path"], "shipped_config": p["shipped_config"],
             "prev_aa_floor": p.get("aa_floor"), "arms": []}
        print(f"== {iid} {'|'.join(p['shapes'])} prev A/A {p.get('aa_floor')}", flush=True)
        ref = torch_ref(torch, rec)

        for knob in arms:
            label = knob or "(A/A control)"
            ratios, us_all, row = [], [], {"knob": label, "raw_knob": knob}
            try:
                for _ in range(a.pairs):
                    inc, _, _ = run_one(ttnn, torch, rec, device, "", a.reps, a.bursts, False)
                    arm, _, o = run_one(ttnn, torch, rec, device, knob, a.reps, a.bursts, True)
                    ratios.append(inc / arm)
                    us_all.append(arm)
                row["ratio"] = round(st.median(ratios), 4)
                row["ratio_spread"] = round(max(ratios) / min(ratios), 4)
                row["us"] = round(st.median(us_all), 2)
                row["pair_ratios"] = [round(x, 4) for x in ratios]
                row.update(err_vs(o, ref))
            except Exception as ex:                                       # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
            e["arms"].append(row)
            msg = row.get("error") or (
                f"{row['ratio']:6.3f}x  (pairs {row['pair_ratios']}, spread "
                f"{row['ratio_spread']:.3f})")
            print(f"   {label:26s} {msg}", flush=True)
            json.dump(out | {"instances": out["instances"] + [e]}, open(a.out, "w"), indent=1)

        aa = [x for x in e["arms"] if x["raw_knob"] == ""]
        e["aa_floor"] = aa[0].get("ratio_spread") if aa else None
        e["aa_ratio"] = aa[0].get("ratio") if aa else None
        out["instances"].append(e)
        json.dump(out, open(a.out, "w"), indent=1)

    ttnn.close_device(device)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
