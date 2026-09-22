#!/usr/bin/env python3
"""of3t-f64route: the host float64 softmax's reach, read from a fold's own per-pid census.

`TT_BIO_CAPACITY_CENSUS=<dir>` makes every process that imported `tt_bio.tenstorrent` dump its
counters at exit, one file per pid. That is the only census that survives a fold running in a
process the launcher did not start, which is why every `INFERENCE_AB_*.json` in this campaign
reads `softmax_calls_per_fold` 0 on both arms: its census is an atexit hook in the LAUNCHER, and
the launcher builds no model.

  reach_report.py <census-dir> <model> <out.json>
"""
import glob
import json
import os
import sys


def main() -> int:
    d, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "capacity_*.json"))):
        j = json.load(open(f))
        fp = j.get("fp32_softmax") or {}
        rows.append({
            "pid_file": os.path.basename(f),
            "host_f64_softmax": j.get("host_f64_softmax"),
            "selected_per_site": j.get("host_f64_softmax_selected_per_site"),
            "sites_resolved": j.get("host_f64_softmax_sites"),
            "reach": j.get("host_f64_softmax_reach"),
            "fp32_softmax_calls": fp.get("calls"),
            "fp32_softmax_tail_invocations": fp.get("fused", 0) + fp.get("unfused", 0),
        })
    folding = [r for r in rows if (r["host_f64_softmax"] or {}).get("selected")]
    rep = {"model": model, "env": "TT_BIO_HOST_F64_SOFTMAX_AB=all",
           "what": "arrivals at the host float64 gate in ONE inference fold, counted in the "
                   "process that folded. Every arrival is `refused` because a fold opens no "
                   "tape, which is the inference guarantee measured at the call rather than "
                   "argued from the source.",
           "processes": len(rows),
           "processes_that_built_a_model": len(folding),
           "per_process": rows}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
