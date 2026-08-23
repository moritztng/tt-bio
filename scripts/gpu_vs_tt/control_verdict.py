#!/usr/bin/env python3
"""Score a harness-control run against an already-published cell, and say so in a file.

H200 is the perf page's index platform: the cost model sets DGX H200 = 1.00x and the
"beat the Nvidia server" bar is derived from an H200 number, so a new H200 cell that is not
comparable to the published column moves the bar every other row is judged against. Before any
new row is trusted, one already-published cell is re-measured and this script decides whether it
reproduced -- with the bands fixed in advance, so the answer cannot be chosen after seeing it.

    COMPARABLE            |d| <= 3 %   the same-machine H200 leg of perf-page-host-device-split
                                       reproduced four published cells inside 2.3 %
    COMPARABLE-WITH-NOTE  3-5 %        the B200 pass's config-identical-sibling repro spanned
                                       +3.0 % to -4.1 %; real stack/machine drift, still usable
                                       if the doc states it
    NOT-COMPARABLE        > 5 %        larger than any reproduction this campaign has measured

Writes CONTROL_RERUN_WANTED next to --out for anything but COMPARABLE. That file is what makes
the box re-run the control against the published package pins unattended, instead of the agent
diagnosing a stack difference by hand while the meter runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True, help="the control run's result JSON")
    ap.add_argument("--published", type=float, required=True, help="the published cell, seconds")
    ap.add_argument("--label", default="", help="what the control arm was, for the record")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rec = {"result_file": str(args.result), "published_s": args.published, "arm": args.label}
    if not args.result.exists():
        rec.update(verdict="NO-RESULT", why="the control produced no result JSON")
    else:
        d = json.loads(args.result.read_text())
        r = d.get("result") or {}
        med = r.get("warm_median_s")
        if d.get("error") or med is None:
            rec.update(verdict="NO-RESULT", why=f"error={d.get('error')!r} warm_median_s={med!r}")
        else:
            delta = 100.0 * (med - args.published) / args.published
            rec.update(measured_s=med, warm_n=r.get("warm_n"), spread_pct=r.get("warm_spread_pct"),
                       cold_s=r.get("cold_s"), delta_pct=round(delta, 2),
                       packages=d.get("packages"), torch=d.get("torch_version"),
                       gpu=d.get("gpu"), host_cpu=d.get("host_cpu"),
                       vcpu_cgroup=d.get("vcpu_cgroup"), peak_mem_MiB=d.get("peak_mem_MiB"))
            a = abs(delta)
            rec["verdict"] = ("COMPARABLE" if a <= 3.0 else
                              "COMPARABLE-WITH-NOTE" if a <= 5.0 else "NOT-COMPARABLE")
            rec["why"] = (f"{med} s against a published {args.published} s is {delta:+.2f} %")

    args.out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    if rec["verdict"] != "COMPARABLE":
        (args.out.parent / "CONTROL_RERUN_WANTED").write_text(rec["verdict"] + "\n")
        print("wrote CONTROL_RERUN_WANTED: the pinned-package control arm should run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
