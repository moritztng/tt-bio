#!/usr/bin/env python3
"""Crossings and bytes per round, per arm, off the meter's own event log.

Every device event carries the shapes of the positional arrays its callback was handed, and
each of those is a float32 array that crossed the seam to the host. The return trip is the
same pair tensor coming back, so a forward moves `pair` in and `pair` out and the backward
moves the pair cotangent each way. Counted, not composed.
"""
import json
import statistics as st
import sys


def main(paths):
    out = {}
    for path in paths:
        d = json.load(open(path))
        ev, stamp = d["events"], d["stamp"]
        arm = "device" if stamp.get("extra_msa_on_device") else "jax"
        starts = [e for e in ev if e["kind"] == "round_start"]
        stop = [e for e in ev if e["kind"] == "round_stop"]
        bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
        rows = out.setdefault(arm, [])
        for i in range(1, len(bounds) - 1):          # round 1 carries the jit compile
            t0, t1 = bounds[i], bounds[i + 1]
            dev = [e for e in ev if e["kind"] == "device" and e.get("t1") is not None
                   and e["t0"] >= t0 and e["t1"] <= t1]
            row = {"crossings": len(dev), "to_host_MB": 0.0}
            for e in dev:
                for shape in e.get("shapes", []):
                    if isinstance(shape, list):
                        n = 1
                        for s in shape:
                            n *= s
                        row["to_host_MB"] += n * 4 / 1e6
                key = e.get("module", "evoformer") + "_" + e["phase"]
                row[key] = row.get(key, 0) + 1
            row["to_host_MB"] = round(row["to_host_MB"], 2)
            rows.append(row)
    for arm, rows in out.items():
        keys = sorted({k for r in rows for k in r})
        print(arm, "n=%d" % len(rows),
              " ".join(f"{k}={st.median([r.get(k, 0) for r in rows])}" for k in keys))


if __name__ == "__main__":
    main(sys.argv[1:])
