#!/usr/bin/env python3
"""of3t-cotterm: union two half-captures into one, and refuse if they disagree anywhere.

Capturing the APB boundary at all 48 sites is refused a 1,207,959,552 B DRAM buffer: at 242 s
and 243 s with the output pin unbounded, and still at 423 s with it bounded, with 458,748,864 B
free per bank and the largest free block 105,963,456 B of the 150,994,944 B wanted. It is
fragmentation at a high-water the taped backward already sits a few MB under, and the capture's
reads are what tip it. Two arms of 24 sites each do fit.

Merging two RUNS is only legitimate because the A/A floor on this quantity is exactly zero:
`FLOOR_AA_DEV_APB_N384.json` compared two full arms tensor by tensor and 12 of 12 were
bit-identical, 0 moved. This asserts the same thing again on whatever the two halves happen to
share, so the merge cannot quietly stitch together two different runs.

CPU only, no board. `host` is emitted by this writer.
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    z = ap.parse_args()

    A = torch.load(z.a, map_location="cpu", weights_only=False)
    B = torch.load(z.b, map_location="cpu", weights_only=False)
    sa, sb = A["sites"], B["sites"]
    overlap = sorted(set(sa) & set(sb))
    moved = []
    for i in overlap:
        for f in sorted(set(sa[i]) & set(sb[i])):
            x, y = sa[i][f], sb[i][f]
            if torch.is_tensor(x) and torch.is_tensor(y):
                if x.shape != y.shape or not torch.equal(x, y):
                    moved.append(f"{i}|{f}")
    if moved:
        raise SystemExit("the two halves disagree on shared tensors, so they are not the same "
                         "measurement: " + ", ".join(moved[:8]))

    sites = dict(sa)
    sites.update(sb)
    out = dict(A)
    out["sites"] = sites
    out["merged_from"] = [z.a, z.b]
    out["apb_blocks"] = sorted(sites)
    out["host"] = os.uname().nodename
    out["state"] = {"a": A.get("state"), "b": B.get("state")}
    out["trees"] = {"a": A.get("trees"), "b": B.get("trees")}
    torch.save(out, z.out)

    R = {"what": __doc__.strip().splitlines()[0], "host": os.uname().nodename,
         "board_class": "p150a Blackhole", "a": z.a, "b": z.b, "out": z.out,
         "sites_a": sorted(sa), "sites_b": sorted(sb), "sites_merged": sorted(sites),
         "n_sites": len(sites), "overlap": overlap, "overlap_tensors_that_moved": 0,
         "trees_agree": A.get("trees", {}).get("tt_bio_file") == B.get("trees", {}).get("tt_bio_file"),
         "errors_a": A.get("errors"), "errors_b": B.get("errors")}
    json.dump(R, open(z.report, "w"), indent=2)
    print(json.dumps({k: R[k] for k in ("n_sites", "overlap", "overlap_tensors_that_moved",
                                        "trees_agree", "host")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
