#!/usr/bin/env python3
"""The fine pass's shape, across a whole refinement.

Reads the TT_RELION_SHAPE logs the shape arm writes (one line per fine call, one file per rank)
and reports the distribution of every quantity a parallelisation choice would turn on. The point
is that none of these are constants: the fine pass runs on a data-dependent significant subset, so
a scheme tuned on one iteration is the tuned-at-512 trap in a new costume
(memory tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa).

  python3 shape_report.py /home/ttuser/relion-scratch/fine/shape
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np


def main(stem):
    rows = []
    for path in sorted(glob.glob(stem + ".*")):
        for line in open(path):
            f = line.split()
            if len(f) != 7 or f[0] != "F":
                continue
            rows.append([int(v) for v in f[1:]])
    if not rows:
        print("no fine-shape lines under %s.* -- did the arm run with TT_RELION_SHAPE set?" % stem)
        return 1
    a = np.array(rows, dtype=np.int64)
    O, T, SIG, JOBS, P, DISTINCT = (a[:, i] for i in range(6))

    def dist(name, v, fmt="%d"):
        v = np.asarray(v, dtype=np.float64)
        q = np.percentile(v, [0, 25, 50, 75, 99, 100])
        print(("%-28s n=%-7d mean=" + fmt + "  min/p25/med/p75/p99/max = "
               + " / ".join([fmt] * 6))
              % ((name, len(v), v.mean()) + tuple(q)))

    print("fine-pass calls logged: %d  (across %d ranks)"
          % (len(a), len(glob.glob(stem + ".*"))))
    dist("orientation_num", O, "%.1f")
    dist("translation_num", T, "%.1f")
    dist("significant_num", SIG, "%.1f")
    dist("job_num_count", JOBS, "%.1f")
    dist("image_size", P, "%.0f")
    dist("distinct rot_idx", DISTINCT, "%.1f")
    dist("jobs per distinct orient", JOBS / np.maximum(DISTINCT, 1), "%.3f")
    dist("density sig/(O*T)", SIG / np.maximum(O * T, 1), "%.4f")

    print()
    print("distinct translation_num values: %s"
          % dict(zip(*[x.tolist() for x in np.unique(T, return_counts=True)])))
    print("distinct image_size values:      %s"
          % dict(zip(*[x.tolist() for x in np.unique(P, return_counts=True)])))
    print()
    # The number the kernel design turns on: RELION's CPU fine kernel re-projects once per job,
    # the whole-tensor form projects once per distinct orientation.
    saving = JOBS.sum() / max(DISTINCT.sum(), 1)
    print("projection work, RELION's fine kernel / the whole-tensor form: %.3fx" % saving)

    out = stem + "_report.json"
    with open(out, "w") as fh:
        json.dump({
            "calls": int(len(a)),
            "orientation_num": {"mean": float(O.mean()), "median": float(np.median(O)),
                                "p99": float(np.percentile(O, 99)), "max": int(O.max())},
            "translation_num": {"values": {str(k): int(v) for k, v in
                                           zip(*np.unique(T, return_counts=True))}},
            "significant_num": {"mean": float(SIG.mean()), "median": float(np.median(SIG)),
                                "p99": float(np.percentile(SIG, 99)), "max": int(SIG.max())},
            "job_num_count": {"mean": float(JOBS.mean()), "median": float(np.median(JOBS))},
            "image_size": {"values": {str(k): int(v) for k, v in
                                      zip(*np.unique(P, return_counts=True))}},
            "jobs_per_distinct_orientation": float(saving),
            "density": float((SIG / np.maximum(O * T, 1)).mean()),
        }, fh, indent=2)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  "/home/ttuser/relion-scratch/fine/shape"))
