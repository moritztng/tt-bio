#!/usr/bin/env python3
"""Leg 5: the two-arm byte census, family by family.

Both arms are the same block harness with `--rne-kernel` the only difference, so a family whose
bytes move and is not the residual is the kernel doing something the flag did not advertise.

Per-block counters are composed into a round exactly the way `bcx-p10-devmap/analyze.py` does it
(2 taped forwards + 1 backward, 48 Evoformer blocks + 4 extra-MSA), so the GB here are
comparable with `bcx-CALLS.md` leg 3 and not a different unit.
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bcx_p10_devmap"))
from analyze import per_block, to_round          # noqa: E402

TOL_GB = 0.05      # a family under this is byte-identical between the arms


def fam_rows(path):
    blob = json.load(open(path))
    rows = to_round(per_block(blob))
    out = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        e = out[r["family"]]
        e["GB"] += r["read_GB"] + r["written_GB"]
        e["read_GB"] += r["read_GB"]
        e["written_GB"] += r["written_GB"]
        e["calls"] += r["calls"]
        e["device_s"] += r["device_s"]
    return blob, out


def clk(blob):
    lo, hi, med = [], [], []
    for rec in blob["records"]:
        a = rec.get("aiclk") or {}
        if a.get("median"):
            lo.append(a["min"]); hi.append(a["max"]); med.append(a["median"])
    return (min(lo), max(hi), sorted(med)[len(med) // 2]) if med else (0, 0, 0)


def main(p0, p1):
    b0, f0 = fam_rows(p0)
    b1, f1 = fam_rows(p1)
    fams = sorted(set(f0) | set(f1), key=lambda k: -(f0[k]["GB"] + f1[k]["GB"]))
    print("AICLK off %s   on %s   (min, max, median over every timed record)" % (clk(b0), clk(b1)))
    print("loadavg off %.2f->%.2f   on %.2f->%.2f"
          % (b0["loadavg_start"][0], b0["loadavg_end"][0],
             b1["loadavg_start"][0], b1["loadavg_end"][0]))
    print()
    print("%-18s %9s %9s %9s %8s %9s %9s %8s"
          % ("family", "off_GB", "on_GB", "dGB", "ratio", "off_calls", "on_calls", "dcalls"))
    tot = [0.0, 0.0]
    moved = []
    for fam in fams:
        a, b = f0[fam], f1[fam]
        d = b["GB"] - a["GB"]
        tot[0] += a["GB"]; tot[1] += b["GB"]
        ratio = (a["GB"] / b["GB"]) if b["GB"] else float("inf")
        print("%-18s %9.1f %9.1f %+9.1f %8.2fx %9.0f %9.0f %+8.0f"
              % (fam, a["GB"], b["GB"], d, ratio, a["calls"], b["calls"],
                 b["calls"] - a["calls"]))
        if fam != "residual" and abs(d) > TOL_GB:
            moved.append((fam, d))
    print("%-18s %9.1f %9.1f %+9.1f %8.2fx" % ("TOTAL", tot[0], tot[1], tot[1] - tot[0],
                                               tot[0] / tot[1] if tot[1] else 0))
    print()
    r0, r1 = f0["residual"], f1["residual"]
    print("residual: %.1f -> %.1f GB a round (%.2fx, %+.1f GB), calls %.0f -> %.0f"
          % (r0["GB"], r1["GB"], r0["GB"] / r1["GB"] if r1["GB"] else 0,
             r1["GB"] - r0["GB"], r0["calls"], r1["calls"]))
    if moved:
        print("MOVED, not the residual: " + ", ".join("%s %+.2f GB" % m for m in moved))
    else:
        print("every other family byte-identical within %.2f GB a round" % TOL_GB)
    json.dump({"off": {k: dict(v) for k, v in f0.items()},
               "on": {k: dict(v) for k, v in f1.items()},
               "aiclk": {"off": clk(b0), "on": clk(b1)}},
              open(pathlib.Path(p0).parent / "bytes_ab.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
