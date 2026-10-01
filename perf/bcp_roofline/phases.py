#!/usr/bin/env python3
"""Per-phase (seam) family table for the profiled round: kernel s, DRAM GB and % of roof.

    phases.py <cpp report .csv.gz> <tracy_ops_data.csv> [--rounds 4] [--roof-gbs 512]

Seams are cut on the card's own clock: the profiler drain after every seam leaves a > 20 ms gap
on the device timeline, so each run of back-to-back ops is one seam. A seam is named by its op
count, which is fixed per seam on this program (Evoformer forward 7,584 / 7,728, Evoformer backward
30,386 / 31,007, extra-MSA forward 320 and backward 1,432, template forward 164); everything else is
`other` (template backward, marshalling, the lazy trunk load).
"""
import argparse
import collections
import csv
import gzip
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import census as C  # noqa: E402

NAMES = {7584: "evo_fwd", 7728: "evo_fwd", 30386: "evo_bwd", 31007: "evo_bwd", 320: "xmsa_fwd",
         1432: "xmsa_bwd", 164: "tmpl_fwd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpp")
    ap.add_argument("opsdata")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--roof-gbs", type=float, default=512.0)
    a = ap.parse_args()
    meta = {}
    for o in C.ops(a.opsdata):
        ins = [C.tensor_info(t) for t in o.get("input_tensors", []) if isinstance(t, dict)]
        outs = [C.tensor_info(t) for t in o.get("output_tensors", []) if isinstance(t, dict)]
        code = o.get("op_code", "")
        ks = o.get("kernel_info", {}).get("compute_kernels", [])
        src = ks[0].get("source", "") if ks else ""
        dram = sum(b for _, b, buf in ins + outs if buf == "DRAM")
        if outs and (code == "SliceDeviceOperation" or "rne_add" in src):
            dram = (2 if code == "SliceDeviceOperation" else 3) * sum(b for _, b, _x in outs)
        fam = ("Generic:" + src.rsplit("/", 1)[-1]) if code == "GenericOpDeviceOperation" else code
        meta[int(o["global_call_count"])] = (fam, dram)
    rows = []
    for r in csv.DictReader(gzip.open(a.cpp, "rt")):
        rows.append((int(r["DEVICE FW START CYCLE"]), int(r["DEVICE FW END CYCLE"]),
                     float(r["DEVICE KERNEL DURATION [ns]"] or 0) * 1e-9,
                     int(r["GLOBAL CALL COUNT"])))
    rows.sort()
    cyc_per_s = 1.35e9                       # AICLK 1350, read off FW cycles / FW ns in this report
    segs, cur = [], [rows[0]]
    for p, q in zip(rows, rows[1:]):
        if (q[0] - p[1]) / cyc_per_s > 0.02:
            segs.append(cur)
            cur = []
        cur.append(q)
    segs.append(cur)
    ph = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    span = collections.Counter()
    for sg in segs:
        name = NAMES.get(len(sg), "other")
        span[name] += (sg[-1][1] - sg[0][0]) / cyc_per_s
        for s, e, k, g in sg:
            fam, dram = meta[g]
            ph[name][fam].update(kernel=k, dram=dram, calls=1)
            ph[name]["_all"].update(kernel=k, dram=dram, calls=1)
    R = a.rounds
    print(f"{'phase':10s} {'calls':>7s} {'kern s':>7s} {'span s':>7s} {'busy':>6s} {'DRAM GB':>8s} "
          f"{'GB/s':>6s} {'%roof':>6s}")
    for name in ("evo_fwd", "evo_bwd", "xmsa_fwd", "xmsa_bwd", "tmpl_fwd", "other"):
        c = ph[name]["_all"]
        k, gb = c["kernel"] / R, c["dram"] / R / 1e9
        print(f"{name:10s} {c['calls'] / R:7.0f} {k:7.3f} {span[name] / R:7.3f} "
              f"{k / (span[name] / R) * 100 if span[name] else 0:5.1f}% {gb:8.1f} "
              f"{gb / k if k else 0:6.1f} {gb / k / a.roof_gbs * 100 if k else 0:5.1f}%")
    for name in ("evo_fwd", "evo_bwd"):
        print(f"\n== {name}, top families ==")
        for fam, c in sorted(ph[name].items(), key=lambda x: -x[1]["kernel"])[1:14]:
            k, gb = c["kernel"] / R, c["dram"] / R / 1e9
            print(f"  {fam[:40]:40s} {c['calls'] / R:7.0f} {k:7.3f} {gb:7.1f} GB "
                  f"{gb / k / a.roof_gbs * 100 if k else 0:5.1f}% roof")


if __name__ == "__main__":
    main()
