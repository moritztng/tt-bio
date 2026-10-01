#!/usr/bin/env python3
"""Dump the profiled round's op sequence, one row per device op, cut into seams.

    seq_dump.py <cpp report .csv.gz> <tracy_ops_data.csv> <out.tsv.gz>

Reuses bcp-roofline's parsers (census.ops / tensor_info) and its seam cut (a > 20 ms gap on the
card's clock, phases.py). Columns: seam index, seam name, position in seam, global call count,
family, kernel ns, DRAM bytes, input shapes, output shapes, compute kernel source.
"""
import csv
import gzip
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bcp_roofline"))
import census as C  # noqa: E402
import phases as P  # noqa: E402


def main():
    cpp, opsdata, out = sys.argv[1:4]
    meta = {}
    for o in C.ops(opsdata):
        ins = [C.tensor_info(t) for t in o.get("input_tensors", []) if isinstance(t, dict)]
        outs = [C.tensor_info(t) for t in o.get("output_tensors", []) if isinstance(t, dict)]
        code = o.get("op_code", "")
        ks = o.get("kernel_info", {}).get("compute_kernels", [])
        src = ks[0].get("source", "") if ks else ""
        dram = sum(b for _, b, buf in ins + outs if buf == "DRAM")
        if outs and (code == "SliceDeviceOperation" or "rne_add" in src):
            dram = (2 if code == "SliceDeviceOperation" else 3) * sum(b for _, b, _x in outs)
        fam = code
        if code == "GenericOpDeviceOperation":
            fam = "Generic:" + src.rsplit("/", 1)[-1]
        if code == "BinaryNgDeviceOperation":
            fam = "BinaryNg:" + o.get("attributes", {}).get("binary_op_type", "").split("::")[-1]
        if code == "UnaryDeviceOperation":
            fam = "Unary:" + str(o.get("attributes", {}).get("op_chain", ""))[:40]
        sh = lambda xs: ";".join("x".join(map(str, s)) + ("" if buf == "DRAM" else "@" + buf[:2])
                                 for s, _b, buf in xs)
        meta[int(o["global_call_count"])] = (fam, dram, sh(ins), sh(outs),
                                             src.rsplit("/", 1)[-1])
    rows = []
    for r in csv.DictReader(gzip.open(cpp, "rt")):
        rows.append((int(r["DEVICE FW START CYCLE"]), int(r["DEVICE FW END CYCLE"]),
                     float(r["DEVICE KERNEL DURATION [ns]"] or 0), int(r["GLOBAL CALL COUNT"])))
    rows.sort()
    segs, cur = [], [rows[0]]
    for p, q in zip(rows, rows[1:]):
        if (q[0] - p[1]) / 1.35e9 > 0.02:
            segs.append(cur)
            cur = []
        cur.append(q)
    segs.append(cur)
    with gzip.open(out, "wt") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["seam", "name", "pos", "gcc", "fam", "kern_ns", "dram_b", "ins", "outs", "src"])
        for i, sg in enumerate(segs):
            name = P.NAMES.get(len(sg), "other")
            for j, (_s, _e, k, g) in enumerate(sg):
                fam, dram, si, so, src = meta.get(g, ("?", 0, "", "", ""))
                w.writerow([i, name, j, g, fam, int(k), int(dram), si, so, src])


if __name__ == "__main__":
    main()
