#!/usr/bin/env python3
"""Price the layout / dtype / fidelity axes of a Boltz-2 512 aa fold from the wave-1
device-profiler CSVs (qb2 card 0, one Blackhole of a p300c, tt-metal v0.68.0 source build).

The CSVs are per-op device-profiler dumps taken by ws:b2z-kernel-cycle-census; they carry
per-op dtype, layout, math fidelity, shapes and CB-wait, which the JSON census dropped.
We re-window them by matching the JSON census's op-code signature, then answer four
questions with no new device time:

  1. how many programs in a block/step move data without computing anything, and what they cost
  2. where fp32 storage still reaches the device in the folding path
  3. what math fidelity each site actually runs at, and what that site costs
  4. whether cost tracks tile COUNT (the 71.3 ns/tile claim) or something else

Usage: census.py <block_csv.gz> <block_census.json> <step_csv.gz> <step_census.json> <outdir>
"""
import csv
import gzip
import re
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Programs that perform no arithmetic on their operands: they exist only to change where a
# datum lives or how it is laid out. Slice is in the list because on ttnn a slice is a copy,
# never a view (memory tt-bio-ttnn-slice-not-a-view-and-allocation-order-sensitivity).
MOVEMENT = {
    "TransposeDeviceOperation", "PermuteDeviceOperation", "SliceDeviceOperation",
    "ReshapeViewDeviceOperation", "ConcatDeviceOperation", "CopyDeviceOperation",
    "CloneOperation", "PadDeviceOperation",
    "TilizeDeviceOperation", "TilizeWithValPaddingDeviceOperation",
    "UntilizeDeviceOperation", "UntilizeWithUnpaddingDeviceOperation",
    "NlpCreateHeadsDeviceOperation", "NLPConcatHeadsDeviceOperation",
}
# Of those, the ones that are a pure layout transition in the workstream's sense.
LAYOUT_ONLY = {
    "TilizeDeviceOperation", "TilizeWithValPaddingDeviceOperation",
    "UntilizeDeviceOperation", "UntilizeWithUnpaddingDeviceOperation",
    "ReshapeViewDeviceOperation",
}


def load(csv_gz, census_json):
    cen = json.load(open(census_json))
    sig = [o["op"] for o in cen["ops"]]
    rows = list(csv.DictReader(gzip.open(csv_gz, "rt")))
    codes = [r["OP CODE"] for r in rows]
    n = len(sig)
    hits = [i for i in range(len(codes) - n) if codes[i:i + n] == sig]
    if not hits:
        raise SystemExit(f"op-code signature of {census_json} not found in {csv_gz}")
    # Take a repetition whose successor starts exactly n rows later: that one is a clean,
    # uninterrupted window with no profiler-inserted rows in the middle.
    win = next((h for h, nxt in zip(hits, hits[1:]) if nxt - h == n), hits[-1])
    return cen, rows[win:win + n]


def f(r, k, d=0.0):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _dim(v):
    """'512[512]' -> 512 (the PADDED extent, which is what the device actually moves)."""
    if not v:
        return 0
    m = re.match(r"\s*(\d+)", str(v))
    return int(m.group(1)) if m else 0


def shape(r, slot):
    return tuple(_dim(r.get(f"{slot}_{a}_PAD[LOGICAL]", "")) for a in "WZYX")


def fp32_acc(r):
    """True if this program accumulates in fp32 in DST, which halves the tiles a DST pass holds."""
    return "fp32_dest_acc_en=1" in (r.get("ATTRIBUTES") or "")


def ksrc(r):
    """Shortest identifying name of the compute kernel this program ran."""
    s = (r.get("COMPUTE KERNEL SOURCE") or "").strip()
    names = re.findall(r"([A-Za-z0-9_]+)\.cpp", s)
    return "+".join(sorted(set(names))) or "-"


def tiles(sh):
    """Tiles a 32x32-tiled tensor of this shape occupies."""
    if len(sh) != 4 or sh[2] == 0 or sh[3] == 0:
        return 0
    w, z, y, x = sh
    return max(w, 1) * max(z, 1) * ((int(y) + 31) // 32) * ((int(x) + 31) // 32)


def analyse(tag, cen, rows):
    span = cen["median_ms"]["span_ms"]
    kern = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in rows) / 1e6

    by_code = defaultdict(lambda: [0, 0.0, 0.0])           # n, kernel_ms, trisc1_ms
    move_rows, fp32_rows, fid = [], [], defaultdict(lambda: [0, 0.0])
    site = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    fp32acc_rows = []
    for r in rows:
        code = r["OP CODE"]
        ms = f(r, "DEVICE KERNEL DURATION [ns]") / 1e6
        t1 = f(r, "DEVICE TRISC1 KERNEL DURATION [ns]") / 1e6
        e = by_code[code]
        e[0] += 1
        e[1] += ms
        e[2] += t1
        if code in MOVEMENT:
            move_rows.append(r)
        dts = [r.get(f"INPUT_{i}_DATATYPE", "") for i in range(4)] + [r.get("OUTPUT_0_DATATYPE", "")]
        if any(d.strip() == "FLOAT32" for d in dts):
            fp32_rows.append(r)
        if fp32_acc(r):
            fp32acc_rows.append(r)
        mf = (r.get("MATH FIDELITY") or "-").strip() or "-"
        fid[mf][0] += 1
        fid[mf][1] += ms
        key = (code, ksrc(r), mf, "fp32acc" if fp32_acc(r) else "bf16acc",
               "x".join(map(str, shape(r, "INPUT_0"))),
               "x".join(map(str, shape(r, "OUTPUT_0"))),
               (r.get("INPUT_0_DATATYPE") or "").strip(),
               (r.get("OUTPUT_0_DATATYPE") or "").strip(),
               (r.get("INPUT_0_LAYOUT") or "").strip(),
               (r.get("OUTPUT_0_LAYOUT") or "").strip())
        e2 = site[key]
        e2[0] += 1
        e2[1] += ms
        e2[2] += t1
        e2[3] += f(r, "CORE COUNT")

    move_ms = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in move_rows) / 1e6
    layout_rows = [r for r in rows if r["OP CODE"] in LAYOUT_ONLY]
    layout_ms = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in layout_rows) / 1e6
    fp32_ms = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in fp32_rows) / 1e6
    fp32acc_ms = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in fp32acc_rows) / 1e6

    # tile-rate check: cost per input+output tile for every op, so we can see whether the
    # 71.3 ns/tile datum rate is a property of the silicon or of a handful of sites.
    rate = []
    for r in rows:
        t = sum(tiles(shape(r, f"INPUT_{i}")) for i in range(4)) + tiles(shape(r, "OUTPUT_0"))
        ms = f(r, "DEVICE KERNEL DURATION [ns]")
        if t:
            rate.append((ms / t, t, ms, r["OP CODE"], r.get("MATH FIDELITY", "")))

    return {
        "tag": tag, "span_ms": span, "kernel_ms_sum": kern, "programs": len(rows),
        "by_code": {k: {"n": v[0], "kernel_ms": round(v[1], 4), "trisc1_ms": round(v[2], 4)}
                    for k, v in sorted(by_code.items(), key=lambda kv: -kv[1][1])},
        "movement": {"n": len(move_rows), "kernel_ms": round(move_ms, 4),
                     "pct_of_span": round(100 * move_ms / span, 2)},
        "layout_only": {"n": len(layout_rows), "kernel_ms": round(layout_ms, 4),
                        "pct_of_span": round(100 * layout_ms / span, 2)},
        "fp32_touching": {"n": len(fp32_rows), "kernel_ms": round(fp32_ms, 4),
                          "pct_of_span": round(100 * fp32_ms / span, 2),
                          "by_code": dict(sorted(
                              ((k, sum(1 for r in fp32_rows if r["OP CODE"] == k))
                               for k in {r["OP CODE"] for r in fp32_rows}), key=lambda kv: -kv[1]))},
        "fidelity": {k: {"n": v[0], "kernel_ms": round(v[1], 4)} for k, v in
                     sorted(fid.items(), key=lambda kv: -kv[1][1])},
        "fp32_dest_acc": {"n": len(fp32acc_rows), "kernel_ms": round(fp32acc_ms, 4),
                          "pct_of_span": round(100 * fp32acc_ms / span, 2)},
        "sites": [
            {"op": k[0], "kernel": k[1], "fidelity": k[2], "acc": k[3],
             "in0": k[4], "out0": k[5], "in0_dt": k[6], "out0_dt": k[7],
             "in0_layout": k[8], "out0_layout": k[9],
             "n": v[0], "kernel_ms": round(v[1], 4), "trisc1_ms": round(v[2], 4),
             "cores_avg": round(v[3] / v[0], 1),
             "pct_of_span": round(100 * v[1] / span, 3)}
            for k, v in sorted(site.items(), key=lambda kv: -kv[1][1])],
        "ns_per_tile": {
            "median": round(statistics.median(x[0] for x in rate), 2),
            "mean": round(statistics.mean(x[0] for x in rate), 2),
            "n_ops": len(rate),
            "total_tiles": sum(x[1] for x in rate),
            "aggregate": round(sum(x[2] for x in rate) / max(sum(x[1] for x in rate), 1), 2),
        },
        "movement_detail": [
            {"i": i, "op": r["OP CODE"], "ms": round(f(r, "DEVICE KERNEL DURATION [ns]") / 1e6, 4),
             "cores": int(f(r, "CORE COUNT")),
             "in0": shape(r, "INPUT_0"), "in0_dt": r.get("INPUT_0_DATATYPE", ""),
             "in0_layout": r.get("INPUT_0_LAYOUT", ""),
             "out0": shape(r, "OUTPUT_0"), "out0_dt": r.get("OUTPUT_0_DATATYPE", ""),
             "out0_layout": r.get("OUTPUT_0_LAYOUT", "")}
            for i, r in enumerate(rows) if r["OP CODE"] in MOVEMENT],
        "fp32_detail": [
            {"i": i, "op": r["OP CODE"], "ms": round(f(r, "DEVICE KERNEL DURATION [ns]") / 1e6, 4),
             "fidelity": r.get("MATH FIDELITY", ""),
             "in": [(shape(r, f"INPUT_{j}"), r.get(f"INPUT_{j}_DATATYPE", "")) for j in range(3)],
             "out": (shape(r, "OUTPUT_0"), r.get("OUTPUT_0_DATATYPE", ""))}
            for i, r in enumerate(rows) if r in fp32_rows],
    }


def main():
    bcsv, bjson, scsv, sjson, outdir = sys.argv[1:6]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for tag, c, j in (("PairformerLayer", bcsv, bjson), ("DiffusionStep", scsv, sjson)):
        cen, rows = load(c, j)
        res[tag] = analyse(tag, cen, rows)
    (out / "census.json").write_text(json.dumps(res, indent=1))

    for tag, a in res.items():
        print(f"\n===== {tag}: {a['programs']} programs, span {a['span_ms']:.4f} ms, "
              f"kernel sum {a['kernel_ms_sum']:.4f} ms =====")
        print(f"  movement programs   : {a['movement']['n']:4d}  "
              f"{a['movement']['kernel_ms']:8.4f} ms  {a['movement']['pct_of_span']:5.2f} % of span")
        print(f"  layout-only programs: {a['layout_only']['n']:4d}  "
              f"{a['layout_only']['kernel_ms']:8.4f} ms  {a['layout_only']['pct_of_span']:5.2f} %")
        print(f"  fp32-touching       : {a['fp32_touching']['n']:4d}  "
              f"{a['fp32_touching']['kernel_ms']:8.4f} ms  {a['fp32_touching']['pct_of_span']:5.2f} %")
        print(f"  ns/tile aggregate   : {a['ns_per_tile']['aggregate']}  "
              f"(median op {a['ns_per_tile']['median']}, {a['ns_per_tile']['total_tiles']} tiles)")
        print(f"  fp32 DEST accumulate: {a['fp32_dest_acc']['n']:4d}  "
              f"{a['fp32_dest_acc']['kernel_ms']:8.4f} ms  {a['fp32_dest_acc']['pct_of_span']:5.2f} %")
        print("  fidelity:", {k: (v["n"], round(v["kernel_ms"], 2)) for k, v in a["fidelity"].items()})
        print(f"  distinct sites: {len(a['sites'])}")
        print("  top op codes:")
        for k, v in list(a["by_code"].items())[:10]:
            print(f"    {k:42s} {v['n']:4d}  {v['kernel_ms']:8.4f} ms")
        if a["fp32_touching"]["n"]:
            print("  fp32 by op code:", a["fp32_touching"]["by_code"])
    print(f"\nwrote {out / 'census.json'}")


if __name__ == "__main__":
    main()
