"""Cut a device-profiler ops report into the regions perf/mol1/bh/opgap.py fenced, and tabulate where the time goes.

    python perf/mol1/bh/opgap_table.py --ops <ops_perf_results.csv> --regions <regions.json> [--wall <block_walls.json>]
        [--detail 'pf48/b10'] [--detail denoise] [--json out.json]

Per region: ops, sum of kernel time, sum of op-to-op gaps (device idle between one op's end and the next op's
start), and the device span (first op start to last op end). Per kind of region (stack and width, or denoise) the
median over its calls. For each --detail region (first match in the last cycle) every op with its kernel time,
cores and the gap to the next op, then the same ops summed by op code.
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict

FENCE = "1x1x32x224"   # opgap.py's fence tensor; nothing in the model makes it


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def shape_of(row, k):
    """INPUT_k as WxZxYxX, whatever this tt-metal version calls the columns."""
    dims = []
    for d in "WZYX":
        for col in (f"INPUT_{k}_{d}_PAD[LOGICAL]", f"INPUT_{k}_{d}"):
            if col in row and row[col] not in ("", None):
                dims.append(re.sub(r"\[.*\]", "", str(row[col])).strip())
                break
    return "x".join(dims)


def load(ops_csv):
    with open(ops_csv) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("DEVICE KERNEL DURATION [ns]") not in ("", None)]
    key = "DEVICE FW START CYCLE"
    rows.sort(key=lambda r: num(r.get(key)))
    for r in rows:
        r["_k"] = num(r["DEVICE KERNEL DURATION [ns]"])
        r["_fw"] = num(r.get("DEVICE FW DURATION [ns]")) or r["_k"]
        r["_gap"] = num(r.get("OP TO OP LATENCY [ns]"))
        r["_cores"] = int(num(r.get("CORE COUNT")))
        r["_in0"] = shape_of(r, 0)
        r["_fence"] = r["_in0"] == FENCE
    return rows


def cut(rows, labels):
    fences = [i for i, r in enumerate(rows) if r["_fence"]]
    # The warm fold's fences are the last len(labels) (the cold fold ran no fences, but be safe).
    if len(fences) < len(labels):
        raise SystemExit(f"{len(fences)} fence ops in the report, {len(labels)} fences enqueued")
    fences = fences[-len(labels):]
    regions = []
    for (lab, i), (lab1, j) in zip(zip(labels, fences), zip(labels[1:], fences[1:])):
        if lab.startswith("start ") and lab1 == "end " + lab[6:]:
            ops = rows[i + 1:j]
            regions.append({"label": lab[6:], "ops": ops, "next_gap": rows[j]["_gap"]})
    return regions


def summarise(reg):
    ops = reg["ops"]
    k = sum(o["_k"] for o in ops)
    fw = sum(o["_fw"] for o in ops)
    gaps = sum(o["_gap"] for o in ops[1:])
    return {"label": reg["label"], "ops": len(ops), "kernel_ms": k / 1e6, "fw_ms": fw / 1e6,
            "gap_ms": gaps / 1e6, "span_ms": (fw + gaps) / 1e6,
            "small_grid_ms": sum(o["_k"] for o in ops if o["_cores"] < 100) / 1e6,
            "gaps_over_20us": sum(1 for o in ops[1:] if o["_gap"] > 20e3),
            "gap_over_20us_ms": sum(o["_gap"] for o in ops[1:] if o["_gap"] > 20e3) / 1e6}


def kind(label):
    # pf48/b10/c320/n256 -> pf48/c320 ; denoise/M5 -> denoise/M5
    p = label.split("/")
    return p[0] if p[0].startswith("denoise") else f"{p[0]}/{p[2]}"


def detail(reg):
    ops = reg["ops"]
    lines = ["| # | op | input 0 | kernel us | cores | gap to next us |", "|---|---|---|---|---|---|"]
    for n, o in enumerate(ops):
        nxt = ops[n + 1]["_gap"] if n + 1 < len(ops) else reg["next_gap"]
        lines.append(f"| {n} | {o['OP CODE']} | {o['_in0']} | {o['_k'] / 1e3:.1f} | {o['_cores']} | {nxt / 1e3:.1f} |")
    by = defaultdict(lambda: [0, 0.0, 0.0, []])
    for n, o in enumerate(ops):
        nxt = ops[n + 1]["_gap"] if n + 1 < len(ops) else reg["next_gap"]
        b = by[o["OP CODE"]]
        b[0] += 1
        b[1] += o["_k"]
        b[2] += nxt
        b[3].append(o["_cores"])
    agg = ["| op | calls | kernel us | gap after us | cores (median) |", "|---|---|---|---|---|"]
    for op, (c, k, g, cores) in sorted(by.items(), key=lambda kv: -kv[1][1] - kv[1][2]):
        agg.append(f"| {op} | {c} | {k / 1e3:.0f} | {g / 1e3:.0f} | {int(statistics.median(cores))} |")
    return "\n".join(lines), "\n".join(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--wall", default=None)
    ap.add_argument("--detail", action="append", default=[])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = load(a.ops)
    regs = cut(rows, json.load(open(a.regions)))
    sums = [summarise(r) for r in regs]
    walls = defaultdict(list)
    if a.wall:
        for w in json.load(open(a.wall)):
            walls[kind(w["block"])].append(w["ms"])
    groups = defaultdict(list)
    for s in sums:
        groups[kind(s["label"])].append(s)
    print("| region | calls | ops | kernel ms | gaps ms | device span ms | gaps > 20 us (n / ms) | kernel on < 100 cores ms | synced wall ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    table = {}
    for g, ss in groups.items():
        med = {k: statistics.median(s[k] for s in ss) for k in ss[0] if k != "label"}
        wl = statistics.median(walls[g]) if walls.get(g) else None
        table[g] = dict(med, calls=len(ss), wall_ms=wl)
        print(f"| {g} | {len(ss)} | {med['ops']:.0f} | {med['kernel_ms']:.2f} | {med['gap_ms']:.2f} | {med['span_ms']:.2f} "
              f"| {med['gaps_over_20us']:.0f} / {med['gap_over_20us_ms']:.2f} | {med['small_grid_ms']:.2f} | "
              f"{'' if wl is None else f'{wl:.2f}'} |")
    for d in a.detail:
        hits = [r for r in regs if r["label"].startswith(d)]
        if not hits:
            print(f"\nno region matches {d}")
            continue
        reg = hits[-1]
        per, agg = detail(reg)
        print(f"\n### {reg['label']} ({len(reg['ops'])} ops)\n\n{agg}\n\n<details><summary>every op</summary>\n\n{per}\n\n</details>")
    if a.json:
        json.dump({"groups": table, "regions": sums}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
