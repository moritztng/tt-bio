"""Turn bc4_census.py's JAX profiler trace into the numbers state/bc4/CENSUS-GPU.md quotes: device-busy fraction
(so, host-bound or not), forward vs backward, time by op family, and the achieved share of the card's own roofs.

    python3 bc4_census_parse.py <census dir> [--json out.json]

The trace is the standard trace_viewer JSON (plugins/profile/<run>/*.trace.json.gz). Device work is every event on a
GPU "XLA Ops" lane; backward is an op whose XLA metadata names a transposed jvp (how JAX names reverse-mode ops).
Where the trace carries XLA's per-op bytes_accessed and flops stats, they give achieved GB/s and TFLOP/s against the
roofs bc4_census.py measured on the same card in the same process; where it does not, this says so instead of guessing.
"""
import argparse, collections, gzip, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dir", type=Path)
ap.add_argument("--json", type=Path)
a = ap.parse_args()

census = json.loads((a.dir / "census.json").read_text()) if (a.dir / "census.json").exists() else {}
traces = sorted(a.dir.rglob("*.trace.json.gz")) + sorted(a.dir.rglob("*.trace.json"))
if not traces:
    sys.exit(f"no trace under {a.dir}")
raw = (gzip.open if traces[0].suffix == ".gz" else open)(traces[0], "rt").read()
ev = json.loads(raw)
events, pids = ev["traceEvents"], {}

names = {}  # pid/tid -> lane name
for e in events:
    if e.get("ph") == "M" and e.get("name") in ("process_name", "thread_name"):
        key = (e["pid"], e.get("tid")) if e["name"] == "thread_name" else (e["pid"], None)
        names[key] = e["args"]["name"]

BACK = re.compile(r"transpose\(jvp|/backward|_bwd|grad\(")
dev_lane = lambda e: "XLA Ops" in (names.get((e["pid"], e.get("tid"))) or "")
host_lane = lambda e: "XLA Modules" in (names.get((e["pid"], e.get("tid"))) or "")

by_family, split, total_us, nops = collections.Counter(), {"fwd": 0.0, "bwd": 0.0}, 0.0, 0
bytes_acc, flops_acc, with_stats = 0.0, 0.0, 0
span = [float("inf"), 0.0]
for e in events:
    if e.get("ph") != "X" or not dev_lane(e):
        continue
    d = float(e.get("dur", 0))
    total_us += d
    nops += 1
    span[0], span[1] = min(span[0], e["ts"]), max(span[1], e["ts"] + d)
    args = e.get("args") or {}
    meta = f"{args.get('long_name', '')} {args.get('op_name', '')} {e.get('name', '')}"
    split["bwd" if BACK.search(meta) else "fwd"] += d
    fam = (args.get("hlo_category") or e.get("name", "?").split(".")[0])
    by_family[fam] += d
    b, f = args.get("bytes_accessed"), args.get("flops")
    if b or f:
        with_stats += 1
        bytes_acc += float(b or 0)
        flops_acc += float(f or 0)

wall_us = span[1] - span[0] if span[1] else 0
out = dict(trace=str(traces[0]), device_ops=nops, device_busy_us=round(total_us, 1), window_us=round(wall_us, 1),
           device_busy_frac=round(total_us / wall_us, 4) if wall_us else None,
           backward_frac=round(split["bwd"] / total_us, 4) if total_us else None,
           by_family={k: round(v / total_us, 4) for k, v in by_family.most_common(15)} if total_us else {},
           ops_with_xla_stats=with_stats, roofs=census.get("roofs"),
           trajectories=census.get("trajectories"))
if with_stats and wall_us:
    out["achieved_gbs"] = round(bytes_acc / (wall_us * 1e-6) / 1e9, 1)
    out["achieved_tflops"] = round(flops_acc / (wall_us * 1e-6) / 1e12, 2)
    if census.get("roofs", {}).get("hbm_copy_gbs"):
        out["hbm_roof_frac"] = round(out["achieved_gbs"] / census["roofs"]["hbm_copy_gbs"], 3)
    if census.get("roofs", {}).get("bf16_matmul_tflops"):
        out["compute_roof_frac"] = round(out["achieved_tflops"] / census["roofs"]["bf16_matmul_tflops"], 3)
print(json.dumps(out, indent=1))
if a.json:
    a.json.write_text(json.dumps(out, indent=1) + "\n")
