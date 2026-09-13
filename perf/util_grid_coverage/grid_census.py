#!/usr/bin/env python3
"""Per-op grid-coverage census of the 512 aa Boltz-2 fold on Blackhole.

Reads the tracy ops reports captured by `b2z-kernel-cycle-census` (qb2 physical card 0, one
Blackhole of a p300c, 11x10 = 110 worker cores, tt-metal v0.68.0 source build with Tracy) and
answers one question per dispatched program: how many of the 110 cores did it run on.

The captures are reused rather than re-taken so the census and the published kernel-cycle numbers
come from the SAME instrument. This file reproduces `census_report.py`'s region-finding and its
median-over-reps fold, and prints that reproduction against the published totals as its own check.

Coverage is always weighted by device kernel time. An op on 4 cores that costs 8 us is not the
same shortfall as an op on 86 cores that costs 1.3 ms.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

FENCE_DIM = 32
FENCE_N = 3


def f(row, key, default=0.0):
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except ValueError:
        return default


DIMRE = re.compile(r"\s*(\d+)")


def _dim(v):
    """The report writes an extent as `512[512]` (padded[logical]); take the padded one."""
    m = DIMRE.match(str(v or ""))
    return int(m.group(1)) if m else 0


def shape(row, pfx):
    return tuple(_dim(row.get(f"{pfx}_{d}_PAD[LOGICAL]", "")) for d in ("W", "Z", "Y", "X"))


def tiles(sh):
    w, z, y, x = sh
    return max(w, 1) * max(z, 1) * ((y + 31) // 32) * ((x + 31) // 32)


def is_fence(row):
    if not row.get("OP CODE", "").startswith("Unary"):
        return False
    return shape(row, "INPUT_0")[2:] in ((FENCE_DIM, FENCE_DIM), (0, 0))


def find_region(rows):
    runs, i = [], 0
    while i < len(rows):
        if is_fence(rows[i]):
            j = i
            while j < len(rows) and is_fence(rows[j]):
                j += 1
            if j - i >= FENCE_N:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < 2:
        raise SystemExit(f"expected >=2 fence runs, found {len(runs)}")
    return rows[runs[-2][1]:runs[-1][0]]


GRID_RE = re.compile(r"compute_with_storage_grid_size=\(x=(\d+);y=(\d+)\)")
PCFG_RE = re.compile(r"'program_config': '([A-Za-z0-9]+)\((.*?)\)'")
INT_RE = {k: re.compile(k + r"=(\d+)") for k in
          ("in0_block_w", "out_subblock_h", "out_subblock_w", "out_block_h", "out_block_w",
           "per_core_M", "per_core_N", "fuse_batch", "mcast_in0", "transpose_mcast")}
END_RE = re.compile(r"'(?:slice_end|output_tensor_end|output_padded_shape|logical_output_shape)': "
                    r"'Shape\(\[([^\]]*)\]\)")
MC_RE = re.compile(r"'use_multicore': '(\w+)'")
HEADS_RE = re.compile(r"'num_q_heads': '(\d+)'")
DIM_RE = re.compile(r"'dim': '(\d+)'")


def attrs(row):
    """The ATTRIBUTES fields that say WHY a program took the grid it took."""
    a = row.get("ATTRIBUTES", "") or ""
    out = {}
    m = PCFG_RE.search(a)
    if m:
        out["program_config"] = m.group(1)
        body = m.group(2)
        mg = GRID_RE.search(body)
        if mg:
            out["cfg_grid"] = f"{mg.group(1)}x{mg.group(2)}"
        for k, rx in INT_RE.items():
            mm = rx.search(body)
            if mm:
                out[k] = int(mm.group(1))
    m = END_RE.search(a)
    if m:
        out["shape_hint"] = m.group(1).replace("; ", "x")
    for key, rx in (("use_multicore", MC_RE), ("num_q_heads", HEADS_RE), ("dim", DIM_RE)):
        m = rx.search(a)
        if m:
            out[key] = m.group(1)
    return out


def load(csv_path: Path, reps: int):
    op = gzip.open if csv_path.name.endswith(".gz") else open
    with op(csv_path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    region = find_region(rows)
    if len(region) % reps:
        raise SystemExit(f"{len(region)} ops is not divisible by {reps} reps")
    n = len(region) // reps
    chunks = [region[i * n:(i + 1) * n] for i in range(reps)]
    ratios = [f(r, "DEVICE FW DURATION [ns]") /
              (f(r, "DEVICE FW END CYCLE") - f(r, "DEVICE FW START CYCLE"))
              for r in region if f(r, "DEVICE FW END CYCLE") > f(r, "DEVICE FW START CYCLE")]
    ns_per_cycle = st.median(ratios) if ratios else 1.0

    progs = []
    for i in range(n):
        rs = [ch[i] for ch in chunks]
        r0 = rs[0]
        progs.append({
            "i": i,
            "op": r0.get("OP CODE", "?"),
            "cores": int(f(r0, "CORE COUNT")),
            "avail": int(f(r0, "AVAILABLE WORKER CORE COUNT", 110)) or 110,
            "kernel_ns": st.median([f(r, "DEVICE KERNEL DURATION [ns]") for r in rs]),
            "gap_ns": st.median([f(r, "OP TO OP LATENCY [ns]") for r in rs]),
            "trisc1_ns": st.median([f(r, "DEVICE TRISC1 KERNEL DURATION [ns]") for r in rs]),
            "cb_wait_ns": st.median([f(r, "DEVICE COMPUTE CB WAIT FRONT [ns]") for r in rs]),
            "fidelity": r0.get("MATH FIDELITY", ""),
            "in0": shape(r0, "INPUT_0"), "in1": shape(r0, "INPUT_1"),
            "out0": shape(r0, "OUTPUT_0"),
            "in0_dtype": r0.get("INPUT_0_DATATYPE", ""),
            "in1_dtype": r0.get("INPUT_1_DATATYPE", ""),
            "out_dtype": r0.get("OUTPUT_0_DATATYPE", ""),
            "in0_mem": (r0.get("INPUT_0_MEMORY", "") or "").replace("dev_", ""),
            "out_mem": (r0.get("OUTPUT_0_MEMORY", "") or "").replace("dev_", ""),
            **attrs(r0),
        })
    spans = []
    for ch in chunks:
        spans.append((f(ch[-1], "DEVICE FW END CYCLE") -
                      f(ch[0], "DEVICE FW START CYCLE")) * ns_per_cycle / 1e6)
    return progs, st.median(spans)


def bucket(p):
    """Group programs into the census's op classes.

    The class is the op code plus the grid it actually took, because two programs of the same op
    code on different grids are two different shortfalls with two different reasons. The matmul
    program config's own grid is carried alongside so a shortfall can be attributed to the config
    rather than to the op.
    """
    return (p["op"], p["cores"], p.get("program_config", ""),
            p["in0"], p["in1"], p["out0"], p.get("fidelity", ""))


def census(progs, label, calls_per_fold, span_ms):
    total_kernel = sum(p["kernel_ns"] for p in progs)
    avail = max(p["avail"] for p in progs)
    by = defaultdict(list)
    for p in progs:
        by[bucket(p)].append(p)
    rows = []
    for (op, cores, pcfg, in0, in1, out0, fid), ps in by.items():
        k = sum(p["kernel_ns"] for p in ps)
        rows.append({
            "op": op, "cores": cores, "avail": avail,
            "cfg_grid": ps[0].get("cfg_grid", ""),
            "fuse_batch": ps[0].get("fuse_batch"),
            "in0": in0, "in1": in1, "out0": out0, "fidelity": fid,
            "Mt": tiles((1, 1, out0[2], 32)) if out0[2] else 0,
            "Nt": (out0[3] + 31) // 32, "Kt": (in0[3] + 31) // 32,
            "batch": max(out0[0], 1) * max(out0[1], 1),
            "in0_dtype": ps[0].get("in0_dtype", ""), "out_dtype": ps[0].get("out_dtype", ""),
            "in0_mem": ps[0].get("in0_mem", ""), "out_mem": ps[0].get("out_mem", ""),
            "programs": len(ps),
            "kernel_ms": k / 1e6,
            "pct_of_kernel": 100.0 * k / total_kernel,
            "idle_core_ms": k / 1e6 * (avail - cores) / avail,
            "s_per_fold": k / 1e6 * calls_per_fold / 1e3,
            "program_config": pcfg,
            "per_core_M": ps[0].get("per_core_M"),
            "per_core_N": ps[0].get("per_core_N"),
            "shape_hint": ps[0].get("shape_hint", ""),
            "use_multicore": ps[0].get("use_multicore", ""),
            "median_prog_us": st.median([p["kernel_ns"] for p in ps]) / 1e3,
        })
    rows.sort(key=lambda r: -r["idle_core_ms"])
    weighted_cores = sum(p["kernel_ns"] * p["cores"] for p in progs) / total_kernel
    return {
        "label": label,
        "programs": len(progs),
        "calls_per_fold": calls_per_fold,
        "span_ms": span_ms,
        "kernel_ms": total_kernel / 1e6,
        "gap_ms": span_ms - total_kernel / 1e6,
        "gap_pct": 100.0 * (span_ms - total_kernel / 1e6) / span_ms,
        "avail_cores": avail,
        "weighted_cores": weighted_cores,
        "coverage_pct": 100.0 * weighted_cores / avail,
        "s_per_fold": total_kernel / 1e6 * calls_per_fold / 1e3,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.dir / "census_512_qb2.json"

    res = {}
    for label, fn, reps, calls in (
            ("pairformer block", "ops_perf_block_qb2c0.csv.gz", 3, 264),
            ("diffusion step", "ops_perf_step_qb2c0.csv.gz", 3, 200)):
        progs, span = load(a.dir / fn, reps)
        res[label] = census(progs, label, calls, span)
        res[label]["per_program"] = progs

    out.write_text(json.dumps(res, indent=1))
    for label, c in res.items():
        print(f"\n=== {label}: {c['programs']} programs, span {c['span_ms']:.4f} ms, "
              f"kernel {c['kernel_ms']:.4f} ms, gap {c['gap_pct']:.1f} %")
        print(f"    time-weighted coverage {c['weighted_cores']:.1f} of {c['avail_cores']} cores "
              f"= {c['coverage_pct']:.1f} %; {c['s_per_fold']:.3f} s/fold of kernel time")
        print(f"    {'op':<30} {'cores':>5} {'progs':>5} {'ms':>9} {'%kern':>6} "
              f"{'idlecore':>9} {'s/fold':>7}  {'out (B,Mt,Nt,Kt)':<22} {'cfg':<14} "
              f"{'pcM':>4} {'pcN':>4} {'fb':>3}")
        for r in c["rows"][:22]:
            print(f"    {r['op'].replace('DeviceOperation',''):<30} {r['cores']:>5} "
                  f"{r['programs']:>5} {r['kernel_ms']:>9.4f} "
                  f"{r['pct_of_kernel']:>6.2f} {r['idle_core_ms']:>9.4f} {r['s_per_fold']:>7.3f}  "
                  f"{f'{r["batch"]}x{r["Mt"]}x{r["Nt"]}x{r["Kt"]}':<22} "
                  f"{r['program_config'].replace('ProgramConfig','').replace('MatmulMultiCoreReuse','mm'):<14} "
                  f"{str(r['per_core_M'] if r['per_core_M'] is not None else ''):>4} "
                  f"{str(r['per_core_N'] if r['per_core_N'] is not None else ''):>4} "
                  f"{str(r['fuse_batch'] if r['fuse_batch'] is not None else ''):>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
