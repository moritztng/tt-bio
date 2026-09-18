#!/usr/bin/env python3
"""CPU-only screen of the fourteen device op classes nobody has priced.

`c12-profiled-fold` composed 13.2090 s of in-situ device time out of the 14.8810 s fold. Every C12
row since has worked the top five classes (Matmul, GenericOp, BinaryNg, LayerNorm, Transpose =
11.6417 s). The other 1.5672 s -- SDPA, NlpCreateHeads and a bucket documented in every campaign
doc as "twelve smaller 0.8330 s" -- has never been itemised, let alone priced.

This script itemises it from the SAME artifacts that produced 13.2090 s: the six fenced profiler
windows on `origin/wk/c12-profiled-fold`. It reads them through `git show` at a pinned SHA rather
than copying 9 MB of CSV onto a second branch, so the input is the committed artifact and is named
as such in the output.

Every shape it prints is the shape tt-metal's own ops report recorded for an executed program.
That distinction is load-bearing: a census KEY LABEL like `[1,16,512,128]` ships at no size and
inflated a call count 2.4-2.9x on two earlier C12 rows by splitting one call into 16 sub-calls.

Two controls, both printed:
  KA1  per-unit device ms/call and programs/call must reproduce `runs/composed.json` exactly.
       If the fence window this script finds is not the window `reduce.py` found, every second
       below is wrong and this control says so before any table is read.
  KA2  the fourteen classes must sum to 1.5672 s and the whole table to 13.2090 s.

Roofs are the ones measured on the same part in the same session as the campaign's other rows,
`perf/c12_genop_rate/runs/roofs_qb2c3_r40.json` (qb2 card 3, 110 cores, 1350 MHz):
  bw_add8192   2R+1W DRAM   435.73 GB/s
  bw_clone     1R+1W DRAM   393.31 GB/s     (1.108x apart -- one roof cannot serve both mixes)
  cube4096     HiFi2 bf16   113.68 TFLOP/s
so machine balance is 113.68e12 / 435.73e9 = 260.9 FLOP/byte. An op below that is traffic-bound
and is priced on bytes; an op above it is arithmetic-bound and is priced on FLOP.

DRAM bytes and L1 bytes are counted SEPARATELY, per operand, from each operand's own recorded
memory config. An operand already in L1 costs no DRAM traffic, and pricing it as if it did is how
`insitu_sites.py:47` put one site at 37.99 % of roof when it was at 75.96 %.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PF_REF = "a63d6d8e3749843c1c58814cbabf87cd19a3bc17"   # origin/wk/c12-profiled-fold, pass-3 tip
PF_DIR = "perf/c12_profiled_fold/runs"

# unit -> (run dir, index within that run's `unit_order`, reps, calls per fold)
# calls per fold and the composed figures both come from runs/composed.json; the reps and the
# multi-unit ordering from each run's own unit.json.
UNITS = [
    ("PairformerLayer",        "pfl_prof2", 0, 3,  264),
    ("MSALayer",               "msal_prof", 0, 5,   16),
    ("DiffusionModule",        "dm_prof",   0, 3,  200),
    ("PairConditioningDevice", "stage_prof", 0, 5,   1),
    ("RelPosGather",           "stage_prof", 1, 10,  2),
    ("PairAssemblyDevice",     "stage_prof", 2, 5,   2),
]
N_UNITS_IN_RUN = {"pfl_prof2": 1, "msal_prof": 1, "dm_prof": 1, "stage_prof": 3}

# The five classes every C12 row since c12-profiled-fold has worked. Everything else is this
# screen's scope.
WORKED = ("MatmulDeviceOperation", "GenericOpDeviceOperation", "BinaryNgDeviceOperation",
          "LayerNormDeviceOperation", "TransposeDeviceOperation")

CLOCK_MHZ = 1350.0
FOLD_S = 14.8810
DEVICE_S = 13.2090
BW_2R1W = 435.7261026435504e9      # bw_add8192,  roofs_qb2c3_r40.json
BW_1R1W = 393.3089503261313e9      # bw_clone,    roofs_qb2c3_r40.json
CUBE_TFLOPS = 113.68334140686571e12
BALANCE = CUBE_TFLOPS / BW_2R1W    # 260.9 FLOP/byte

ITEMSIZE = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1.0625, "UINT32": 4, "INT32": 4,
            "UINT16": 2, "UINT8": 1, "BFLOAT4_B": 0.5625}

# ---------------------------------------------------------------------------------------------
# The traffic model, one entry per device op code, with the read set and the write set NAMED.
#
# A blanket "charge every recorded operand" rule is wrong and the first run of this screen proved
# it: it put SliceDeviceOperation at 1336.2 GB/s, 3.4x the part's DRAM roof, because it charged the
# whole [1,512,512,128] input for a slice that only reads the 47 rows it returns. A rate above the
# roof is not a fast op, it is a byte model that counted bytes the kernel never moved.
#
# So each op is charged from its own semantics, and `covers` reports the charged bytes as a
# fraction of ALL recorded operand bytes so that nothing is dropped silently. `insitu_sites.py:47`
# dropped a `repair_B` field and put one site at 37.99 % of roof when it was at 75.96 %; the
# fraction is printed for every op here for exactly that reason.
#
#   "all"      every recorded operand, in and out          (the op genuinely streams all of them)
#   "out2"     twice the output extent                     (slice: reads only what it returns)
#   "io"       inputs + outputs, equal volume              (1R+1W layout moves)
#   "write"    outputs only, plus any index operand        (gather: the table is L1-resident)
#   None       not modelled -- reported UNPRICED, never priced against a roof
#
# `mix` picks the roof: a 1R+1W move is priced on bw_clone (393.31 GB/s) and anything read-heavier
# on bw_add8192 (435.73 GB/s). The two differ 1.108x on this part, so one roof cannot serve both.
TRAFFIC = {
    "SDPAOperation":                 ("sdpa", "2R1W", "Q, the bias and O once; K and V ONCE PER "
                                      "Q CHUNK, with the chunk count taken from the q_chunk_size "
                                      "in the row's own executed program_config. Charging every "
                                      "operand once understates the token site by 1.50x and put "
                                      "it at 37.4 % of roof when it is at 56.0 %"),
    "NlpCreateHeadsDeviceOperation": ("io",   "1R1W", "packed qkv read, three head-major views "
                                      "written; equal volume, zero FLOP"),
    "SliceDeviceOperation":          ("out2", "1R1W", "reads only the extent it returns"),
    "ReshapeViewDeviceOperation":    ("io",   "1R1W", "equal volume in and out"),
    "PadDeviceOperation":            ("io",   "1R1W", "input read, larger padded output written"),
    "PermuteDeviceOperation":        ("io",   "1R1W", "equal volume in and out"),
    "ConcatDeviceOperation":         ("io",   "1R1W", "every input read once, output written once"),
    "UntilizeDeviceOperation":       ("io",   "1R1W", "equal volume, layout only"),
    "TilizeDeviceOperation":         ("io",   "1R1W", "equal volume, layout only"),
    "EmbeddingsDeviceOperation":     ("write", "1R1W", "index tensor read and output written; the "
                                      "weight table is 2-66 rows x 128 and is L1-resident, so its "
                                      "reads are not DRAM traffic"),
    "SoftmaxDeviceOperation":        ("io",   "1R1W", "equal volume in and out"),
    "CopyDeviceOperation":           ("io",   "1R1W", "L1<->DRAM copy; only the DRAM side is "
                                      "charged, by operand location"),
    "NLPConcatHeadsDeviceOperation": ("io",   "1R1W", "equal volume, layout only"),
    "UnaryNgDeviceOperation":        (None,   None,   "UNPRICED. Three of this class's 202 "
                                      "programs carry a [1,512,512,128] bf16 operand (67.1 MB) "
                                      "and complete in 2.2-3.0 us on 110 cores, 70x faster than "
                                      "the part's DRAM roof allows for that volume, while the "
                                      "same op on the same shape in PairAssemblyDevice takes "
                                      "485 us (276 GB/s, plausible). Either the shape or the "
                                      "duration is wrong on those rows, so this class is reported "
                                      "and not priced. It is 0.0015 s of 13.2090 s"),
}

# Zero-FLOP classes: the traffic-vs-arithmetic question resolves by inspection, no balance number
# needed. Only SDPA and Softmax do any arithmetic at all.
ZERO_FLOP = tuple(k for k in TRAFFIC if k not in ("SDPAOperation", "SoftmaxDeviceOperation"))


_QC = re.compile(r"q_chunk_size=(\d+)")


def q_chunks(r):
    """K/V re-read factor: one pass over K and V per q chunk, from the EXECUTED program config."""
    m = _QC.search(r.get("ATTRIBUTES") or "")
    q = shape(r, "INPUT_0")
    if not m or not q:
        return 1
    qc = int(m.group(1))
    return max(1, math.ceil(q[2] / qc))


def charged(r, how):
    """(dram_B, l1_B, all_recorded_B) under the named traffic model."""
    ops = operands(r)
    allB = sum(o[4] for o in ops)
    if how is None:
        return 0.0, 0.0, allB
    if how == "sdpa":
        n = q_chunks(r)
        d = l = 0.0
        for slot, _sh, _dt, mem, nb in ops:
            w = n if slot in ("INPUT_1", "INPUT_2") else 1      # K and V per q chunk
            if loc(mem) == "DRAM":
                d += nb * w
            elif loc(mem) == "L1":
                l += nb * w
        return d, l, allB
    if how == "all":
        sel = ops
    elif how == "out2":
        outs = [o for o in ops if o[0].startswith("OUTPUT")]
        sel = outs + outs                       # read the extent, write the extent
    elif how == "io":
        sel = ops
    elif how == "write":
        sel = [o for o in ops if o[0].startswith("OUTPUT")
               or o[2] in ("UINT32", "INT32", "UINT16")]
    else:
        raise ValueError(how)
    d = sum(o[4] for o in sel if loc(o[3]) == "DRAM")
    l = sum(o[4] for o in sel if loc(o[3]) == "L1")
    return d, l, allB


def flops(r):
    """Arithmetic, for the two classes that do any. Everything else is a pure data move."""
    code = r["OP CODE"]
    if code == "SDPAOperation":
        q = shape(r, "INPUT_0")
        k = shape(r, "INPUT_1")
        if not q or not k:
            return 0.0
        b, h, sq, d = q
        sk = k[2]
        return 4.0 * b * h * sq * sk * d            # QK^T and AV, 2 FLOP per MAC each
    if code == "SoftmaxDeviceOperation":
        s = shape(r, "INPUT_0")
        if not s:
            return 0.0
        return 5.0 * s[0] * s[1] * s[2] * s[3]      # max, subtract, exp, sum, divide
    return 0.0


FENCE_OP = "UnaryDeviceOperation"
FENCE_N = 3


def gitshow(repo: Path, ref: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "show", "%s:%s" % (ref, path)],
                          check=True, stdout=subprocess.PIPE).stdout


def load_csv(repo: Path, ref: str, run: str):
    import gzip
    raw = gitshow(repo, ref, "%s/%s/ops_perf_results.csv.gz" % (PF_DIR, run))
    return list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode())))


def dim(r, slot, ax, logical=False):
    v = str(r.get("%s_%s_PAD[LOGICAL]" % (slot, ax), "") or "").strip()
    if logical and "[" in v:
        v = v[v.index("[") + 1:]
    n = ""
    for c in v:
        if c.isdigit():
            n += c
        else:
            break
    return int(n) if n else 0


def shape(r, slot, logical=False):
    t = tuple(dim(r, slot, a, logical) for a in "WZYX")
    return t if (t[2] and t[3]) else None


def is_fence(r, d=32):
    s = shape(r, "INPUT_0")
    return (r.get("OP CODE") == FENCE_OP and s is not None
            and s[2] == d and s[3] == d and s[0] <= 1 and s[1] <= 1)


def fence_runs(rows):
    out, i = [], 0
    while i < len(rows):
        if is_fence(rows[i]):
            j = i
            while j < len(rows) and is_fence(rows[j]):
                j += 1
            if j - i >= FENCE_N:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def window(rows, k, n_units):
    runs = fence_runs(rows)
    n_fence = sum(1 for r in rows if is_fence(r))
    if len(runs) < 2 * (k + 1):
        raise SystemExit("run has %d fence runs, need %d" % (len(runs), 2 * (k + 1)))
    if n_fence != 2 * FENCE_N * n_units:
        raise SystemExit("found %d fence rows, expected %d" % (n_fence, 2 * FENCE_N * n_units))
    a, b = runs[2 * k][1], runs[2 * k + 1][0]
    return rows[a:b]


def ns(r):
    try:
        return float(r.get("DEVICE KERNEL DURATION [ns]") or 0)
    except ValueError:
        return 0.0


def operands(r):
    """(slot, shape, dtype, memory, bytes) for every recorded operand, in and out."""
    out = []
    for kind in ("INPUT", "OUTPUT"):
        for i in range(12):
            slot = "%s_%d" % (kind, i)
            s = shape(r, slot)
            if s is None:
                continue
            dt = (r.get(slot + "_DATATYPE") or "").strip()
            mem = (r.get(slot + "_MEMORY") or "").strip()
            n = s[0] * s[1] * s[2] * s[3]
            out.append((slot, s, dt, mem, n * ITEMSIZE.get(dt, 2)))
    return out


def loc(mem):
    if "DRAM" in mem:
        return "DRAM"
    if "L1" in mem:
        return "L1"
    return "?"


def sig(r):
    """The executed signature: op code + every operand shape/dtype/location + core count."""
    ops = operands(r)
    return (r["OP CODE"],
            tuple((o[0], o[1], o[2], loc(o[3])) for o in ops),
            int(float(r.get("CORE COUNT") or 0)),
            (r.get("ATTRIBUTES") or "")[:0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--ref", default=PF_REF)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    composed = json.loads(gitshow(a.repo, a.ref, "%s/composed.json" % PF_DIR))
    want = {u["unit"]: u for u in composed["units"]}

    cache: dict = {}
    per_unit = []
    agg: dict = defaultdict(lambda: {"s": 0.0, "programs": 0.0, "dram_B": 0.0, "l1_B": 0.0,
                                     "all_B": 0.0, "flop": 0.0, "cores": Counter(),
                                     "sigs": defaultdict(lambda: {"n": 0.0, "s": 0.0,
                                                                  "dram_B": 0.0, "l1_B": 0.0,
                                                                  "flop": 0.0,
                                                                  "units": set()})})
    for name, run, k, reps, calls in UNITS:
        if run not in cache:
            cache[run] = load_csv(a.repo, a.ref, run)
        win = window(cache[run], k, N_UNITS_IN_RUN[run])
        dev_ms_per_call = sum(ns(r) for r in win) / 1e6 / reps
        progs_per_call = len(win) / reps
        w = want[name]
        per_unit.append({
            "unit": name, "run": run, "reps": reps, "calls_per_fold": calls,
            "device_ms_per_call": round(dev_ms_per_call, 5),
            "programs_per_call": round(progs_per_call, 2),
            "s_per_fold": round(dev_ms_per_call * calls / 1e3, 4),
            "KA1_device_ms_matches_composed": abs(dev_ms_per_call - w["device_ms_per_call"]) < 5e-5,
            "KA1_programs_matches_composed": abs(progs_per_call - w["programs_per_call"]) < 1e-6,
        })
        scale = calls / reps                     # window rows -> programs per fold
        for r in win:
            code = r["OP CODE"]
            e = agg[code]
            how = TRAFFIC.get(code, ("all", "2R1W", ""))[0] if code in TRAFFIC else "all"
            d, l, allB = charged(r, how) if code in TRAFFIC else (0.0, 0.0, 0.0)
            f = flops(r)
            e["s"] += ns(r) / 1e9 * scale
            e["programs"] += scale
            e["dram_B"] += d * scale
            e["l1_B"] += l * scale
            e["all_B"] += allB * scale
            e["flop"] += f * scale
            e["cores"][int(float(r.get("CORE COUNT") or 0))] += scale
            g = e["sigs"][sig(r)]
            g["n"] += scale
            g["s"] += ns(r) / 1e9 * scale
            g["dram_B"] += d * scale
            g["l1_B"] += l * scale
            g["flop"] += f * scale
            g["units"].add(name)

    total_s = sum(v["s"] for v in agg.values())
    tail = {k: v for k, v in agg.items() if k not in WORKED}
    tail_s = sum(v["s"] for v in tail.values())

    print("SOURCE  origin/wk/c12-profiled-fold @ %s : %s/{%s}/ops_perf_results.csv.gz"
          % (a.ref[:9], PF_DIR, ",".join(sorted({u[1] for u in UNITS}))))
    print("ROOFS   perf/c12_genop_rate/runs/roofs_qb2c3_r40.json -- bw_add8192 %.2f GB/s (2R+1W), "
          "bw_clone %.2f GB/s (1R+1W), cube4096 %.2f TFLOP/s"
          % (BW_2R1W / 1e9, BW_1R1W / 1e9, CUBE_TFLOPS / 1e12))
    print("CLOCK   %.0f MHz, pinned and sampled inside every profiled region by the source row\n"
          % CLOCK_MHZ)

    print("KA1 -- fence windows reproduce runs/composed.json")
    print("%-24s %8s %10s %12s %8s %6s" % ("unit", "calls", "ms/call", "composed", "progs", "ok"))
    ok1 = True
    for u in per_unit:
        w = want[u["unit"]]
        good = u["KA1_device_ms_matches_composed"] and u["KA1_programs_matches_composed"]
        ok1 &= good
        print("%-24s %8d %10.5f %12.5f %8.1f %6s"
              % (u["unit"], u["calls_per_fold"], u["device_ms_per_call"],
                 w["device_ms_per_call"], u["programs_per_call"], "OK" if good else "MISMATCH"))
    print("KA1: %s" % ("all six units reproduce" if ok1 else "FAILED"))
    print("KA2 -- table %.4f s vs composed %.4f s (%+.4f s); tail %.4f s vs 1.5672 s (%+.4f s)\n"
          % (total_s, DEVICE_S, total_s - DEVICE_S, tail_s, tail_s - 1.5672))

    print("THE FOURTEEN. `floor` is DRAM bytes / the mix-matched roof; `%roof` = floor / measured.")
    print("`AI` is FLOP per charged DRAM byte against the measured %.1f FLOP/byte balance."
          % BALANCE)
    print("%-30s %8s %8s %8s %8s %8s %7s %7s %7s %6s"
          % ("device op", "s/fold", "progs", "Mcyc", "DRAM GB", "floor s", "%roof", "GB/s",
             "AI", "cov%"))
    rows_out = []
    for code, v in sorted(tail.items(), key=lambda kv: -kv[1]["s"]):
        how, mix, why = TRAFFIC.get(code, (None, None, "no traffic model"))
        roof = BW_2R1W if mix == "2R1W" else BW_1R1W
        if how is None:
            print("%-30s %8.4f %8.0f %8.1f %8s %8s %7s %7s %7s %6s"
                  % (code, v["s"], v["programs"], v["s"] * CLOCK_MHZ,
                     "-", "UNPRICED", "-", "-", "-", "-"))
            rows_out.append({"op": code, "s_per_fold": round(v["s"], 5),
                             "programs_per_fold": round(v["programs"], 1),
                             "Mcycles": round(v["s"] * CLOCK_MHZ, 1),
                             "priced": False, "why": why})
            continue
        floor = v["dram_B"] / roof
        pct = 100 * floor / v["s"] if v["s"] else 0.0
        ai = v["flop"] / v["dram_B"] if v["dram_B"] else 0.0
        cov = 100 * (v["dram_B"] + v["l1_B"]) / v["all_B"] if v["all_B"] else 0.0
        print("%-30s %8.4f %8.0f %8.1f %8.2f %8.4f %7.1f %7.1f %7.1f %6.0f"
              % (code, v["s"], v["programs"], v["s"] * CLOCK_MHZ, v["dram_B"] / 1e9, floor,
                 pct, v["dram_B"] / v["s"] / 1e9, ai, cov))
        rows_out.append({
            "op": code, "s_per_fold": round(v["s"], 5),
            "programs_per_fold": round(v["programs"], 1),
            "Mcycles": round(v["s"] * CLOCK_MHZ, 1), "priced": True,
            "traffic_model": how, "mix": mix, "model_note": why,
            "dram_GB": round(v["dram_B"] / 1e9, 4), "l1_GB": round(v["l1_B"] / 1e9, 4),
            "recorded_operand_GB": round(v["all_B"] / 1e9, 4),
            "charged_frac_of_recorded_pct": round(cov, 1),
            "roof_GB_s": round(roof / 1e9, 2), "floor_s": round(floor, 5),
            "pct_of_roof": round(pct, 2), "GB_s": round(v["dram_B"] / v["s"] / 1e9, 2),
            "TFLOP_s": round(v["flop"] / v["s"] / 1e12, 4),
            "AI_FLOP_per_B": round(ai, 2),
            "bound": "traffic" if ai < BALANCE else "arithmetic",
            "recoverable_to_roof_s": round(v["s"] - floor, 5),
            "cores": dict(sorted(((c, round(n, 1)) for c, n in v["cores"].items()),
                                 key=lambda kv: -kv[1])),
        })
    print("%-30s %8.4f %8.0f %8.1f" % ("TAIL TOTAL", tail_s,
                                       sum(v["programs"] for v in tail.values()),
                                       tail_s * CLOCK_MHZ))
    priced = [r for r in rows_out if r["priced"]]
    print("\nIf every priced class reached its own mix-matched roof: %.4f s of %.4f s, "
          "i.e. %.4f s recoverable (%.1f Mcycles). Nothing on this part has ever reached its roof; "
          "the best of generic_op's six sites reaches 79.39 %%."
          % (sum(r["floor_s"] for r in priced), sum(r["s_per_fold"] for r in priced),
             sum(r["recoverable_to_roof_s"] for r in priced),
             CLOCK_MHZ * sum(r["recoverable_to_roof_s"] for r in priced)))
    print("BOUND: %d of %d classes do ZERO arithmetic, so they are traffic-bound by inspection. "
          "The two that compute: %s."
          % (sum(1 for r in priced if r["AI_FLOP_per_B"] == 0), len(priced),
             ", ".join("%s AI %.1f FLOP/B -> %s" % (r["op"], r["AI_FLOP_per_B"], r["bound"])
                       for r in priced if r["AI_FLOP_per_B"] > 0)))

    print("\nEXECUTED SHAPES, every signature over 0.005 s/fold")
    sigs_out = []
    for code, v in sorted(tail.items(), key=lambda kv: -kv[1]["s"]):
        how, mix, _why = TRAFFIC.get(code, (None, None, ""))
        roof = BW_2R1W if mix == "2R1W" else BW_1R1W
        gs = sorted(v["sigs"].items(), key=lambda kv: -kv[1]["s"])
        print("\n  %s -- %.4f s, %.0f programs, %d distinct executed signatures"
              % (code, v["s"], v["programs"], len(gs)))
        for sg, g in gs:
            if g["s"] < 0.005:
                continue
            opnd = " ".join("%s%s%s/%s" % (sl.replace("INPUT_", "i").replace("OUTPUT_", "o"),
                                           list(sh), dt.replace("BFLOAT16", "bf16")
                                           .replace("FLOAT32", "fp32").replace("BFLOAT8_B", "bfp8"),
                                           lc) for sl, sh, dt, lc in sg[1])
            fl = g["dram_B"] / roof if how else 0.0
            print("    %8.4f s %7.0f prog %3d core  %5.1f%% of roof  %s  [%s]"
                  % (g["s"], g["n"], sg[2], 100 * fl / g["s"] if (how and g["s"]) else 0.0,
                     opnd, ",".join(sorted(g["units"]))))
            sigs_out.append({"op": code, "s_per_fold": round(g["s"], 5),
                             "programs_per_fold": round(g["n"], 1), "cores": sg[2],
                             "dram_GB": round(g["dram_B"] / 1e9, 4),
                             "floor_s": round(fl, 5),
                             "pct_of_roof": round(100 * fl / g["s"], 2) if (how and g["s"]) else None,
                             "AI_FLOP_per_B": round(g["flop"] / g["dram_B"], 2)
                             if g["dram_B"] else 0.0,
                             "operands": [[sl, list(sh), dt, lc] for sl, sh, dt, lc in sg[1]],
                             "units": sorted(g["units"])})

    print("\nWORKED FIVE, for the non-overlap audit -- this screen claims nothing inside these")
    for code in WORKED:
        v = agg[code]
        print("  %-30s %9.4f s %10.0f programs" % (code, v["s"], v["programs"]))

    if a.out:
        a.out.write_text(json.dumps({
            "source": {"ref": a.ref, "dir": PF_DIR, "runs": sorted({u[1] for u in UNITS})},
            "clock_MHz": CLOCK_MHZ, "fold_s": FOLD_S, "device_s": DEVICE_S,
            "roofs": {"bw_add8192_GB_s": BW_2R1W / 1e9, "bw_clone_GB_s": BW_1R1W / 1e9,
                      "cube4096_TFLOP_s": CUBE_TFLOPS / 1e12, "balance_FLOP_B": BALANCE,
                      "from": "perf/c12_genop_rate/runs/roofs_qb2c3_r40.json"},
            "controls": {"KA1_all_units_reproduce_composed": ok1,
                         "KA2_table_total_s": round(total_s, 5),
                         "KA2_tail_total_s": round(tail_s, 5)},
            "per_unit": per_unit, "tail": rows_out, "signatures": sigs_out,
            "worked_five": {c: {"s_per_fold": round(agg[c]["s"], 5),
                                "programs_per_fold": round(agg[c]["programs"], 1)}
                            for c in WORKED},
        }, indent=1))
        print("\nwrote", a.out)
    return 0 if ok1 else 1


if __name__ == "__main__":
    sys.exit(main())
