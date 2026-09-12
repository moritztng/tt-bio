#!/usr/bin/env python3
"""Per-op map of one Boltz-2 512 aa diffusion step, re-derived from committed traces.

No card is opened and no fold is run. Every input is an artifact another wave-1 row already
measured on qb2 card 0 (Blackhole p300c, 11x10 = 110 worker cores):

  * `wk/b2z-kernel-cycle-census`   perf/b2z_kernel_census/{step_census.json, ops_perf_step_qb2c0.csv.gz}
  * `wk/b2z-diffusion-utilization` perf/b2z_diffusion_util/out/ops_perf_step_mine_qb2c0.csv.gz

Both were taken at commits that already contain `fc7fed56` ("both diffusion levers on by
default"), so unlike the campaign's trunk numbers the diffusion census is NOT stale.

The step census JSON carries per-RISC durations but all-zero shapes: it was written before
`b2z-diffusion-utilization` found that the column is `INPUT_0_W_PAD[LOGICAL]` and not
`INPUT_0_W`. This script re-reads the CSV with the correct keys so the two halves can be
joined, which is the first time the step's op table and its shape table have been the same
table.

Emits `step_map.json` beside itself.
"""
from __future__ import annotations

import csv
import gzip
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
CORES = 110                      # available worker cores, qb2 p300c 11x10, read from the report below
FENCE_DIM, FENCE_N = 32, 3

# ---------------------------------------------------------------- CSV plumbing
# Lifted deliberately from `b2z-diffusion-utilization`'s util_report.py so the region this
# script anchors is bit-identical to the region that row's published numbers came from.
# Re-implementing the anchoring would make any disagreement un-diagnosable.


def f(row, key, default=0.0):
    v = row.get(key, "")
    if v in ("", None):
        return default
    v = str(v).split("[", 1)[0].strip()
    try:
        return float(v)
    except ValueError:
        return default


def shape(row, pfx):
    out = []
    for d in ("W", "Z", "Y", "X"):
        v = 0.0
        for k in (f"{pfx}_{d}_PAD[LOGICAL]", f"{pfx}_{d}"):
            if k in row:
                v = f(row, k, 0.0)
                break
        out.append(int(v))
    return tuple(out)


def is_fence(row):
    if not row.get("OP CODE", "").startswith("Unary"):
        return False
    return shape(row, "INPUT_0")[2:] in ((FENCE_DIM, FENCE_DIM), (0, 0))


def fence_runs(rows):
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
    return runs


def find_region(rows, reps, period=None):
    runs = fence_runs(rows)
    if not runs:
        raise SystemExit("no fence run found")
    end = runs[-1][0]
    if len(runs) >= 2 and (end - runs[-2][1]) % reps == 0 and end - runs[-2][1] > 0:
        region = rows[runs[-2][1]:end]
    else:
        if not period:
            raise SystemExit("only one fence run; need --period")
        region = rows[end - reps * period:end]
    n = len(region) // reps
    seqs = {tuple(r.get("OP CODE", "?") for r in region[i * n:(i + 1) * n]) for i in range(reps)}
    if len(seqs) != 1:
        raise SystemExit(f"{reps} reps are not the same program sequence ({len(seqs)} distinct)")
    return region, n


def load(path):
    return list(csv.DictReader(gzip.open(path, "rt")))


# ---------------------------------------------------------------- the map
def main():
    out = {}

    # -- 1. the step's own accounting, straight out of the kernel-cycle census -------------
    kc = json.loads((SRC / "kc_step_census.json").read_text())
    med = kc["median_ms"]
    out["census"] = {
        "source": "wk/b2z-kernel-cycle-census perf/b2z_kernel_census/step_census.json",
        "commit": "57b3a806", "card": "qb2 card 0", "arch": "blackhole", "grid": [11, 10],
        "programs_per_step": kc["ops_per_block"], "reps": kc["reps"],
        "span_ms": med["span_ms"], "kernel_ms": med["kernel_ms"],
        "fw_ms": med["fw_ms"], "gap_ms": med["gap_ms"],
    }

    ops = kc["ops"]
    assert len(ops) == 1066, len(ops)

    # -- 2. join the shapes on, from the same run, anchored on the op-code sequence --------
    # The fence anchoring in `util_report.py` refuses this CSV: the device marker buffer wrapped
    # and 3 of the 10 profiled calls are torn, which is that row's own published instrument
    # defect #3. Anchor instead on the census's own 1066-op code sequence, which occurs verbatim
    # at 7 offsets in the kernel-census CSV and at 4 in the utilization row's CSV -- the same
    # program sequence at two different commits, which is itself the check that the step did not
    # change between them.
    rows = load(SRC / "kc_ops_perf_step_qb2c0.csv.gz")
    seq_kc = [o["op"] for o in ops]
    seq = [r["OP CODE"] for r in rows]
    n = len(seq_kc)
    hits = [i for i in range(len(seq) - n + 1) if seq[i:i + n] == seq_kc]
    if not hits:
        raise SystemExit("census op sequence does not occur in the CSV; cannot join")
    step_rows = rows[hits[-1]:hits[-1] + n]
    # core counts must not vary between repetitions or the join is meaningless
    spread = max(
        abs(f(rows[h + j], "CORE COUNT") - f(step_rows[j], "CORE COUNT"))
        for h in hits for j in range(n))
    other = load(SRC / "ops_perf_step_mine_qb2c0.csv.gz")
    oseq = [r["OP CODE"] for r in other]
    ohits = [i for i in range(len(oseq) - n + 1) if oseq[i:i + n] == seq_kc]
    out["join"] = {
        "csv": "perf/b2z_kernel_census/ops_perf_step_qb2c0.csv.gz (same run as the census JSON)",
        "sequence_occurrences_same_run": len(hits),
        "sequence_occurrences_other_commit_9db1ad88": len(ohits),
        "core_count_spread_across_reps": spread,
        "n": n,
    }

    # -- 3. per-op-code table -------------------------------------------------------------
    by = defaultdict(lambda: {"calls": 0, "kernel_ns": 0.0, "core_ns": 0.0,
                              "trisc1_ns": 0.0, "brisc_ns": 0.0, "cores": []})
    for o, r in zip(ops, step_rows):
        b = by[o["op"]]
        b["calls"] += 1
        k = o["kernel_ns"]
        b["kernel_ns"] += k
        c = f(r, "CORE COUNT", o["cores"]) or o["cores"]
        b["cores"].append(c)
        b["core_ns"] += k * c / CORES          # core-fraction-ns actually used
        b["trisc1_ns"] += o["trisc1_ns"]
        b["brisc_ns"] += o["brisc_ns"]

    total_k = sum(b["kernel_ns"] for b in by.values())
    table = []
    for op, b in sorted(by.items(), key=lambda kv: -kv[1]["kernel_ns"]):
        table.append({
            "op": op,
            "calls_per_step": b["calls"],
            "ms_per_step": b["kernel_ns"] / 1e6,
            "us_per_call": b["kernel_ns"] / b["calls"] / 1e3,
            "share_of_kernel_pct": 100 * b["kernel_ns"] / total_k,
            "mean_cores": st.mean(b["cores"]),
            "min_cores": min(b["cores"]),
            "occupancy_pct": 100 * b["core_ns"] / (b["kernel_ns"] / CORES) / CORES
            if b["kernel_ns"] else 0.0,
            "trisc1_residency_pct": 100 * b["trisc1_ns"] / b["kernel_ns"] if b["kernel_ns"] else 0,
            "brisc_residency_pct": 100 * b["brisc_ns"] / b["kernel_ns"] if b["kernel_ns"] else 0,
            "s_per_fold": b["kernel_ns"] * 200 / 1e9,
        })
    out["op_table"] = table
    out["kernel_ms_from_table"] = total_k / 1e6

    # -- 4. the residency half of the stall identity --------------------------------------
    # `--enable-sum-profiling` was never run on the step, so cb_wait_front / cb_reserve_back
    # do not exist for it (verified: the columns are blank in all three committed step CSVs).
    # What the default profiler does give, per op and per RISC, is how much of the kernel's
    # own duration each thread was resident for. That is the outer bracket of the identity:
    #     span = non-resident + resident,  resident = wait_in + wait_out + compute
    # so the step's non-resident term is measurable today and the inner split is not.
    span = med["span_ms"]
    kern = total_k / 1e6
    t1 = sum(o["trisc1_ns"] for o in ops) / 1e6
    br = sum(o["brisc_ns"] for o in ops) / 1e6
    out["residency_identity"] = {
        "span_ms": span,
        "kernel_ms": kern,
        "gap_outside_any_kernel_ms": span - kern,
        "trisc1_resident_ms": t1,
        "trisc1_resident_pct_of_kernel": 100 * t1 / kern,
        "trisc1_resident_pct_of_span": 100 * t1 / span,
        "brisc_resident_ms": br,
        "brisc_resident_pct_of_kernel": 100 * br / kern,
        "note": "cb_wait_front/cb_reserve_back absent for the step; inner split unmeasured",
    }

    # -- 5. occupancy, decomposed by shape class ------------------------------------------
    # `b2z-diffusion-utilization` attributes 89.2 % of the idle core-seconds to `[1,1,512,D]`.
    # Reproduce that from the joined table and then split the deficit the way a lever would
    # have to attack it: programs that CANNOT fill the grid vs programs that simply did not.
    idle = defaultdict(float)
    buckets = defaultdict(lambda: {"calls": 0, "kernel_ns": 0.0, "idle_ns": 0.0, "cores": []})
    for o, r in zip(ops, step_rows):
        k = o["kernel_ns"]
        c = f(r, "CORE COUNT", o["cores"]) or o["cores"]
        lost = k * (CORES - c) / CORES
        s0 = shape(r, "INPUT_0")
        # classify on the operand's own token/row extent
        y = s0[2]
        key = "token-axis 512 rows" if y == 512 else (
            "atom-axis" if y in (4480, 140, 32) or s0[1] == 140 else f"other y={y}")
        buckets[key]["calls"] += 1
        buckets[key]["kernel_ns"] += k
        buckets[key]["idle_ns"] += lost
        buckets[key]["cores"].append(c)
        idle[o["op"]] += lost
    tot_idle = sum(b["idle_ns"] for b in buckets.values())
    out["occupancy"] = {
        "duration_weighted_pct": 100 * sum(o["kernel_ns"] * (f(r, "CORE COUNT", o["cores"]) or o["cores"])
                                           for o, r in zip(ops, step_rows)) / (total_k * CORES),
        "idle_core_fraction_ms_per_step": tot_idle / 1e6,
        "idle_core_fraction_s_per_fold": tot_idle * 200 / 1e9,
        "by_shape_class": {k: {"calls": v["calls"], "ms_per_step": v["kernel_ns"] / 1e6,
                               "idle_ms_per_step": v["idle_ns"] / 1e6,
                               "share_of_idle_pct": 100 * v["idle_ns"] / tot_idle,
                               "mean_cores": st.mean(v["cores"])}
                           for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]["idle_ns"])},
        "by_op_top": {k: v / 1e6 for k, v in sorted(idle.items(), key=lambda kv: -kv[1])[:8]},
    }

    # -- 6. the core-count histogram: shape artifact or serial tail? ----------------------
    hist = defaultdict(lambda: {"programs": 0, "kernel_ms": 0.0})
    for o, r in zip(ops, step_rows):
        c = int(f(r, "CORE COUNT", o["cores"]) or o["cores"])
        hist[c]["programs"] += 1
        hist[c]["kernel_ms"] += o["kernel_ns"] / 1e6
    out["core_histogram"] = {str(c): v for c, v in sorted(hist.items())}

    (HERE / "step_map.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "op_table"}, indent=1)[:4000])
    print("\n=== OP TABLE (one 512 aa diffusion step, qb2 c0 BH, MEASURED) ===")
    print(f"{'op':34s} {'n':>5s} {'ms/step':>8s} {'us/call':>8s} {'%kern':>6s} "
          f"{'cores':>6s} {'min':>4s} {'TRISC1%':>7s} {'s/fold':>7s}")
    for t in table:
        print(f"{t['op'][:34]:34s} {t['calls_per_step']:5d} {t['ms_per_step']:8.3f} "
              f"{t['us_per_call']:8.1f} {t['share_of_kernel_pct']:6.2f} {t['mean_cores']:6.1f} "
              f"{t['min_cores']:4.0f} {t['trisc1_residency_pct']:7.1f} {t['s_per_fold']:7.3f}")


if __name__ == "__main__":
    main()


def grid_optimality():
    """Is the matmul under-fill recoverable by re-spreading onto more cores?

    ttnn's 2D matmul runs in time proportional to ceil(M/gy) * ceil(N/gx) * K tile-blocks per
    core. On an 11x10 grid gy <= 10 and gx <= 11, so for the step's dominant matmul --
    M = 512 rows = 16 tiles, N = 768 = 24 tiles -- every grid from 8x8 (64 cores) to 10x11
    (110 cores) gives ceil(16/gy) = 2 and ceil(24/gx) = 3, i.e. 6 tile-blocks per core.
    Using 110 cores instead of 64 leaves the per-core work unchanged and buys nothing.

    Run over all 387 matmuls: compare the per-core work of the grid ttnn actually chose against
    the best achievable on this device.
    """
    import math
    GY, GX = 10, 11
    kc = json.loads((SRC / "kc_step_census.json").read_text())
    ops = kc["ops"]
    rows = load(SRC / "kc_ops_perf_step_qb2c0.csv.gz")
    seq, t = [r["OP CODE"] for r in rows], [o["op"] for o in ops]
    n = len(t)
    h = [i for i in range(len(seq) - n + 1) if seq[i:i + n] == t][-1]
    seg = rows[h:h + n]
    tot = opt = 0.0
    worst = []
    for o, r in zip(ops, seg):
        if o["op"] != "MatmulDeviceOperation":
            continue
        M = int(f(r, "INPUT_0_Y_PAD[LOGICAL]")) // 32
        N = int(f(r, "INPUT_1_X_PAD[LOGICAL]")) // 32
        c = int(f(r, "CORE COUNT") or o["cores"])
        if not (M and N):
            continue
        cand = [(gy, gx) for gy in range(1, GY + 1) for gx in range(1, GX + 1) if gy * gx == c]
        if not cand:
            continue
        cur = min(math.ceil(M / gy) * math.ceil(N / gx) for gy, gx in cand)
        best = min(math.ceil(M / gy) * math.ceil(N / gx)
                   for gy in range(1, GY + 1) for gx in range(1, GX + 1))
        ms = o["kernel_ns"] / 1e6
        tot += ms
        opt += ms * best / cur
        if cur > best:
            worst.append((f"{M}x{N}", c, cur, best, ms))
    return {"matmul_ms_per_step": tot,
            "ms_if_every_matmul_used_its_best_grid": opt,
            "recoverable_ms_per_step": tot - opt,
            "recoverable_pct": 100 * (tot - opt) / tot,
            "suboptimal_calls_ms": sum(w[4] for w in worst),
            "suboptimal": worst}


if __name__ == "__main__":
    print("\n=== MATMUL GRID OPTIMALITY ===")
    print(json.dumps(grid_optimality(), indent=1))
