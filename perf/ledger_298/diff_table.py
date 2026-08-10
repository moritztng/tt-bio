#!/usr/bin/env python3
"""Phase-1 per-op table for post-G1 protenix-v2 diffusion, on THIS card.

Reads the in-fold per-op record (diff_step_ops.py) and the self-measured roofs, emits one row
per op INSTANCE CLASS with every column the donecheck enforces: FLOPs/bytes from real shapes,
arithmetic intensity vs the measured machine balance, binding roof + % achieved, core
utilisation (cores engaged of 130, from the real _batched_matmul_config chooser for the
batched sites, or the mcast_1d derivation for plain ttnn.matmul), and compute/comm overlap
(max = overlap, sum = serialise; floor = max(in0/read, in1/read + out/write) per G1).

This is a PROJECTION of attribution (per-step us times 200 steps), not a fold gain.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import torch  # noqa: F401
import ttnn
import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import get_device

TILE = 32
DB = {"BFLOAT16": 2, "bfloat16": 2, "FLOAT32": 4, "float32": 4, "BFLOAT8_B": 1.0625,
       "bfloat8_b": 1.0625, "BFLOAT4_B": 0.5625, "bfloat4_b": 0.5625, "UINT32": 4,
       "uint32": 4, "INT32": 4, "int32": 4, "UINT16": 2, "uint16": 2, "UINT8": 1, "uint8": 1}
MATMUL = ("linear", "matmul", "minimal_matmul")
ELEM = {"add": 1, "add_": 1, "multiply": 1, "multiply_": 1, "subtract": 1, "relu": 1,
         "sigmoid": 1, "silu": 2, "gelu": 4, "reciprocal": 1, "typecast": 0}
NORM = {"layer_norm": 5, "rms_norm": 4, "softmax": 5}
MOVE = ("permute", "transpose", "concat", "reshape", "unsqueeze", "squeeze", "chunk",
        "slice", "pad", "clone", "to_layout", "to_memory_config", "reallocate", "repeat", "sum")


def nb(t):
    return math.prod(t["shape"]) * DB.get(t["dtype"], 2)


def is_dram(t):
    return "DRAM" in (t.get("buf") or "").upper()


def mkn(rec):
    ins = [t for t in rec["in"] if t is not None]
    if len(ins) < 2:
        return None
    a, b = ins[0]["shape"], ins[1]["shape"]
    m, k = a[-2], a[-1]
    if b[-2] == k:
        n = b[-1]
    elif b[-1] == k:
        n = b[-2]
    else:
        return None
    return m, k, n, max(math.prod(a[:-2]), 1)


def flops(rec):
    op = rec["op"]
    outs = [rec["out"]] if rec["out"] else []
    if op in MATMUL:
        g = mkn(rec)
        if not g:
            return 0, "matmul: K not matched"
        m, k, n, batch = g
        return 2 * batch * m * k * n, f"2*{batch}*{m}*{k}*{n}"
    if op in ELEM:
        e = sum(math.prod(t["shape"]) for t in outs)
        return e * ELEM[op], f"{ELEM[op]} FLOP/out-elem"
    if op in NORM:
        e = sum(math.prod(t["shape"]) for t in outs)
        return e * NORM[op], f"~{NORM[op]} FLOP/out-elem"
    if op in MOVE:
        return 0, "data movement"
    return 0, "unclassified"


def classkey(rec):
    def k(t):
        return None if t is None else (tuple(t["shape"]), t["dtype"], t["buf"], t["layout"])
    return (rec["op"], rec["site"], tuple(k(t) for t in rec["in"]), k(rec["out"]))


def op_dtype(rec):
    for t in (rec["in"] or []) + ([rec["out"]] if rec["out"] else []):
        if t and t.get("dtype") in ("FLOAT32", "float32"):
            return "fp32"
    return "bf16"


def cores_plain(m, k, n, batch, grid):
    """Plain ttnn.matmul automatic path: get_mcast_1d_config splits M and N only, batch serial."""
    gx, gy = grid
    mt, nt = -(-m // TILE), -(-n // TILE)
    if m <= n:
        pcm, pcn = mt, -(-nt // (gx * gy))
    else:
        pcm, pcn = -(-mt // (gx * gy)), nt
    blocks = max(1, mt // max(1, pcm)) * max(1, nt // max(1, pcn))
    return min(blocks, gx * gy), pcm, pcn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True)
    ap.add_argument("--roofs", required=True)
    ap.add_argument("--calls-per-fold", type=float, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    R = json.load(open(args.roofs))
    cr = R["compute_roof"]
    bf16_roof = cr["bf16_peak_TFLOPs"] * 1e12
    fp32_roof = cr["fp32_peak_TFLOPs"] * 1e12
    rd_roof = R["dram_roofs"]["read_peak_GBs"] * 1e9
    wr_roof = R["dram_roofs"]["write_peak_GBs"] * 1e9
    g = R["device"]["core_grid_main"].split("x")
    grid = (int(g[0]), int(g[1]))
    ncores = grid[0] * grid[1]

    dev = get_device()
    live = (T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y)
    live_cores = live[0] * live[1]
    print(f"roofs grid={grid} ncores={ncores}  live grid={live} live_cores={live_cores}", flush=True)

    D = json.load(open(args.ops))
    stage_ms = D.get("marks", {}).get("stage_warm_s", 0.0) * 1000
    scale = args.calls_per_fold
    agg = defaultdict(lambda: {"n": 0, "s": 0.0})
    meta = {}
    for rec in D["records"]:
        key = classkey(rec)
        agg[key]["n"] += 1
        agg[key]["s"] += rec["s"]
        meta.setdefault(key, rec)

    rows = []
    for key, a in agg.items():
        rec = meta[key]
        fl, basis = flops(rec)
        ins = [t for t in rec["in"] if t]
        out = rec["out"]
        rd = sum(nb(t) for t in ins if is_dram(t))
        wr = nb(out) if out and is_dram(out) else 0
        secs = a["s"] / a["n"] if a["n"] else 0.0
        dt = op_dtype(rec)
        c_roof = fp32_roof if dt == "fp32" else bf16_roof
        bal = c_roof / rd_roof
        cands = [("COMPUTE", fl / c_roof), ("DRAM-READ", rd / rd_roof),
                 ("DRAM-WRITE", wr / wr_roof)]
        binding, t_ideal = max(cands, key=lambda x: x[1])
        frac = t_ideal / secs if secs > 0 else 0.0
        ai = fl / (rd + wr) if (rd + wr) else None

        row = {
            "op": rec["op"], "site": rec["site"], "n_per_step": a["n"],
            "in_shapes": [t["shape"] for t in ins],
            "out_shape": out["shape"] if out else None, "dtype": dt,
            "us_per_call": round(secs * 1e6, 2),
            "ms_per_fold": round(a["s"] * scale * 1e3, 2),
            "pct_of_stage": round(100 * a["s"] * scale * 1e3 / stage_ms, 2) if stage_ms else None,
            "GFLOP_per_call": round(fl / 1e9, 5), "flop_basis": basis,
            "dram_read_MB": round(rd / 1e6, 3), "dram_write_MB": round(wr / 1e6, 3),
            "AI_FLOP_per_byte": round(ai, 1) if ai is not None else None,
            "machine_balance": round(bal, 1),
            "achieved_TFLOPs": round(fl / secs / 1e12, 2) if secs and fl else None,
            "achieved_read_GBs": round(rd / secs / 1e9, 1) if secs and rd else None,
            "achieved_write_GBs": round(wr / secs / 1e9, 1) if secs and wr else None,
            "binding_roof": binding, "pct_of_binding_roof": round(100 * frac, 1),
        }

        # core utilisation + overlap (matmul family only)
        if rec["op"] in MATMUL:
            g2 = mkn(rec)
            if g2:
                m, k, n, batch = g2
                mt, kt, nt = m // TILE, k // TILE, n // TILE
                row["tiles_MKN"] = [mt, kt, nt, batch]
                cfg = None
                if batch >= 2:
                    try:
                        cfg = T._batched_matmul_config(batch, mt, kt, nt,
                                                        4 if dt == "fp32" else 2)
                    except Exception as e:
                        cfg = None
                        row["chooser_err"] = str(e)[:120]
                if cfg is not None:
                    pcm = cfg.per_core_M
                    blocks = batch * mt // pcm
                    engaged = min(blocks, live_cores)
                    rounds = -(-blocks // live_cores)
                    tail = blocks - (rounds - 1) * live_cores if blocks > live_cores else blocks
                    row["config"] = {"per_core_M": pcm, "per_core_N": cfg.per_core_N,
                                     "in0_block_w": cfg.in0_block_w}
                    row["cores_engaged"] = engaged
                    row["blocks"] = blocks
                    row["rounds"] = rounds
                    row["tail_round_cores"] = tail
                    row["occupancy_pct"] = round(100 * tail / live_cores, 1) if blocks > live_cores else 100.0
                    row["config_path"] = "batched_matmul (batch on grid)"
                else:
                    eng, pcm, pcn = cores_plain(m, k, n, batch, live)
                    row["config"] = {"per_core_M": pcm, "per_core_N": pcn}
                    row["cores_engaged"] = eng
                    row["blocks"] = None
                    row["occupancy_pct"] = round(100 * eng / live_cores, 1)
                    row["config_path"] = "plain ttnn.matmul (batch serial)"
                # overlap model: in0 read by one RISC; in1 read + out write by the other.
                in0 = nb(ins[0]) if ins else 0
                in1 = nb(ins[1]) if len(ins) > 1 else 0
                outb = nb(out) if out else 0
                comm = max(in0 / rd_roof, in1 / rd_roof + outb / wr_roof)
                comp = fl / c_roof if fl else 0.0
                row["comm_floor_us"] = round(comm * 1e6, 2)
                row["compute_floor_us"] = round(comp * 1e6, 2)
                row["overlap"] = "max (overlap)" if comp > 0 and comm > 0 else (
                    "comm-only" if comp == 0 else "compute-only")
                row["floor_vs_measured"] = round(t_ideal * 1e6 / secs, 2) if secs else None

        if fl == 0 and rd + wr < 1e5:
            row["verdict"] = "OVERHEAD"
        elif frac >= 1.05:
            row["verdict"] = "ACCOUNTING-SUSPECT"
        elif frac >= 0.70:
            row["verdict"] = {"COMPUTE": "COMPUTE-BOUND", "DRAM-READ": "READ-BOUND",
                              "DRAM-WRITE": "WRITE-BOUND"}[binding]
        else:
            row["verdict"] = "LATENCY-or-OCCUPANCY-BOUND"
        row["gap_ms_per_fold"] = round(max(0.0, secs - t_ideal) * a["n"] * scale * 1e3, 2)
        rows.append(row)

    rows.sort(key=lambda r: -r["ms_per_fold"])
    sumfold = sum(r["ms_per_fold"] for r in rows)
    out = {
        "roofs": {"bf16_TFLOPs": bf16_roof / 1e12, "fp32_TFLOPs": fp32_roof / 1e12,
                  "read_GBs": rd_roof / 1e9, "write_GBs": wr_roof / 1e9,
                  "machine_balance_bf16": round(bf16_roof / rd_roof, 1),
                  "machine_balance_fp32": round(fp32_roof / rd_roof, 1)},
        "grid": list(grid), "ncores": ncores, "live_cores": live_cores,
        "stage_ms_per_fold_measured": round(stage_ms, 1),
        "block_wall_ms": round(D["block_wall_s"] * 1e3, 3),
        "per_op_sum_ms": round(D["sum_s"] * 1e3, 3),
        "per_op_coverage_pct": round(100 * D["sum_s"] / D["block_wall_s"], 1),
        "calls_per_fold": scale, "per_op_sum_ms_per_fold": round(sumfold, 1),
        "n_classes": len(rows), "n_calls_per_step": D["n_ops"], "rows": rows,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"stage/fold measured={stage_ms:.1f} ms  block_wall={out['block_wall_ms']} ms  "
          f"per-op sum={out['per_op_sum_ms']} ms ({out['per_op_coverage_pct']}% of block)  "
          f"per-op sum/fold={sumfold:.1f} ms", flush=True)
    print(f"{'op':14s} {'site':26s} {'n':>3s} {'us/c':>7s} {'ms/fld':>7s} {'AI':>6s} "
          f"{'roof':>10s} {'%roof':>6s} {'cores':>6s} {'overlap':>14s} verdict")
    for r in rows[:args.top]:
        ai = r["AI_FLOP_per_byte"]
        pr = r["pct_of_binding_roof"]
        ce = r.get("cores_engaged")
        ov = r.get("overlap", "-")
        print(f"{r['op'][:14]:14s} {r['site'][-26:]:26s} {r['n_per_step']:3d} "
              f"{r['us_per_call']:7.1f} {r['ms_per_fold']:7.1f} "
              f"{(f'{ai:.0f}' if ai else '-'):>6s} {r['binding_roof']:>10s} "
              f"{(f'{pr:.0f}' if pr is not None else '-'):>6s} "
              f"{(f'{ce}' if ce is not None else '-'):>6s} {ov[:14]:>14s} {r['verdict']}")


if __name__ == "__main__":
    main()
