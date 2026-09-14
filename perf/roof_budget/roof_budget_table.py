#!/usr/bin/env python3
"""The 512 aa fold's roofline budget: one table, one instrument, one session.

Every column below comes from the same 26 ttnn.graph captures taken in one process on qb2 card 2 at
`f072ae02f`, plus that same process's bracketed fold for the times. Nothing is composed across
instruments, which is the error this task exists to stop repeating.

  FLOPs        `exec_flops.py` on the capture: the device's own operand shapes, 2*b*M*N*K per
               matmul, the fused generic_op kernels included. Its known-answer control is a dense
               matmul, exact to 1.000000 on all 16 ladder shapes (`instrument_control.json`).
  bytes        `real_traffic.py` (origin/wk/b2x-diffusion-layer-bytes) on the SAME capture, deduped
               on buffer address. Its known-answer control is the same matmul: it returns 4/3 of the
               exact minimum, and the whole excess is one phantom read of the capture's terminal
               output buffer, which has no consumer inside the capture. The per-unit size of that
               term is reported as `terminal_MB` so every row's overcharge is visible.
  time         median over that unit's calls in the bracketed fold, device-synced both sides, so
               each is an upper bound and every achieved rate is a floor.
  roofs        measured in the same session on the same card, dispatch amortised: HiFi4 dense
               matmul 100.37 TFLOP/s at N=8192, starved bf16 add 424.7 GB/s at 8192^2.

The binding roof is decided per row by arithmetic intensity against the machine balance
(compute roof / stream roof FLOP per byte), not asserted.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import exec_flops as EF                                                       # noqa: E402
from itemize import itemize                                                   # noqa: E402
from real_traffic import counts as byte_counts                                # noqa: E402

# The three disjoint units that tile the fold: trunk pairformer blocks (64 x 4 trunk passes + 8
# confidence), MSA blocks (4 x 4), and the denoiser (200 steps). Every other captured unit is a
# child of one of these.
TOP = ["PairformerLayer|1x512x384,1x512x512x128",
       "MSALayer|1x512x512x128,1x1024x512x64",
       "DiffusionModule|"]


def terminal_MB(nodes):
    """Bytes the counter charges as a read of a buffer nothing in the capture consumes."""
    _ops, rows = itemize({"nodes": nodes})
    return sum(r["size"] for r in rows
               if r["kind"] == "DRAM" and r["n_consumers"] == 0) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "attrib2_512_tip_qb2c2.json")
    ap.add_argument("--baseline", type=Path, default=HERE / "attrib_512_tip_qb2c2.json")
    ap.add_argument("--captures", type=Path, default=HERE / "captures")
    ap.add_argument("--control", type=Path, default=HERE / "instrument_control.json")
    ap.add_argument("--stream", type=Path, default=HERE / "stream_roof2.json")
    ap.add_argument("--cell-s", type=float, default=17.340, help="the benchlocked fold of record")
    ap.add_argument("--out-json", type=Path, default=HERE / "roof_budget_512_qb2c2.json")
    ap.add_argument("--out-md", type=Path, default=HERE / "ROOF_BUDGET.md")
    a = ap.parse_args()

    ctl = json.loads(a.control.read_text())
    st = json.loads(a.stream.read_text())
    run = json.loads(a.run.read_text())
    base = json.loads(a.baseline.read_text())

    compute_roof = max(r["TFLOPs"] for r in ctl["rows"] if r["label"].startswith("cube")) * 1e12
    stream_roof = max(r["GBps"] for r in st["stream"]
                      if r["op"] == "add" and r["N"] == 8192) * 1e9
    balance = compute_roof / stream_roof                       # FLOP per byte
    rate = {r["label"]: r["TFLOPs"] * 1e12 for r in ctl["rows"] if "TFLOPs" in r}

    sigs = run["attrib"]["sigs"]
    fold_s = base["baseline_summary"]["plain_median_s"]
    scale = a.cell_s / fold_s

    rows = []
    for cap in sorted(glob.glob(str(a.captures / "cap_*.json.gz"))):
        sig = Path(cap).name[len("cap_"):-len(".json.gz")].replace("__", "|")
        m = sigs.get(sig)
        if m is None:
            continue
        nodes = EF.nodes_of(cap)
        t = EF.totals(nodes)
        bc = byte_counts({"nodes": nodes})
        B = bc["real_MB"] * 1e6
        F = t["matmul_padded"] + t["eltwise_padded"]
        ms, calls = m["median_ms"], m["calls"]
        s_fold = calls * ms / 1e3
        ai = F / B if B else 0.0
        bound = "compute" if ai > balance else "bandwidth"
        s_roof = calls * (F / compute_roof if bound == "compute" else B / stream_roof)
        rows.append({
            "sig": sig, "calls": calls, "ms_per_call": round(ms, 4),
            "s_per_fold": round(s_fold, 3), "s_per_fold_at_cell": round(s_fold * scale, 3),
            "GFLOP_per_call": round(F / 1e9, 3),
            "GFLOP_logical_per_call": round((t["matmul_logical"] + t["eltwise_logical"]) / 1e9, 3),
            "tile_pad_factor": round(t["tile_pad_factor"], 5),
            "MB_per_call": round(B / 1e6, 3),
            "terminal_MB_per_call": round(terminal_MB(nodes), 3),
            "ops_per_call": t["n_ops"],
            "AI_flop_per_byte": round(ai, 2), "binding_roof": bound,
            "achieved_TFLOPs": round(F / (ms / 1e3) / 1e12, 3),
            "achieved_GBps": round(B / (ms / 1e3) / 1e9, 1),
            "pct_binding_roof": round(100 * s_roof / s_fold, 1) if s_fold else 0.0,
            "s_at_roof": round(s_roof, 3),
            "s_above_roof": round(s_fold - s_roof, 3),
            "s_above_roof_at_cell": round((s_fold - s_roof) * scale, 3),
        })
    rows.sort(key=lambda r: -r["s_above_roof"])
    by = {r["sig"]: r for r in rows}
    top = [by[s] for s in TOP if s in by]

    fold_F = sum(r["GFLOP_per_call"] * r["calls"] for r in top) * 1e9
    fold_F_log = sum(r["GFLOP_logical_per_call"] * r["calls"] for r in top) * 1e9
    fold_B = sum(r["MB_per_call"] * r["calls"] for r in top) * 1e6
    fold_term = sum(r["terminal_MB_per_call"] * r["calls"] for r in top) * 1e6
    top_s = sum(r["s_per_fold"] for r in top)
    top_above = sum(r["s_above_roof"] for r in top)

    # shape-honest arithmetic floor: every matmul priced at the rate ITS shape reaches, not the
    # 8192-cube rate. Mapping is by the ladder label measured in the same session.
    SHAPE = {
        "PairformerLayer|1x512x384,1x512x512x128": ["trimul-inproj", "trimul-einsum",
                                                    "trimul-outproj", "triatt-qkv",
                                                    "triatt-outproj", "pair-transition-up",
                                                    "pair-transition-down"],
        "MSALayer|1x512x512x128,1x1024x512x64": ["opm-outer", "opm-proj", "pwa-proj",
                                                 "trimul-inproj", "trimul-einsum", "triatt-qkv"],
        "DiffusionModule|": ["token-dit-qkv", "token-dit-transition", "atom-dit-qkv"],
    }
    honest = 0.0
    honest_rows = []
    for r in top:
        labels = SHAPE[r["sig"]]
        rr = sum(rate[x] for x in labels) / len(labels)
        s = r["GFLOP_per_call"] * 1e9 * r["calls"] / rr
        honest += s
        honest_rows.append({"sig": r["sig"], "mean_shape_rate_TFLOPs": round(rr / 1e12, 2),
                            "s": round(s, 3), "labels": labels})

    summary = {
        "head": "f072ae02f", "host": "tt-quietbox2", "card": 2, "ttnn": "0.68.0",
        "session_fold_s": fold_s, "session_loadavg": base["baseline"][0]["loadavg"],
        "cell_of_record_s": a.cell_s, "cell_scale": round(scale, 4),
        "compute_roof_TFLOPs": round(compute_roof / 1e12, 2),
        "stream_roof_GBps": round(stream_roof / 1e9, 1),
        "machine_balance_flop_per_byte": round(balance, 1),
        "fold_TFLOP_executed": round(fold_F / 1e12, 3),
        "fold_TFLOP_device_shapes_untiled": round(fold_F_log / 1e12, 3),
        "fold_tile_pad_factor": round(fold_F / fold_F_log, 5),
        "fold_TB": round(fold_B / 1e12, 4),
        "fold_TB_terminal_overcharge": round(fold_term / 1e12, 4),
        "arithmetic_floor_s_at_cube_roof": round(fold_F / compute_roof, 3),
        "arithmetic_floor_s_shape_honest": round(honest, 3),
        "traffic_floor_s": round(fold_B / stream_roof, 3),
        "binding_floor_s": round(max(fold_B / stream_roof, honest), 3),
        "top_level_s_per_fold": round(top_s, 3),
        "top_level_s_above_roof": round(top_above, 3),
        "top_level_s_above_roof_at_cell": round(top_above * scale, 3),
        "shape_honest_rows": honest_rows,
    }
    a.out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    L = ["# Boltz-2 512 aa at f072ae02f: the roofline budget",
         "",
         f"qb2 card 2 (p300c, 11x10 grid, AICLK 800 MHz), ttnn 0.68.0, one process, one session. "
         f"FLOPs and bytes come off the same 26 captures; times come off the same process's "
         f"bracketed fold. Session fold {fold_s:.3f} s at loadavg "
         f"{', '.join(base['baseline'][0]['loadavg'])}; the benchlocked cell of record is "
         f"{a.cell_s:.3f} s, so seconds are also given scaled by {scale:.4f}.",
         "",
         f"Roofs measured in this session with dispatch amortised: dense bf16 HiFi4 "
         f"**{compute_roof/1e12:.2f} TFLOP/s**, starved 8192^2 add **{stream_roof/1e9:.1f} GB/s**. "
         f"Machine balance {balance:.1f} FLOP/byte.",
         "",
         "| unit | calls | ms/call | GFLOP/call | MB/call | FLOP/byte | binding roof | % of it | "
         "s/fold | **s above roof** | at the cell |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| `{sig}` | {calls} | {ms_per_call:.3f} | {GFLOP_per_call:.1f} | "
                 "{MB_per_call:.1f} | {AI_flop_per_byte} | {binding_roof} | {pct_binding_roof} % | "
                 "{s_per_fold} | **{s_above_roof}** | {s_above_roof_at_cell} |".format(**r))
    L += ["", "## The three headline numbers", "",
          f"- **{summary['fold_TFLOP_executed']:.2f} TFLOP executed** over the three disjoint "
          f"top-level units. Tile padding accounts for "
          f"{100*(summary['fold_tile_pad_factor']-1):.2f} % of it.",
          f"- **{summary['fold_TB']:.4f} TB moved**, deduped on buffer address, of which "
          f"{summary['fold_TB_terminal_overcharge']:.4f} TB is the counter's terminal-output "
          f"charge (an upper bound on its overcount).",
          f"- **floor {summary['binding_floor_s']:.3f} s**: traffic "
          f"{summary['traffic_floor_s']:.3f} s at {stream_roof/1e9:.1f} GB/s against arithmetic "
          f"{summary['arithmetic_floor_s_shape_honest']:.3f} s at the rate each shape actually "
          f"reaches ({summary['arithmetic_floor_s_at_cube_roof']:.3f} s if every FLOP is priced at "
          f"the 8192-cube rate, which no op in this fold runs at).",
          ""]
    a.out_md.write_text("\n".join(L) + "\n")
    print(json.dumps(summary, indent=1))
    for r in rows[:12]:
        print("%-52s %5d x %8.3f ms  %8.1f MB  %8.1f GF  AI %7.1f  %-9s %5.1f %%  above %6.3f s"
              % (r["sig"], r["calls"], r["ms_per_call"], r["MB_per_call"], r["GFLOP_per_call"],
                 r["AI_flop_per_byte"], r["binding_roof"], r["pct_binding_roof"],
                 r["s_above_roof"]))
    print("WROTE", a.out_json, a.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
