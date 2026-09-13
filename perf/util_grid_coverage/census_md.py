#!/usr/bin/env python3
"""Write CENSUS.md: the per-op grid-coverage table, ranked, with each shortfall's reason."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FOLD_S = 23.841           # b2x-baseline-attrib, 512 aa cdk2x2, qb2 p300c, median n=3 warm


def reason(r):
    """Why this class runs on fewer than 110 cores, from its own program config."""
    if r["cores"] >= r["avail"]:
        return "full grid"
    op, pcm, pcn = r["op"], r["per_core_M"], r["per_core_N"]
    if op.startswith("Matmul"):
        if r["program_config"].endswith("MultiCast1DProgramConfig"):
            return (f"1D split over M only: {pcm} tile-rows/core is the smallest block that fits "
                    f"the {r['cores']} cores it needs; a 111th core would have no row to take")
        return (f"2D split {pcm}x{pcn} tiles/core; the output is {r['Mt']}x{r['Nt']} tiles and no "
                f"finer rectangle fits an 11x10 grid")
    if op.startswith("LayerNorm"):
        return f"row-parallel kernel: {r['Mt']} tile rows exist, so {r['Mt']} cores"
    if op.startswith("NlpCreateHeads"):
        return f"one core per head, {r['cores']} heads"
    if r["use_multicore"] == "false":
        return "single-core fallback (use_multicore=false)"
    return f"{r['Mt']}x{r['Nt']} output tiles"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--curve", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cen = json.loads(a.census.read_text())
    cur = json.loads(a.curve.read_text())

    lines = []
    P = lines.append
    P("# Every op of the 512 aa fold, and how many of the 110 cores it gets")
    P("")
    P("`ws:util-grid-coverage`. Blackhole, 11x10 = 110 worker cores. The core counts and device")
    P("times are read out of the shipped fold: the profiler captures committed by")
    P("`b2z-kernel-cycle-census` on qb2 card 0 (tt-metal v0.68.0 source build, `ENABLE_TRACY=ON`).")
    P("The scaling curves are measured separately on qb2 card 2 with the same build.")
    P("")
    P("## Headline")
    P("")
    for label, c in cen.items():
        P(f"- **{label}: {c['weighted_cores']:.1f} of {c['avail_cores']} cores**, weighted by "
          f"device kernel time ({c['coverage_pct']:.1f} %). {c['programs']} programs, "
          f"{c['kernel_ms']:.4f} ms of kernel in a {c['span_ms']:.4f} ms span "
          f"({c['gap_pct']:.1f} % of the span is in no kernel at all).")
    P("")
    P("## The census")
    P("")
    for label, c in cen.items():
        tot_short = sum(r["kernel_ms"] for r in c["rows"] if r["cores"] < r["avail"])
        P(f"### {label} — {c['calls_per_fold']} calls/fold, {c['s_per_fold']:.3f} s/fold of "
          f"kernel time")
        P("")
        P(f"{tot_short:.4f} ms of the {c['kernel_ms']:.4f} ms runs on a partial grid "
          f"({100 * tot_short / c['kernel_ms']:.1f} %).")
        P("")
        P("| op | cores | progs | kernel ms | % | s/fold | out tiles BxMxNxK | why not 110 |")
        P("|---|---:|---:|---:|---:|---:|---|---|")
        for r in c["rows"]:
            if r["cores"] >= r["avail"] and r["kernel_ms"] < 0.3:
                continue
            P(f"| {r['op'].replace('DeviceOperation', '')} | {r['cores']} | {r['programs']} | "
              f"{r['kernel_ms']:.4f} | {r['pct_of_kernel']:.2f} | {r['s_per_fold']:.3f} | "
              f"{r['batch']}x{r['Mt']}x{r['Nt']}x{r['Kt']} | {reason(r)} |")
        P("")
    P("## The measured scaling curves")
    P("")
    P("Each class rebuilt at its shipped shape, dtype, fidelity, memory config and fused")
    P("activation, with the matmul program config written out explicitly so the engaged core")
    P("count is exact. Device kernel time, median of 20 back-to-back calls, qb2 card 2.")
    P("")
    P("| class | in-situ cores | max reachable | best cores | in-situ us | best us | gain | "
      "s/fold | recoverable s |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = 0.0
    for case, v in sorted(cur.items(), key=lambda kv: -kv[1]["s_per_fold_at_situ"]):
        tot += v["s_per_fold_recoverable"] or 0
        P(f"| {case} | {v['situ_cores']} | {v['max_cores_reachable']} | {v['best_cores']} | "
          f"{v['situ_us']:.2f} | {v['best_us']:.2f} | {v['speedup_best_over_situ']:.3f}x | "
          f"{v['s_per_fold_at_situ']:.3f} | {v['s_per_fold_recoverable']:.3f} |")
    P(f"| **total** | | | | | | | | **{tot:.3f}** |")
    P("")
    P("Full ladders, including the points a wider grid refuses, are in `curve_512_qb2c2.json`.")
    P("")
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}, {len(lines)} lines, recoverable total {tot:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
