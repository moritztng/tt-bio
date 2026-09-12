#!/usr/bin/env python3
"""Render perf/b2z2_layout/census.json as the per-site table, markdown."""
import json
import sys
from pathlib import Path

c = json.load(open(sys.argv[1]))
out = [
    "# Boltz-2 512 aa: every device program of a Pairformer block and a diffusion step, by site",
    "",
    "ARCH: BH. qb2 physical card 0, one Blackhole processor of a p300c, 11x10 grid, tt-metal built",
    "from source at v0.68.0 with the device profiler on. Source data is the per-op profiler CSV",
    "taken by `ws:b2z-kernel-cycle-census`; `perf/b2z2_layout/census.py` re-windows it by op-code",
    "signature and reads the dtype, layout, math-fidelity and shape columns the JSON census dropped.",
    "No new device time was spent to produce this table.",
    "",
    "A site is one (op, compute kernel, fidelity, accumulator width, input shape, output shape,",
    "input dtype, output dtype, input layout, output layout). `acc` is `fp32acc` when the program",
    "carries `fp32_dest_acc_en=1`, which halves the dest register file from 8 tiles to 4.",
    "",
]
for tag in ("PairformerLayer", "DiffusionStep"):
    a = c[tag]
    out += [
        f"## {tag}", "",
        f"{a['programs']} device programs, span **{a['span_ms']:.4f} ms**, "
        f"kernel time **{a['kernel_ms_sum']:.4f} ms**.", "",
        "| | programs | kernel ms | % of span |",
        "|---|---|---|---|",
        f"| pure movement (transpose, permute, slice, concat, copy, pad, tilize, heads) "
        f"| {a['movement']['n']} | {a['movement']['kernel_ms']:.4f} | **{a['movement']['pct_of_span']:.2f}** |",
        f"| layout transitions only (tilize / untilize / reshape) "
        f"| {a['layout_only']['n']} | {a['layout_only']['kernel_ms']:.4f} | **{a['layout_only']['pct_of_span']:.2f}** |",
        f"| any fp32 operand or result (storage) "
        f"| {a['fp32_touching']['n']} | {a['fp32_touching']['kernel_ms']:.4f} | **{a['fp32_touching']['pct_of_span']:.2f}** |",
        f"| fp32 dest accumulate "
        f"| {a['fp32_dest_acc']['n']} | {a['fp32_dest_acc']['kernel_ms']:.4f} | **{a['fp32_dest_acc']['pct_of_span']:.2f}** |",
        "",
        "Math fidelity, as executed:", "",
        "| fidelity | programs | kernel ms |", "|---|---|---|",
    ]
    for k, v in a["fidelity"].items():
        out.append(f"| {k} | {v['n']} | {v['kernel_ms']:.4f} |")
    out += ["", "### Sites, most expensive first", "",
            "| op | kernel | fid | acc | in0 | out0 | in0 dt | out0 dt | in0 lay | out0 lay "
            "| n | kernel ms | TRISC1 ms | cores | % span |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in a["sites"]:
        out.append(
            f"| {s['op'].replace('DeviceOperation','').replace('Operation','')} | {s['kernel']} "
            f"| {s['fidelity']} | {s['acc']} | {s['in0']} | {s['out0']} | {s['in0_dt']} "
            f"| {s['out0_dt']} | {s['in0_layout']} | {s['out0_layout']} | {s['n']} "
            f"| {s['kernel_ms']:.4f} | {s['trisc1_ms']:.4f} | {s['cores_avg']} | {s['pct_of_span']:.3f} |")
    out += ["", "### Every movement program, in issue order", "",
            "| # | op | ms | cores | in0 | in0 dt | in0 layout | out0 | out0 dt | out0 layout |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m in a["movement_detail"]:
        out.append(
            f"| {m['i']} | {m['op'].replace('DeviceOperation','').replace('Operation','')} "
            f"| {m['ms']:.4f} | {m['cores']} | {'x'.join(map(str, m['in0']))} | {m['in0_dt']} "
            f"| {m['in0_layout']} | {'x'.join(map(str, m['out0']))} | {m['out0_dt']} "
            f"| {m['out0_layout']} |")
    out.append("")
Path(sys.argv[2]).write_text("\n".join(out))
print("wrote", sys.argv[2], len(out), "lines")
