"""Render fit.py's table.json into the markdown the state doc carries.

Kept separate from fit.py on purpose: fit.py does arithmetic on measured seconds and must stay
readable as arithmetic. This only formats, so a formatting change can never move a number.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HOST_ONLY = ("prepare", "to_batch", "write_result", "predict_step/forward",
             "predict_step/forward/sampler", "predict_step")


def fmt(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def size_block(size, d):
    if "error" in d:
        return f"\n### {size} aa\n\n{d['error']}\n"
    o = [f"\n### {size} aa\n",
         "| item (exclusive) | t@1350 | t@800 | moved | band | F (s) | C (Mcycles) | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    it = sorted(d["items"].items(), key=lambda kv: -abs(kv[1]["F_s"]))
    for k, v in it:
        o.append(f"| `{k}` | {fmt(v['t_hi_s'])} | {fmt(v['t_lo_s'])} | {fmt(v['moved_s'])} "
                 f"| {fmt(v['band_s'])} | **{fmt(v['F_s'])}** | {fmt(v['C_Mcycles'],1)} "
                 f"| {v['verdict']} |")
    f = d["fold"]
    o.append(f"| **whole fold (bare arm)** | {fmt(f['t_hi_s'])} | {fmt(f['t_lo_s'])} "
             f"| {fmt(f['moved_s'])} | {fmt(f['band_s'])} | **{fmt(f['F_s'])}** "
             f"| {fmt(f['C_Mcycles'],1)} | {f['verdict']} |")
    c = d["closure"]
    o.append("")
    o.append(f"CLOSURE: items sum to **{fmt(c['item_F_sum_s'])} s** of F against this row's own "
             f"whole-fold F **{fmt(c['fold_F_s'])} s** (gap {fmt(c['gap_vs_fold_s'])} s, "
             f"{fmt(c['gap_vs_fold_pct'],2)} %) and against c10-fixed-cost's "
             f"{fmt(c['F_reference_s'],4)} s (gap {fmt(c['gap_vs_reference_s'])} s, "
             f"{fmt(c['gap_vs_reference_pct'],2)} %).")
    host = sum(v["F_s"] for k, v in d["items"].items() if k in HOST_ONLY or
               k.startswith("prepare/"))
    o.append(f"Host-Python items only (prepare*, to_batch, write_result, the exclusive bodies of "
             f"predict_step / forward / sampler): F = **{fmt(host)} s**.")
    o.append("")
    o.append("| arm | fold s @1350 | fold s @800 | perturbation vs bare @1350 |")
    o.append("|---|---|---|---|")
    base = d["perturbation"].get("bare", {}).get("1350")
    for arm, v in sorted(d["perturbation"].items()):
        hi = v.get("1350")
        pert = (f"{100.0*(hi-base)/base:+.2f} %"
                if (base and hi and hi == hi and arm != "cacheclear") else "-")
        o.append(f"| {arm} | {fmt(hi)} | {fmt(v.get('800'))} | {pert} |")
    return "\n".join(o)


def main(path):
    d = json.loads(Path(path).read_text())
    out = [f"TABLE: `{path}`. Form `{d['form']}`, gains "
           f"{d['gains'][0]:.6f} / {d['gains'][1]:.6f} at 1350/800 MHz."]
    for size in ("512", "298"):
        if size in d["sizes"]:
            out.append(size_block(size, d["sizes"][size]))
    if "scaling" in d:
        out.append("\n### SCALING: F per item, 512 aa against 298 aa (fold ratio 2.043x)\n")
        out.append("| item | F 512 | F 298 | ratio |")
        out.append("|---|---|---|---|")
        for k, v in sorted(d["scaling"].items(),
                           key=lambda kv: -abs(kv[1]["F_512_s"] or 0)):
            out.append(f"| `{k}` | {fmt(v['F_512_s'])} | {fmt(v['F_298_s'])} "
                       f"| {fmt(v['ratio'],3)} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
