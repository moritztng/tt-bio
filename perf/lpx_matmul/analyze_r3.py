"""Tables for an r3.py run: pad, generic_op kernels, layer_norm -> matmul chain, pad cost, OPM layouts.

usage: analyze_r3.py R1_DIR RUN_DIR [RUN_DIR ...]   -> first RUN_DIR/summary.md
Device us per call (trace replay); "fold s" = us x calls per fold, from R1's capture of the 730-token fold.
"""
import json, sys
from pathlib import Path

R1 = Path(sys.argv[1]); RUNS = [Path(p) for p in sys.argv[2:]]
calls = json.loads((R1 / "calls.json").read_text())
ev = [json.loads(l) for r in RUNS for l in open(r / "bench.jsonl")]
out = [f"# lpx-matmul r3 ({', '.join(r.name for r in RUNS)})", ""]

def ok(res): return [x for x in res if "us" in x]
def best(res, pred=lambda v: True):
    c = [x for x in ok(res) if pred(x["var"])]
    return min(c, key=lambda x: x["us"]) if c else None
def fmt(x, base): return f"{x['var']} | {x['us']:.0f} | {base / x['us']:.2f}" if x else "- | - | -"

pad = [e for e in ev if e["ev"] == "pad"]
if pad:
    out += ["## pad: token axis 736/730 -> 768 (every token dim pads; the padded time carries the extra work)",
            "| # | calls | work x | base us | best at 736 | us | x | best at 768, today's precision | us | x | best at 768, bfp8 LoFi | us | x | fold s base -> bf16@768 -> b8@768 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in pad:
        res = e["results"]; base = next(x["us"] for x in ok(res) if x["var"] == "base")
        b736 = best(res, lambda v: "_736" in v)
        bas = best(res, lambda v: "_768" in v and v.split("_")[0] in ("as", "bf16"))
        b8 = best(res, lambda v: "_768" in v and v.startswith("b8lofi"))
        n = e["n_total"]
        out.append(f"| {e['rank']} | {n} | {e['work_ratio']:.3f} | {base:.0f} | {fmt(b736, base)} | {fmt(bas, base)} | {fmt(b8, base)} | "
                   f"{base * n / 1e6:.2f} -> {(bas or {'us': base})['us'] * n / 1e6:.2f} -> {(b8 or {'us': base})['us'] * n / 1e6:.2f} |")
    out += ["", "in0_block_w ladder at 768, best subblock per ibw (us):", "| # | point | " + " | ".join(f"ibw{b}" for b in (1, 2, 3, 4, 6, 8, 12)) + " |",
            "|---|---|" + "---|" * 7]
    for e in pad:
        for pt in ("as", "bf16", "noacc", "b8lofi"):
            row = []
            for b in (1, 2, 3, 4, 6, 8, 12):
                x = best(e["results"], lambda v, b=b, pt=pt: v.startswith(f"{pt}_768_") and f"ibw{b}_" in v + "_")
                row.append(f"{x['us']:.0f}" if x else "-")
            if any(r != "-" for r in row):
                out.append(f"| {e['rank']} | {pt} | " + " | ".join(row) + " |")

pc = [e for e in ev if e["ev"] == "padcost"]
for e in pc:
    out += ["", "## padcost: ttnn.pad / ttnn.slice of the einsum operands (device us)", "| op | us |", "|---|---|"]
    out += [f"| {x['var']} | {x['us']:.0f} |" if "us" in x else f"| {x['var']} | {x['err']} |" for x in e["results"]]

gen = [e for e in ev if e["ev"] == "gen"]
if gen:
    out += ["", "## generic_op matmul kernels through tt-bio's entry points (device us; base = bf16, HiFi4, fp32 acc, as the fold runs them)",
            "| kernel | calls | " + " | ".join(x["var"] for x in gen[0]["results"]) + " | fold s base -> best |",
            "|---|---|" + "---|" * len(gen[0]["results"]) + "---|"]
    for e in gen:
        res = e["results"]; base = next((x["us"] for x in ok(res) if x["var"] == "base"), None)
        cells = [f"{x['us']:.0f} ({base / x['us']:.2f}x)" if "us" in x and base else (x.get("err", "")[:60]) for x in res]
        b = best(res); n = e["n_calls"]
        out.append(f"| {e['name']} | {n} | " + " | ".join(cells) + f" | {base * n / 1e6:.2f} -> {b['us'] * n / 1e6:.2f} |" if base and b
                   else f"| {e['name']} | {n} | " + " | ".join(cells) + " | - |")

for e in [e for e in ev if e["ev"] == "chain"]:
    res = {x["var"]: x for x in e["results"]}
    base = res.get("base", {}).get("us")
    out += ["", f"## chain: {e['name']} (#{e['rank']}, {e['n_total']} calls/fold), device us", "| arm | us | vs base |", "|---|---|---|"]
    out += [f"| {k} | {x['us']:.0f} | {base / x['us']:.2f}x |" if "us" in x and base else f"| {k} | {x.get('us', x.get('err', ''))} | |"
            for k, x in res.items()]

for e in [e for e in ev if e["ev"] == "opm"]:
    base = next(x["us"] for x in ok(e["results"]) if x["var"] == "base")
    out += ["", f"## opm: z_rows #5 {e['n_total']} calls/fold, base {base:.0f} us, device us (errors folded)", "| arm | us | x |", "|---|---|---|"]
    errs = {}
    for x in e["results"]:
        if "us" in x: out.append(f"| {x['var']} | {x['us']:.0f} | {base / x['us']:.2f} |")
        else: errs.setdefault(x["err"][:120], []).append(x["var"])
    out += [f"| {len(v)} arms failed ({v[0]} ...) | {k} | |" for k, v in errs.items()]

(RUNS[0] / "summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
