"""Turn an lpx_matmul run (calls.json + bench.jsonl) into the SHAPES / SWEEP / BEST / CHAIN tables.

usage: analyze.py RUN_DIR [FOLD_S]   -> RUN_DIR/summary.md and RUN_DIR/summary.json
FOLD_S, default the capture fold's own wall time, is the denominator of "share of fold".
Roofline: Wormhole 8x9 cores at the AICLK the sweep logged, 4096 FLOP/cycle/core at LoFi (/2 HiFi2, /3 HiFi3,
/4 HiFi4), DRAM 288 GB/s; operands whose memory config is L1 are not charged DRAM bytes.
"""
import json, re, sys
from math import prod
from pathlib import Path

RUN = Path(sys.argv[1]); FOLD_S = float(sys.argv[2]) if len(sys.argv) > 2 else None
BYTES = {"BFLOAT16": 2.0, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "FLOAT32": 4.0}
FIDDIV = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
CORES, DRAM = 72, 288e9
T = re.compile(r"(a\d|[a-z_]+)=\('T', \(([\d, ]*)\), 'DataType\.(\w+)', 'Layout\.(\w+)', (MemoryConfig\(.*?\)|None)\)")

# Call-site function -> class, from engine 7c080f158 (the .107 serving engine): head_out/heads_over are
# PairWeightedAveraging, z_rows/project_ab/outer_product_mean are OuterProductMean, _multiply/_in_proj_matmul/
# fused_tail/reblock_* are TriangleMultiplication, _narrow_proj_linear < whole is AttentionPairBias.
CLASSES = [("transition", ("_transition", "swiglu")), ("pwa", ("head_out", "heads_over")),
           ("opm", ("z_rows", "project_ab", "outer_product_mean", "contiguous_ab")),
           ("trimul", ("_multiply", "_in_proj_matmul", "fused_tail", "_transform_chunk_gated", "_channel_move_back")),
           ("triatt_proj", ("_fused_qkvgb", "qkvgb_heads", "gate_and_project")),
           ("atom_tx", ("_attention_m", "_block_m", "_adaln")),
           ("dit_attn", ("batched_matmul < tenstorrent.py:10044", "batched_matmul < tenstorrent.py:10050")),
           ("diffusion_proj", ("_token_dit_device", "_denoise")),
           ("apb_proj", ("_narrow_proj_linear",)), ("confidence", ("confidence",)), ("template", ("template",))]

def classify(sites):
    s = " ".join(sites)
    for name, keys in CLASSES:
        if any(k in s for k in keys):
            return name
    return "other"

def shapes(key):
    return {m.group(1): (tuple(int(x) for x in m.group(2).split(",") if x.strip()), m.group(3), "L1" in m.group(5))
            for m in T.finditer(key)}

def roof(key, out_dt=None, in_dt=(None, None), fid="HiFi4", clk_mhz=1000):
    sh = shapes(key)
    a = sh.get("a0") or sh.get("input_tensor") or sh.get("input_tensor_a")
    b = sh.get("a1") or sh.get("weight_tensor") or sh.get("input_tensor_b")
    if not a or not b:
        return None
    (ash, adt, al1), (bsh, bdt, bl1) = a, b
    tb = "transpose_b=True" in key
    K = ash[-1]; N = bsh[-2] if tb else bsh[-1]
    batch = max(prod(ash[:-2]) if len(ash) > 2 else 1, prod(bsh[:-2]) if len(bsh) > 2 else 1)
    M = ash[-2] if len(ash) > 1 else 1
    flop = 2.0 * batch * M * K * N
    od = out_dt or (re.search(r"dtype=<DataType\.(\w+)", key) or [None, adt])[1]
    byt = (0 if al1 else prod(ash) * BYTES[in_dt[0] or adt]) + (0 if bl1 else prod(bsh) * BYTES[in_dt[1] or bdt]) \
        + batch * M * N * BYTES.get(od, 2)
    peak = CORES * 4096 / FIDDIV.get(fid, 4) * clk_mhz * 1e6
    return dict(M=batch * M, K=K, N=N, gflop=flop / 1e9, mb=byt / 1e6, ai=flop / max(byt, 1),
                t_math_us=flop / peak * 1e6, t_dram_us=byt / DRAM * 1e6, in0=f"{adt}{'(L1)' if al1 else ''}",
                in1=f"{bdt}{'(L1)' if bl1 else ''}", shape_in0=ash, shape_in1=bsh)

calls = json.loads((RUN / "calls.json").read_text())
ev = [json.loads(l) for l in open(RUN / "bench.jsonl")]
sweeps = {e["key"]: e for e in ev if e.get("ev") == "sweep"}
folds = {e["kind"]: e for e in ev if e.get("ev") == "fold"}
cap_s = folds.get("capture", {}).get("wall_s")
FOLD_S = FOLD_S or cap_s   # the synced times were taken in the capture fold, so it is their denominator

out = ["# lpx-matmul summary: " + str(RUN), ""]
out.append(f"capture fold (synced per call) {cap_s and round(cap_s, 1)} s; share denominator {FOLD_S:.1f} s.")
cls_tot = {}
rows = []
for c in calls:
    cl = classify(c["sites"]) + ("/generic_op" if c["op"] == "generic_op" else "")
    cls_tot.setdefault(cl, [0, 0.0]); cls_tot[cl][0] += c["n_total"]; cls_tot[cl][1] += c["est_total_s"]
    sw = sweeps.get(c["key"]); fid = (sw or {}).get("ckc", {}).get("math_fidelity", "HiFi4")
    r = roof(c["key"], fid=fid if fid in FIDDIV else "HiFi4", clk_mhz=(sw or {}).get("aiclk_median") or 1000) \
        if c["op"] != "generic_op" else None
    rows.append((c, cl, r, sw))

out += ["", "## SHAPES (call classes)", "| class | calls | est. s (synced) | share of fold |", "|---|---|---|---|"]
for cl, (n, s) in sorted(cls_tot.items(), key=lambda kv: -kv[1][1]):
    out.append(f"| {cl} | {n} | {s:.1f} | {100 * s / FOLD_S:.1f} % |")

out += ["", "## Signatures (heaviest first)",
        "| # | class | op | in0 | in1 | M x K x N | calls | med us (synced) | est s | GFLOP | MB | FLOP/B | math us | dram us | site |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, (c, cl, r, sw) in enumerate(rows[:60]):
    site = max(c["sites"], key=c["sites"].get).split(" < ")[0] if c["sites"] else ""
    if r:
        out.append(f"| {i} | {cl} | {c['op']} | {list(r['shape_in0'])} {r['in0']} | {list(r['shape_in1'])} {r['in1']} | "
                   f"{r['M']}x{r['K']}x{r['N']} | {c['n_total']} | {1e6 * (c['med_s'] or 0):.0f} | {c['est_total_s']:.2f} | "
                   f"{r['gflop']:.2f} | {r['mb']:.1f} | {r['ai']:.0f} | {r['t_math_us']:.0f} | {r['t_dram_us']:.0f} | {site} |")
    else:
        out.append(f"| {i} | {cl} | {c['op']} | | | | {c['n_total']} | {1e6 * (c['med_s'] or 0):.0f} | "
                   f"{c['est_total_s']:.2f} | | | | | | {site} |")

out += ["", "## SWEEP (per swept signature: variant -> device us, speedup vs base)"]
best = []
for i, (c, cl, r, sw) in enumerate(rows):
    if not sw:
        continue
    res = [x for x in sw["results"] if "us" in x and x.get("finite")]
    base = next((x for x in res if x["var"] == "base"), None)
    out.append(f"\n### #{i} {cl} {c['op']} {r and (str(r['M']) + 'x' + str(r['K']) + 'x' + str(r['N']))} "
               f"calls {c['n_total']} ckc {sw['ckc']} aiclk {sw.get('aiclk_median')}")
    out.append("| variant | us | spread | speedup | mode |"); out.append("|---|---|---|---|---|")
    for x in sorted(sw["results"], key=lambda x: x.get("us", 1e18)):
        if "us" in x:
            out.append(f"| {x['var']} | {x['us']:.1f} | {100 * x['spread']:.1f} % | {x.get('speedup', 0):.3f} | {x['mode']}{'' if x.get('finite') else ' NONFINITE'} |")
        else:
            out.append(f"| {x['var']} | {x.get('err') or x.get('skip')} | | | |")
    if sw.get("chain"):
        out.append("chain: " + json.dumps({k: (round(v["us"], 1) if "us" in v else v.get("err", "")[:80]) for k, v in sw["chain"].items()}))
    if base and res:
        bx = min(res, key=lambda x: x["us"])
        ch = sw.get("chain", {})
        cast = 0.0
        if bx["in0"] in ("b8", "b4"):
            cast += ch.get(f"in_to_{bx['in0']}", {}).get("us", 0.0)
        if bx["out"] == "b8":
            cast += ch.get("out_b8_to_bf16", {}).get("us", 0.0)
        best.append(dict(rank=i, cls=cl, op=c["op"], calls=c["n_total"], base_us=base["us"], best=bx["var"], best_us=bx["us"],
                         speedup=base["us"] / bx["us"], cast_us=cast, chain_speedup=base["us"] / (bx["us"] + cast),
                         saved_s=(base["us"] - bx["us"]) * c["n_total"] / 1e6,
                         ai=r and r["ai"], t_math_us=r and r["t_math_us"], t_dram_us=r and r["t_dram_us"]))

out += ["", "## BEST", "| # | class | calls | base us | best variant | best us | speedup | +cast us | chain speedup | fold s saved (no cast) | FLOP/B | math us | dram us |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for b in best:
    out.append(f"| {b['rank']} | {b['cls']} | {b['calls']} | {b['base_us']:.0f} | {b['best']} | {b['best_us']:.0f} | {b['speedup']:.2f} | "
               f"{b['cast_us']:.0f} | {b['chain_speedup']:.2f} | {b['saved_s']:.1f} | {b['ai'] or 0:.0f} | {b['t_math_us'] or 0:.0f} | {b['t_dram_us'] or 0:.0f} |")
(RUN / "summary.md").write_text("\n".join(out) + "\n")
(RUN / "summary.json").write_text(json.dumps(dict(classes=cls_tot, best=best), indent=1, default=str))
print("\n".join(out[:80]))
