"""Turn census.py's id-attributed records into the fold census: op table, classes, roofline, gaps, ops.json.

Joins progs_<fold>.jsonl to ops_<fold>.jsonl by runtime id, sums device kernel time per op signature, places
each signature on the measured Wormhole roofline (roof.json from roofline_wh.py) and groups into classes.
If the profiled fold ran fewer cycles/steps than the real one, trunk-loop and denoise calls are scaled by
call count (cycles beyond the first x (CF-1)/(C-1), steps beyond the first x (SF-1)/(S-1)).

usage: analyze.py RUN_DIR FOLD ROOF.json CYCLES_FULL STEPS_FULL OUT_PREFIX
"""
import json, re, sys, collections
from pathlib import Path
import numpy as np

RUN = Path(sys.argv[1]); FOLD = sys.argv[2]; ROOF = json.load(open(sys.argv[3]))
CF = int(sys.argv[4]); SF = int(sys.argv[5]); OUTP = Path(sys.argv[6])
meta = next(e for e in map(json.loads, open(RUN / "census.jsonl")) if e.get("ev") == "fold" and e["fold"] == FOLD)
C, S = meta["cycles"], meta["steps"]

# ---------- signatures, calls, programs ----------
sigs = {}
for l in open(RUN / f"sig_{FOLD}.jsonl"):
    e = json.loads(l)
    if "op" in e: sigs[e["sig"]] = e
    else: sigs[e["sig"]]["out"] = e["out"]
calls = np.array([json.loads(l) for l in open(RUN / f"ops_{FOLD}.jsonl")], dtype=np.int64)  # sig,id0,id1,cyc,step
order = np.argsort(calls[:, 1], kind="stable"); calls = calls[order]
P = np.array([[x if x is not None else -1 for x in json.loads(l)] for l in open(RUN / f"progs_{FOLD}.jsonl")],
             dtype=np.float64)
# rid, batch, cores, k, fw, t0, t1, t2, br, nc, start, end
RID, BAT, CORES, K, FW, T0, T1, T2, BR, NC, ST, EN = range(12)
P[:, RID] = np.floor(P[:, RID] / 1024)   # runtime id = device op id << 10
idx = np.searchsorted(calls[:, 1], P[:, RID], side="right") - 1
ok = (idx >= 0) & (P[:, RID] < calls[np.clip(idx, 0, None), 2])
print(f"programs {len(P)}, attributed {ok.sum()}, unattributed {(~ok).sum()} "
      f"(kernel {P[~ok, K].sum() / 1e9:.3f} s)")
P = P[ok]; idx = idx[ok]
expected = int((calls[:, 2] - calls[:, 1]).sum())

def is_reg(sig, name): return name in sigs[sig]["reg"].split("/")
def weight(sig, cyc, step):
    if step >= 0 and is_reg(sig, "denoise"):
        return 1.0 if step == 0 or S <= 1 else (SF - 1) / (S - 1)
    if cyc >= 0 and is_reg(sig, "trunk") and "confidence" not in sigs[sig]["reg"]:
        return 1.0 if cyc == 0 or C <= 1 else (CF - 1) / (C - 1)
    return 1.0
cw = {}
wcall = np.array([cw.setdefault((s, c, t), weight(s, c, t)) for s, c, t in calls[:, [0, 3, 4]].tolist()])
pw = wcall[idx]; psig = calls[idx, 0]

# ---------- device-idle gaps: consecutive programs inside one flush batch ----------
# timestamps are device cycles; convert with the measured ns/cycle of the kernel duration fields
sp = P[:, EN] - P[:, ST]; good = (sp > 0) & (P[:, K] > 0)
ns_per_tick = float(np.median(P[good, K] / sp[good]))
o = np.lexsort((P[:, ST], P[:, BAT]))
same = P[o[1:], BAT] == P[o[:-1], BAT]
gap = np.where(same, (P[o[1:], ST] - np.maximum.accumulate(P[o, EN])[:-1]) * ns_per_tick, 0.0)
gap_before = np.zeros(len(P)); gap_before[o[1:]] = np.clip(gap, 0, None)
span = sum((P[P[:, BAT] == b, EN].max() - P[P[:, BAT] == b, ST].min()) * ns_per_tick
           for b in np.unique(P[:, BAT])) if len(P) else 0.0

# ---------- roofs ----------
def roof_tflops(dt, fid, acc=False):
    v = [r["tflops"] for r in ROOF["matmul"] if "tflops" in r and r["dtype"] == dt and r["fid"] == fid
         and r["fp32_acc"] == acc]
    return max(v) if v else None
BW = max(r["gbs"] for r in ROOF["bw"])
BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024, "UINT32": 4, "INT32": 4,
         "UINT16": 2, "UINT8": 1}
DTN = {"BFLOAT16": "bf16", "BFLOAT8_B": "bfp8", "BFLOAT4_B": "bfp4", "FLOAT32": "fp32"}

def tensors(x):
    if isinstance(x, dict) and "T" in x: yield x["T"]
    elif isinstance(x, list):
        for y in x: yield from tensors(y)
    elif isinstance(x, dict):
        for y in x.values(): yield from tensors(y)
def nbytes(t):
    if "padded" not in t: return 0
    return int(np.prod(t["padded"])) * BYTES.get(t["dtype"].upper(), 2)
def ckcfg(e):
    for v in e.get("kw", {}).values():
        if isinstance(v, dict) and "CKC" in v: return v["CKC"]
    return {}
def kwstr(e): return " ".join(v for v in e.get("kw", {}).values() if isinstance(v, str)) + " " + \
    " ".join(a for a in e.get("args", []) if isinstance(a, str))
def fidelity(e):
    if ckcfg(e).get("math_fidelity"): return ckcfg(e)["math_fidelity"]
    m = re.search(r"math_fidelity=(?:MathFidelity::)?(\w+)", kwstr(e)); return m.group(1) if m else None
def fp32acc(e):
    if "fp32_dest_acc_en" in ckcfg(e): return ckcfg(e)["fp32_dest_acc_en"].lower()
    m = re.search(r"fp32_dest_acc_en=(\w+)", kwstr(e)); return m.group(1) if m else None
def program_config(e):
    for k, v in e.get("kw", {}).items():
        if "program_config" in k and isinstance(v, str): return v
    return None

MM = ("ttnn.matmul", "ttnn.linear", "ttnn.experimental.minimal_matmul")
def flops(e):
    """Useful FLOPs from padded operand shapes: matmul family and SDPA; elementwise is not math worth a roof."""
    op = e["op"]; ins = list(tensors(e.get("args", []))) + list(tensors(e.get("kw", {})))
    outs = list(tensors(e.get("out")))
    if not ins or not outs: return 0
    if op in MM and len(ins) >= 2:
        return 2 * int(np.prod(outs[0]["padded"])) * ins[0]["padded"][-1]
    if "scaled_dot_product_attention" in op and len(ins) >= 2:
        q, k = ins[0]["padded"], ins[1]["padded"]
        return 4 * int(np.prod(q[:-1])) * k[-2] * q[-1]
    return 0

def classify(e):
    reg = e["reg"].split("/"); op = e["op"].split(".")[-1]
    if "confidence" in reg or "conf_zbase" in reg: pre = "confidence: "
    else: pre = ""
    if "triatt" in reg: c = "triangle attention"
    elif "trimul" in reg: c = "triangle multiplication"
    elif "opm" in reg: c = "outer product mean"
    elif "pwa" in reg or "pwa_w" in reg: c = "pair-weighted averaging (MSA row attn w/ pair bias)"
    elif "atom_tx" in reg or "atom_tx_bias" in reg or "input_aae" in reg or "atom_cond" in reg: c = "atom transformer/encoder"
    elif "apb" in reg and ("denoise" in reg or "dit" in reg): c = "diffusion attention (token DiT)"
    elif "apb" in reg: c = "pairformer attention (pair bias)"
    elif "transition" in reg: c = "transitions"
    elif op in ("matmul", "linear", "minimal_matmul"): c = "linear/projection (other)"
    elif re.search(r"norm", op): c = "layernorm"
    elif op == "<unhooked>": c = "unhooked (tensor methods)"
    elif re.search(r"typecast|to_layout|tilize|untilize|reshape|permute|transpose|pad|slice|concat|to_memory_config|"
                   r"reallocate|clone|copy|move|sharded|interleaved|fill|repeat|expand|split|chunk|unsqueeze|"
                   r"squeeze|embedding|view|gather|from_torch|to_device|assign", op): c = "typecast/layout/data movement"
    else: c = "elementwise/other"
    return pre + c if pre and c not in ("unhooked (tensor methods)",) else c
def phase(e):
    reg = e["reg"].split("/")
    if "confidence" in reg or "conf_zbase" in reg: return "confidence"
    if "msa" in reg: return "trunk: MSA module"
    if "template" in reg: return "trunk: template"
    if "pairformer" in reg and "trunk" in reg: return "trunk: pairformer"
    if "trunk" in reg: return "trunk: other"
    if "denoise" in reg or "sampler" in reg: return "diffusion"
    return "other (input embed, conditioning setup)"

# ---------- per-signature aggregation ----------
ns = max(sigs) + 1
def agg(col): return np.bincount(psig, weights=P[:, col] * pw, minlength=ns)
kw_ = agg(K); t1w = agg(T1); gapw = np.bincount(psig, weights=gap_before * pw, minlength=ns)
k_raw = np.bincount(psig, weights=P[:, K], minlength=ns)
nprog = np.bincount(psig, minlength=ns)
ncall_raw = np.bincount(calls[:, 0], minlength=ns); ncall_w = np.bincount(calls[:, 0], weights=wcall, minlength=ns)
cores = np.zeros(ns); np.maximum.at(cores, psig, P[:, CORES])
total = kw_.sum(); total_gap = gapw.sum()

rows = []
for s in np.argsort(-kw_):
    if kw_[s] <= 0: break
    e = sigs[int(s)]; n = ncall_raw[s]; mean_k = k_raw[s] / max(n, 1)
    ins = list(tensors(e.get("args", []))) + list(tensors(e.get("kw", {}))); outs = list(tensors(e.get("out")))
    # an in-place op (trailing underscore) writes its first input back, so its output is not extra traffic
    by = sum(nbytes(t) for t in ins) + (0 if e["op"].endswith("_") else sum(nbytes(t) for t in outs)); fl = flops(e)
    fid = fidelity(e); acc = fp32acc(e) == "true"
    dts = [t.get("dtype", "").upper() for t in ins[:2]]
    dt = "bfp8" if "BFLOAT8_B" in dts else "bfp4" if "BFLOAT4_B" in dts else "bf16"
    pk = (roof_tflops(dt, fid or "HiFi2", acc) or roof_tflops(dt, fid or "HiFi2")) if fl else None
    if fl and "FLOAT32" in dts: pk = pk / 4 if pk else None   # fp32 operands run the fpu at a quarter rate
    t_math = fl / (pk * 1e12) * 1e9 if (fl and pk) else 0.0
    t_mem = by / (BW * 1e9) * 1e9
    roof = max(t_math, t_mem); eff = roof / mean_k if mean_k else 0.0
    lim = "math" if t_math > t_mem else "bw"
    bound = lim if eff >= 0.35 else "overhead"
    rows.append(dict(sig=int(s), e=e, cls=classify(e), phase=phase(e), kw=kw_[s], n=int(n), nw=float(ncall_w[s]),
                     mean_k=mean_k, by=by, fl=fl, fid=fid, acc=fp32acc(e), dt=dt, pk=pk, t_math=t_math, t_mem=t_mem,
                     eff=eff, lim=lim, bound=bound, gap=gapw[s], t1=t1w[s], nprog=nprog[s] / max(n, 1),
                     cores=int(cores[s])))

def tab(key):
    c = collections.Counter()
    for r in rows: c[r[key]] += r["kw"]
    return {k: dict(s=v / 1e9, share=v / total) for k, v in c.most_common()}
cls_gap = collections.Counter()
for r in rows: cls_gap[r["cls"]] += r["gap"]
bound_cls = collections.defaultdict(collections.Counter)
for r in rows: bound_cls[r["cls"]][r["bound"]] += r["kw"]

# ---------- Amdahl ceiling for a smaller format ----------
pk_bf16 = {f: roof_tflops("bf16", f) for f in ("LoFi", "HiFi2", "HiFi4")}
best_lp = max(v for v in (roof_tflops("bfp8", "LoFi"), roof_tflops("bfp4", "LoFi")) if v)
def ceiling(rows, math_x, bw_x, ovh_x=1.0, gaps=0.0):
    t = sum(r["kw"] / (math_x(r) if r["bound"] == "math" else bw_x if r["bound"] == "bw" else ovh_x) for r in rows)
    return (total + gaps) / (t + gaps)
math_bfp8 = lambda r: (roof_tflops("bfp8", "LoFi") or 1) / (r["pk"] or roof_tflops("bf16", "HiFi4"))
math_best = lambda r: best_lp / (r["pk"] or roof_tflops("bf16", "HiFi4"))
flop_rows = [r for r in rows if r["fl"]]
amdahl = dict(
    bfp8_lofi_bound_ops=ceiling(rows, math_bfp8, 2 / (1088 / 1024)),
    bfp4_lofi_bound_ops=ceiling(rows, math_best, 2 / (576 / 1024)),
    every_flop_op_at_bfp4_lofi_peak=(total) / (sum(r["kw"] for r in rows if not r["fl"]) +
                                               sum(r["fl"] / (best_lp * 1e12) * 1e9 * r["nw"] for r in flop_rows)),
    with_gaps=dict(bfp8=ceiling(rows, math_bfp8, 2 / (1088 / 1024), gaps=total_gap),
                   bfp4=ceiling(rows, math_best, 2 / (576 / 1024), gaps=total_gap)))

out = dict(run=str(RUN), fold=FOLD, cycles_profiled=C, steps_profiled=S, cycles_full=CF, steps_full=SF,
           wall_s=meta["wall_s"], aiclk=dict(median=meta["aiclk_median"], min=meta["aiclk_min"], max=meta["aiclk_max"],
                                             n=meta["aiclk_n"]),
           n_calls=len(calls), n_sigs=len(sigs), n_programs=len(P), ids_expected=expected, missing=meta["missing"],
           device_kernel_s=total / 1e9, device_idle_gap_s=total_gap / 1e9, batch_span_s=span / 1e9,
           ns_per_tick=ns_per_tick, peak_bw_gbs=BW,
           peaks={f"{dt}/{f}": roof_tflops(dt, f) for dt in ("bf16", "bfp8", "bfp4") for f in ("LoFi", "HiFi2", "HiFi4")},
           phases=tab("phase"), classes=tab("cls"),
           class_gaps_s={k: v / 1e9 for k, v in cls_gap.most_common()},
           bound=tab("bound"),
           bound_by_class={k: {b: round(v / total, 4) for b, v in c.items()} for k, c in bound_cls.items()},
           amdahl=amdahl)
json.dump(out, open(f"{OUTP}_summary.json", "w"), indent=1, default=str)

spec = []; acc_ = 0.0
for r in rows:
    e = r["e"]
    spec.append(dict(rank=len(spec) + 1, op=e["op"], cls=r["cls"], phase=r["phase"], call_site=e["site"], region=e["reg"],
                     args=e.get("args"), kwargs=e.get("kw"), out=e.get("out"),
                     math_fidelity=r["fid"], fp32_dest_acc=r["acc"], program_config=program_config(e),
                     calls_profiled=r["n"], calls_full_fold=round(r["nw"], 1),
                     device_us_per_call=r["mean_k"] / 1e3, device_s_full_fold=r["kw"] / 1e9, share=r["kw"] / total,
                     idle_gap_before_s_full_fold=r["gap"] / 1e9, programs_per_call=r["nprog"], cores=r["cores"],
                     trisc1_math_over_kernel=r["t1"] / r["kw"] if r["kw"] else None,
                     flops_per_call=r["fl"], bytes_per_call=r["by"],
                     achieved_tflops=r["fl"] / r["mean_k"] / 1e3 if r["fl"] else 0.0,
                     achieved_gbs=r["by"] / r["mean_k"] if r["mean_k"] else 0.0,
                     roof=dict(dtype=r["dt"], peak_tflops=r["pk"], peak_gbs=BW, t_math_us=r["t_math"] / 1e3,
                               t_mem_us=r["t_mem"] / 1e3, fraction_of_roof=r["eff"], limiter=r["lim"]),
                     bound=r["bound"]))
    acc_ += r["kw"] / total
    if acc_ >= 0.85 and len(spec) >= 10: break
json.dump(dict(version=2, note="Protenix-v2 730 tok (580+150), MSA 9947 deep, 5 samples, 10 cycles, 200 steps; one "
               "Wormhole chip (.107 chip 24); device kernel time from the tt-metal device profiler, programs joined "
               "to ttnn calls by runtime id; signature = op + call site + region + arg shapes/dtypes/configs",
               peaks=out["peaks"], peak_bw_gbs=BW, aiclk=out["aiclk"], device_kernel_s=total / 1e9,
               coverage=acc_, n_ops=len(spec), ops=spec), open(f"{OUTP}_ops.json", "w"), indent=1, default=str)

print(json.dumps({k: out[k] for k in ("wall_s", "device_kernel_s", "device_idle_gap_s", "batch_span_s", "n_calls",
                                      "n_sigs", "n_programs", "ids_expected", "missing", "aiclk", "amdahl")},
                 default=str, indent=None))
print("coverage", round(acc_, 4), "ops", len(spec))
for t in ("phases", "classes", "bound"):
    for k, v in out[t].items(): print(f"{t[:5]} {v['share']*100:6.2f} % {v['s']:8.2f} s  {k}")
for s in spec[:50]:
    print(f"{s['rank']:3d} {s['share']*100:5.2f}% {s['device_us_per_call']:9.1f}us x{s['calls_full_fold']:8.0f} "
          f"{s['op'][:38]:38s} {s['cls'][:26]:26s} {s['bound']:8s} roof {s['roof']['fraction_of_roof']:.2f} "
          f"{s['achieved_tflops']:.1f}TF {s['achieved_gbs']:.0f}GB/s {s['math_fidelity']} {(s['call_site'] or [''])[0]}")
