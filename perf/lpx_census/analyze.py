"""Turn census.py's per-op records into the fold census: scaled op table, classes, roofline, ops.json.

Scaling to the real fold (10 trunk cycles, 200 denoise steps) by call count:
  trunk-loop records:   cycle 0 x1, cycles 1..C-1 x (10-1)/(C-1)
  denoise records:      step 0 x1,  steps 1..S-1 x (200-1)/(S-1)
  everything else x1 (featurization, trunk setup, sampler setup, confidence).
Step 0 and cycle 0 keep weight 1 so one-time work done lazily inside them is counted once.

usage: analyze.py RUN_DIR ROOF.json CYCLES_FULL STEPS_FULL OUT_PREFIX
"""
import json, re, sys, collections
from pathlib import Path

RUN = Path(sys.argv[1]); ROOF = json.load(open(sys.argv[2])); CF = int(sys.argv[3]); SF = int(sys.argv[4])
OUTP = Path(sys.argv[5])
recs = [json.loads(l) for l in open(RUN / "ops_op.jsonl")]
meta = {}
for l in open(RUN / "census.jsonl"):
    e = json.loads(l)
    if e.get("ev") == "fold": meta[e["mode"]] = e
C = meta["op"]["cycles"]; S = meta["op"]["steps"]

# ---------- roofs (measured on this chip) ----------
def roof_tflops(dt, fid):
    v = [r["tflops"] for r in ROOF["matmul"] if r["dtype"] == dt and r["fid"] == fid and not r["fp32_acc"]]
    return max(v) if v else None
BW = max(r["gbs"] for r in ROOF["bw"])  # GB/s, best of add/clone at large sizes
BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024, "UINT32": 4, "INT32": 4,
         "UINT16": 2, "UINT8": 1}

def weight(r):
    reg = r["reg"]
    if "denoise" in reg.split("/") and r["step"] >= 0:
        return 1.0 if r["step"] == 0 else (SF - 1) / (S - 1)
    if "trunk" in reg.split("/") and r["cyc"] >= 0 and "confidence" not in reg:
        return 1.0 if r["cyc"] == 0 else (CF - 1) / (C - 1)
    return 1.0

def tensors(x):
    if isinstance(x, dict) and "T" in x: yield x["T"]
    elif isinstance(x, list):
        for y in x: yield from tensors(y)
    elif isinstance(x, dict):
        for y in x.values(): yield from tensors(y)
def nbytes(t):
    if "padded" not in t: return 0
    n = 1
    for s in t["padded"]: n *= s
    return n * BYTES.get(t["dtype"].upper(), 2)
def prod(v):
    n = 1
    for s in v: n *= s
    return n

def kwval(r, key):
    v = r.get("kw", {}).get(key)
    return v if isinstance(v, str) else None
def fidelity(r):
    for v in r.get("kw", {}).values():
        if isinstance(v, str):
            m = re.search(r"math_fidelity=MathFidelity::(\w+)|math_fidelity=(\w+)", v)
            if m: return m.group(1) or m.group(2)
    return None
def fp32acc(r):
    for v in r.get("kw", {}).values():
        if isinstance(v, str):
            m = re.search(r"fp32_dest_acc_en=(\w+)", v)
            if m: return m.group(1)
    return None

def flops(r):
    """Useful FLOPs from the (padded) operand shapes. Only matmul-family and SDPA do math worth counting."""
    op = r["op"]; ins = list(tensors(r.get("args", []))) + list(tensors(r.get("kw", {})))
    outs = list(tensors(r.get("out")))
    if not ins or not outs: return 0
    if op in ("ttnn.matmul", "ttnn.linear", "ttnn.experimental.minimal_matmul") and len(ins) >= 2:
        a, b = ins[0]["padded"], ins[1]["padded"]
        K = a[-1]; o = outs[0]["padded"]
        return 2 * prod(o) * K
    if "scaled_dot_product_attention" in op:
        q, k = ins[0]["padded"], ins[1]["padded"]
        return 4 * prod(q[:-1]) * k[-2] * q[-1]
    return 0

def classify(r):
    reg = r["reg"].split("/"); op = r["op"].split(".")[-1]
    if "triatt" in reg: return "triangle attention"
    if "trimul" in reg: return "triangle multiplication"
    if "opm" in reg: return "outer product mean"
    if "pwa" in reg or "pwa_w" in reg: return "pair-weighted averaging (MSA row attn w/ pair bias)"
    if "apb" in reg and "denoise" not in reg and "atom_tx" not in reg: return "pairformer attention (pair bias)"
    if "denoise" in reg and ("apb" in reg or "scaled_dot_product_attention" in r["op"] or "atom_tx" in reg) \
            and re.search(r"matmul|linear|attention|softmax", r["op"]): return "diffusion attention"
    if "transition" in reg: return "transitions"
    if op in ("matmul", "linear", "minimal_matmul"): return "linear/projection (other)"
    if re.search(r"norm", op): return "layernorm"
    if re.search(r"typecast|to_layout|tilize|untilize|reshape|permute|transpose|pad|slice|concat|to_memory_config|"
                 r"reallocate|clone|copy|move|sharded|interleaved|fill|repeat|expand|split|chunk|unsqueeze|squeeze|"
                 r"embedding|view|gather|from_torch|to_device", op): return "typecast/layout/data movement"
    if op == "<unhooked>": return "unhooked (tensor methods)"
    return "elementwise/other"

ops = []
for r in recs:
    if not r.get("progs"): continue
    k = sum(p.get("k", 0) for p in r["progs"]); w = weight(r)
    ins = list(tensors(r.get("args", []))) + list(tensors(r.get("kw", {})))
    outs = list(tensors(r.get("out")))
    by = sum(nbytes(t) for t in ins) + sum(nbytes(t) for t in outs)
    t1 = sum(p.get("t1", 0) for p in r["progs"])
    ops.append(dict(r=r, k=k, w=w, by=by, fl=flops(r), cls=classify(r), t1=t1,
                    cores=max(p.get("cores", 0) for p in r["progs"]), nprog=len(r["progs"])))

total = sum(o["k"] * o["w"] for o in ops)
raw = sum(o["k"] for o in ops)

def sig(o):
    r = o["r"]
    ins = [(t.get("padded"), t.get("dtype"), t.get("layout"), t.get("mem", "")[:200]) for t in tensors(r.get("args", []))]
    kws = {k: v for k, v in r.get("kw", {}).items() if not isinstance(v, dict)}
    return json.dumps([r["op"], r.get("site", [None])[0] if r.get("site") else None, ins,
                       sorted((k, str(v)[:300]) for k, v in kws.items())], default=str)
groups = collections.OrderedDict()
for o in ops:
    g = groups.setdefault(sig(o), dict(o=o, n=0, nw=0.0, k=0.0, kw=0.0, t1=0.0, regs=collections.Counter()))
    g["n"] += 1; g["nw"] += o["w"]; g["k"] += o["k"]; g["kw"] += o["k"] * o["w"]; g["t1"] += o["t1"]
    g["regs"][o["r"]["reg"]] += 1
rows = sorted(groups.values(), key=lambda g: -g["kw"])

def roofline(o, mean_k_ns):
    r = o["r"]; fid = fidelity(r) or "HiFi4"
    ins = list(tensors(r.get("args", [])))
    dt = {"BFLOAT8_B": "bfp8", "BFLOAT4_B": "bfp4"}.get(ins[1]["dtype"].upper() if len(ins) > 1 else "", "bf16")
    pk = roof_tflops(dt, fid) or roof_tflops("bf16", fid)
    if any(t.get("dtype", "").upper() == "FLOAT32" for t in ins[:2]):
        pk = (pk or 0) / 4 or None  # fp32 operands: no measured fp32 roof; see notes
    t_math = o["fl"] / (pk * 1e12) * 1e9 if (o["fl"] and pk) else 0.0
    t_mem = o["by"] / (BW * 1e9) * 1e9
    roof = max(t_math, t_mem)
    eff = roof / mean_k_ns if mean_k_ns else 0
    lim = "math" if t_math > t_mem else "bw"
    bound = lim if eff >= 0.35 else "overhead"
    return dict(peak_tflops=pk, t_math_ns=t_math, t_mem_ns=t_mem, roof_eff=eff, limiter=lim, bound=bound,
                tflops=(o["fl"] / mean_k_ns / 1e3) if o["fl"] else 0.0, gbs=o["by"] / mean_k_ns)

cls = collections.Counter(); bound = collections.Counter(); bound_cls = collections.defaultdict(collections.Counter)
for g in rows:
    o = g["o"]; mk = g["k"] / g["n"]; rl = roofline(o, mk); g["rl"] = rl
    cls[o["cls"]] += g["kw"]; bound[rl["bound"]] += g["kw"]; bound_cls[o["cls"]][rl["bound"]] += g["kw"]

# ---------- pipelined fold: device-idle gaps ----------
gap = None
if (RUN / "ops_pipe.jsonl").exists():
    busy = idle = 0.0; span = 0.0
    for l in open(RUN / "ops_pipe.jsonl"):
        r = json.loads(l); ps = [p for p in r["progs"] if "s" in p]
        ps.sort(key=lambda p: p["s"])
        for a, b in zip(ps, ps[1:]):
            g_ = b["s"] - a["e"]
            if g_ > 0: idle += g_
        busy += sum(p["k"] for p in ps)
        if ps: span += ps[-1]["e"] - ps[0]["s"]
    gap = dict(busy_ns_raw=busy, idle_ns_within_batches=idle, span_ns=span, wall_s=meta.get("pipe", {}).get("wall_s"))

out = dict(run=str(RUN), cycles_profiled=C, steps_profiled=S, cycles_full=CF, steps_full=SF,
           aiclk=dict((m, dict(median=meta[m]["aiclk_median"], min=meta[m]["aiclk_min"], max=meta[m]["aiclk_max"]))
                      for m in meta),
           device_ns_profiled=raw, device_ns_full=total, n_records=len(recs), n_records_with_programs=len(ops),
           n_signatures=len(rows), peak_bw_gbs=BW,
           peaks={f"{dt}/{f}": roof_tflops(dt, f) for dt in ("bf16", "bfp8", "bfp4") for f in ("LoFi", "HiFi2", "HiFi4")},
           classes={k: dict(s=v / 1e9, share=v / total) for k, v in cls.most_common()},
           bound={k: dict(s=v / 1e9, share=v / total) for k, v in bound.most_common()},
           bound_by_class={k: {b: round(v / total, 4) for b, v in c.items()} for k, c in bound_cls.items()},
           gaps=gap)
json.dump(out, open(f"{OUTP}_summary.json", "w"), indent=1, default=str)

# ---------- ops.json: top signatures to >= 85 % ----------
spec = []; acc = 0.0
for g in rows:
    o = g["o"]; r = o["r"]; rl = g["rl"]
    spec.append(dict(rank=len(spec) + 1, op=r["op"], cls=o["cls"], call_site=r.get("site"),
                     regions=dict(g["regs"].most_common(3)), args=r.get("args"), kwargs=r.get("kw"), out=r.get("out"),
                     math_fidelity=fidelity(r), fp32_dest_acc=fp32acc(r),
                     calls_profiled=g["n"], calls_full_fold=round(g["nw"], 1),
                     device_us_per_call=g["k"] / g["n"] / 1e3, device_s_full_fold=g["kw"] / 1e9,
                     share=g["kw"] / total, programs_per_call=o["nprog"], cores=o["cores"],
                     trisc1_over_kernel=(g["t1"] / g["k"]) if g["k"] else None,
                     flops_per_call=o["fl"], bytes_per_call=o["by"], achieved_tflops=rl["tflops"],
                     achieved_gbs=rl["gbs"], roof=dict(peak_tflops=rl["peak_tflops"], t_math_us=rl["t_math_ns"] / 1e3,
                     t_mem_us=rl["t_mem_ns"] / 1e3, fraction_of_roof=rl["roof_eff"], limiter=rl["limiter"]),
                     bound=rl["bound"]))
    acc += g["kw"] / total
    if acc >= 0.85 and len(spec) >= 10: break
json.dump(dict(version=1, note="Protenix-v2 730 tok (580+150), MSA 9947 deep, 5 samples, full fold = 10 cycles + "
               "200 steps scaled by call count; one Wormhole chip of .107; device kernel time from the tt-metal profiler",
               peaks=out["peaks"], peak_bw_gbs=BW, aiclk=out["aiclk"], device_s_full_fold=total / 1e9,
               coverage=acc, ops=spec), open(f"{OUTP}_ops.json", "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ("device_ns_profiled", "device_ns_full", "n_records", "n_signatures", "aiclk")}, default=str))
print("coverage", round(acc, 4), "ops", len(spec))
for k, v in out["classes"].items(): print(f"{v['share']*100:6.2f} % {v['s']:8.2f} s  {k}")
for k, v in out["bound"].items(): print(f"bound {k}: {v['share']*100:.1f} %")
for s in spec[:40]:
    print(f"{s['rank']:3d} {s['share']*100:5.2f}% {s['device_us_per_call']:9.1f}us x{s['calls_full_fold']:8.1f} "
          f"{s['op'][:40]:40s} {s['cls'][:28]:28s} {s['bound']:8s} roof {s['roof']['fraction_of_roof']:.2f} "
          f"{s['achieved_tflops']:.1f}TF {s['achieved_gbs']:.0f}GB/s {(s['call_site'] or [''])[0]}")
