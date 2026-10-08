"""lpx-matmul r2: the arms r1 could not run, rebuilt from r1's call list (no fold needed).

Every top call in r1 was interleaved (L1 or DRAM), so calls.json's key rebuilds it exactly. Adds:
  * fp32 calls: plain bf16 arms (the r1 grid only went fp32 -> bfp8/bfp4);
  * program configs built from scratch (2D mcast, 1D height, 1D width) for calls on ttnn's auto config, and
    re-tuned explicit ones, each at three format points (as called; fp32 acc off; bfp8 LoFi no-acc bfp8 out);
  * the generic_op matmul kernels (trimul in_proj / fused tail, triangle-attention qkvgb / out_proj) replayed as
    minimal_matmul at the same shapes, per format and fidelity;
  * the producer side of the chain: layer_norm emitting bfp8 instead of bf16, at the PWA / transition inputs.
Device time is trace replay, as in bench_wh.py (devtime.py).

usage (on .107, chip carved for lpx-matmul):  sweep2.py R1_DIR OUT N_TOP
"""
import json, math, os, re, sys, time, traceback
from pathlib import Path

R1 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); NTOP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time(); LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush()
    print(json.dumps(kw, default=str)[:300], flush=True)

import torch, ttnn
import tt_bio.tenstorrent as T
from devtime import Bench

dev = T.get_device(trace="protenix")
B = Bench(dev)
NODE = int(os.environ.get("LPX_NODE", "-1"))
def aiclk():
    try: return int(Path(f"/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read_text().split()[0])
    except Exception: return None
log(ev="start", arch=T.arch_name(), grid=str(dev.compute_with_storage_grid_size()), aiclk=aiclk())
DT, FID = B.DT, B.FID
GX, GY = 8, 9

TSPEC = re.compile(r"\('T', \(([\d, ]*)\), 'DataType\.(\w+)', 'Layout\.(\w+)', MemoryConfig\(.*?buffer_type=BufferType::(\w+).*?\)\)")
def parse(key):
    parts = key.split("|"); op = parts[0]; a, kw = [], {}
    for p in parts[1:]:
        name, _, val = p.partition("=")
        m = TSPEC.fullmatch(val)
        if m:
            v = ("T", tuple(int(x) for x in m.group(1).split(",") if x.strip()), m.group(2), m.group(4))
        elif name in ("memory_config",):
            v = ttnn.L1_MEMORY_CONFIG if "BufferType::L1" in val else ttnn.DRAM_MEMORY_CONFIG
        elif name == "dtype":
            v = getattr(ttnn, {"BFLOAT16": "bfloat16", "BFLOAT8_B": "bfloat8_b", "BFLOAT4_B": "bfloat4_b",
                               "FLOAT32": "float32"}[val.split(".")[-1]])
        elif name == "core_grid":
            x, y = re.findall(r"\d+", val); v = ttnn.CoreGrid(x=int(x), y=int(y))
        elif name == "program_config":
            v = parse_pc(val)
        elif name == "compute_kernel_config":
            continue                                      # taken from r1's sweep event (logged values)
        elif val in ("True", "False", "None"):
            v = {"True": True, "False": False, "None": None}[val]
        else:
            v = val.strip("'")
        if name.startswith("a") and name[1:].isdigit():
            a.append(v)
        else:
            kw[name] = v
    return op, a, kw

def parse_pc(val):
    if val == "None":
        return None
    cls, body = val.split("(", 1)
    d = {}
    for k, v in re.findall(r"(\w+)=([^,()]+|\(x=\d+,y=\d+\)|\{\})", body):
        if k == "compute_with_storage_grid_size":
            x, y = re.findall(r"\d+", v); d[k] = (int(x), int(y))
        elif k in ("fused_activation", "hop_cores", "num_global_cb_receivers", "out_block_h", "out_block_w"):
            continue
        else:
            d[k] = bool(int(v)) if k in ("transpose_mcast", "fuse_batch", "mcast_in0", "gather_in0", "untilize_out") else int(v)
    d.pop("gather_in0", None); d.pop("untilize_out", None)
    return getattr(ttnn, cls)(**d)

def tiles(n): return math.ceil(n / 32)

def best_sub(pm, pn, cap):
    c = [(h, w) for h in range(1, 9) for w in range(1, 9) if h * w <= cap and pm % h == 0 and pn % w == 0]
    return max(c, key=lambda s: (s[0] * s[1], s[1]))

def gen_pcs(a, kw, acc, act):
    """Program configs from scratch for a 2D-weight matmul: 2D mcast, 1D height (mcast_in1), 1D width (mcast_in0)."""
    s0, s1 = a[0][1], a[1][1]
    if len(s1) > 2 and math.prod(s1[:-2]) > 1:
        return []
    tb = kw.get("transpose_b") is True
    Mt = tiles(math.prod(s0[:-1])); Kt = tiles(s0[-1]); Nt = tiles(s1[-2] if tb else s1[-1])
    cap = 4 if acc else 8
    fa = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if act == "silu" else None
    ibws = [b for b in (1, 2, 4, 8, 16) if Kt % b == 0] or [1]
    out = []
    def rounded(n, cores):
        """per-core tile counts to try: the even split, and it rounded up to x2 / x4 while it still fits."""
        p = math.ceil(n / cores)
        return sorted({p} | {math.ceil(p / m) * m for m in (2, 4) if math.ceil(n / (math.ceil(p / m) * m)) <= cores})
    shapes = []
    if Nt >= GX:
        shapes += [("2d", pm, pn) for pm in rounded(Mt, GY) for pn in rounded(Nt, GX)]
    shapes += [("1d_h", pm, Nt) for pm in rounded(Mt, GX * GY)]
    if Nt >= GX * GY // 2:
        shapes += [("1d_w", Mt, pn) for pn in rounded(Nt, GX * GY)]
    for lay, pm, pn in shapes:
        h, w = best_sub(pm, pn, cap)
        for ibw in ibws:
            try:
                if lay == "2d":
                    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(GX, GY), in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w,
                        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=fa)
                else:
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(GX, GY), in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w,
                        per_core_M=pm, per_core_N=pn, fuse_batch=True, fused_activation=fa, mcast_in0=lay == "1d_w")
                out.append((f"{lay}_ibw{ibw}_pm{pm}_pn{pn}_sb{h}x{w}", pc))
            except Exception:
                pass
    return out

def retune(pc, a, kw, acc):
    """Re-tune an explicit config: every legal in0_block_w and the largest subblocks at the same per-core split."""
    j = json.loads(pc.to_json()); pm, pn = j["per_core_M"], j["per_core_N"]
    s0 = a[0][1]; Kt = tiles(s0[-2] if kw.get("transpose_a") else s0[-1])
    out = []
    subs = sorted({(h, w) for h in range(1, 9) for w in range(1, 9) if h * w <= (4 if acc else 8)
                   and pm % h == 0 and pn % w == 0}, key=lambda s: -s[0] * s[1])[:2]
    for ibw in sorted({b for b in (1, 2, 4, 8, 16, Kt) if Kt % b == 0 and b <= 32}):
        for h, w in subs:
            jj = dict(j, in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w)
            try: out.append((f"rt_ibw{ibw}_sb{h}x{w}", type(pc).from_json(json.dumps(jj))))
            except Exception: pass
    return out

OPS = dict(linear=ttnn.linear, matmul=ttnn.matmul, minimal_matmul=ttnn.experimental.minimal_matmul)

def run(op, a, kw, ckc, name, i0=None, i1=None, o=None, fid=None, acc=None, pc="as"):
    args, kws, tens = list(a), dict(kw), []
    for idx, role in ((0, i0), (1, i1)):
        args[idx] = B.mk(args[idx], DT[role] if role else None); tens.append(args[idx])
    for k2, v in list(kws.items()):
        if isinstance(v, tuple) and v[:1] == ("T",):
            kws[k2] = B.mk(v); tens.append(kws[k2])
    kws["compute_kernel_config"] = B.ckc(ckc, fid, acc)
    if o: kws["dtype"] = DT[o]
    if pc != "as":
        key = "config" if op == "minimal_matmul" else "program_config"
        if pc is None: kws.pop(key, None)
        else:
            kws[key] = pc
            if kws.get("activation") == "silu" and op == "linear": kws.pop("activation")
            kws.pop("core_grid", None)
    try:
        r = B.time(lambda: OPS[op](*args, **kws))
        res = dict(var=name, us=r["us"], spread=r["spread"], mode=r["mode"], finite=r["finite"])
    except Exception as e:
        res = dict(var=name, err=f"{type(e).__name__}: {str(e).splitlines()[0][:240]}")
    finally:
        for t in tens: B.free(t)
    return res

calls = json.loads((R1 / "calls.json").read_text())
ck = {}
for l in open(R1 / "bench.jsonl"):
    e = json.loads(l)
    if e.get("ev") == "sweep": ck[e["key"]] = e["ckc"]
TRUNK_CKC = dict(math_fidelity="HiFi4", fp32_dest_acc_en=True, packer_l1_acc=True, math_approx_mode=True)

# ---- 1. replayable top calls: fp32->bf16 arms and program-config search per format point
done = 0
for rank, c in enumerate(calls):
    if done >= NTOP: break
    if c["op"] not in OPS: continue
    op, a, kw = parse(c["key"]); ckc = ck.get(c["key"], TRUNK_CKC)
    acc_on = bool(ckc.get("fp32_dest_acc_en"))
    fp32 = a[0][2] == "FLOAT32"
    t0 = time.monotonic(); res = [run(op, a, kw, ckc, "base")]
    if fp32:
        for nm, f, ac, o in (("bf16", None, None, None), ("bf16_ob16", None, None, "bf16"),
                             ("bf16_noacc_ob16", None, False, "bf16"), ("bf16_hifi2_noacc_ob16", "HiFi2", False, "bf16"),
                             ("bf16_lofi_noacc_ob16", "LoFi", False, "bf16")):
            res.append(run(op, a, kw, ckc, nm, "bf16", "bf16", o, f, ac))
    points = [("as", {}), ("noacc", dict(acc=False)), ("b8lofi", dict(i0="b8", i1="b8", o="b8", fid="LoFi", acc=False))]
    if fp32:
        points[0] = ("bf16", dict(i0="bf16", i1="bf16", o="bf16"))
    pcs = kw.get("program_config")
    for pname, pt in points:
        acc = pt.get("acc", acc_on)
        cands = retune(pcs, a, kw, acc) if pcs is not None else gen_pcs(a, kw, acc, kw.get("activation"))
        if pcs is not None and pname != "as":
            res.append(run(op, a, kw, ckc, f"{pname}_pc_as", pc="as", **pt))
        for cn, pc in cands:
            res.append(run(op, a, kw, ckc, f"{pname}_{cn}", pc=pc, **pt))
    base = next((x for x in res if x["var"] == "base" and "us" in x), None)
    for x in res:
        if base and "us" in x: x["speedup"] = base["us"] / x["us"]
    log(ev="sweep2", rank=rank, op=op, key=c["key"], n_total=c["n_total"], ckc=ckc, results=res,
        s=time.monotonic() - t0, aiclk=aiclk())
    done += 1

# ---- 2. generic_op matmul kernels as minimal_matmul at their shapes
GEN = [("trimul_in_proj", (1, 736, 736, 256), (256, 512), 2160),
       ("trimul_tail_proj", (1, 736, 736, 256), (256, 256), 2 * 1080),
       ("triatt_qkvgb", (1, 736, 736, 256), (256, 1056), 1080),
       ("triatt_out_proj", (1, 736, 736, 256), (256, 256), 1080)]
for name, s0, s1, n in GEN:
    a = [("T", s0, "BFLOAT16", "DRAM"), ("T", s1, "BFLOAT16", "DRAM")]
    kw = dict(memory_config=ttnn.DRAM_MEMORY_CONFIG)
    res = []
    for op in ("minimal_matmul", "linear"):
        for nm, pt in (("bf16_hifi4_acc", {}), ("bf16_hifi4_noacc", dict(acc=False)), ("bf16_lofi_noacc", dict(fid="LoFi", acc=False)),
                       ("b8_lofi_noacc_ob8", dict(i0="b8", i1="b8", o="b8", fid="LoFi", acc=False)),
                       ("b8_hifi2_noacc", dict(i0="b8", i1="b8", fid="HiFi2", acc=False)),
                       ("b4_lofi_noacc_ob8", dict(i0="b4", i1="b4", o="b8", fid="LoFi", acc=False))):
            res.append(run(op, a, kw, TRUNK_CKC, f"{op}_{nm}", **pt))
    log(ev="generic_equiv", name=name, shape=[s0, s1], n_calls=n, results=res, aiclk=aiclk())

# ---- 3. producer side: layer_norm emitting bf16 vs bfp8 (PWA input, transition input)
for name, shape, mc in (("pwa_ln", (512, 736, 128), ttnn.DRAM_MEMORY_CONFIG),
                        ("transition_ln", (1, 5, 736, 256), ttnn.L1_MEMORY_CONFIG)):
    x = B.mk(("T", shape, "BFLOAT16", "L1" if mc == ttnn.L1_MEMORY_CONFIG else "DRAM"))
    g = B.mk(("T", (shape[-1],), "BFLOAT16", "DRAM"), layout=ttnn.ROW_MAJOR_LAYOUT)
    res = {}
    for nm, extra in (("ln_bf16", {}), ("ln_b8", dict(dtype=ttnn.bfloat8_b))):
        try:
            r = B.time(lambda: ttnn.layer_norm(x, weight=g, bias=g, memory_config=mc, **extra))
            res[nm] = dict(us=r["us"], spread=r["spread"])
        except Exception as e:
            res[nm] = dict(err=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    try:
        r = B.time(lambda: ttnn.typecast(x, ttnn.bfloat8_b)); res["typecast_b8"] = dict(us=r["us"])
    except Exception as e:
        res["typecast_b8"] = dict(err=str(e)[:200])
    B.free(x); B.free(g)
    log(ev="producer", name=name, shape=shape, results=res, aiclk=aiclk())
log(ev="end")
os._exit(0)
