"""lpx-matmul r3: the levers the format x fidelity x config sweep could not reach.

  pad    the token axis padded to 768 (K = 24 tiles instead of the prime 23), so in0_block_w 2/4/8/12 become
         legal on the trimul einsums and the other K = 730/736 matmuls. Every dim of the token axis pads, not only
         K, so the padded number carries the pad's full extra work (x1.136 on a cubic einsum).
  gen    the generic_op matmul kernels (trimul in_proj, triatt qkvgb / out_proj, trimul fused tail) through
         tt-bio's own entry points, bfp8 operands / LoFi / no fp32 acc. Their CB formats follow the tensors.
  chain  layer_norm -> consumer matmul as ONE traced call: bf16 throughout, layer_norm emitting bfp8, and
         layer_norm bf16 + typecast. gamma/beta are [1, 1, W/32, 32] ROW_MAJOR (r2 passed [W] and TT_FATALed).
         ttnn.layer_norm has no output dtype in this wheel (its output takes the input's format), so "emits
         bfp8" is measured as layer_norm reading a bfp8 copy of its input, i.e. a bfp8 residual upstream.
  opm    OPM z_rows [23552, 512] x [23552, 512]^T: the auto config is the only one whose per-core block fits.
         Tries in1 pre-transposed, and M-chunked explicit configs whose output block fits L1 only at bfp8.
Device time is trace replay (devtime.Bench), as in r1/r2.

  padcost  ttnn.pad 736 -> 768 of the einsum operands and the slice back, the price of the pad lever when the
         producer does not write the padded layout itself.
usage (on .107, chip carved for lpx-matmul):  r3.py R1_DIR OUT_DIR [section,...]
"""
import json, math, os, sys, time
from pathlib import Path

R1 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SECTIONS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["pad", "gen", "chain", "opm"]
LOG = open(OUT / "bench.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time(); LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush()
    print(json.dumps(kw, default=str)[:300], flush=True)

import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import mm_generic as G, mm_dualnoc, triatt_qkv, trimul_tail
from devtime import Bench
from replay import parse

dev = T.get_device(trace="protenix")
B = Bench(dev)
DT = B.DT
NODE = int(os.environ.get("LPX_NODE", "-1"))
def aiclk():
    try: return int(Path(f"/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read_text().split()[0])
    except Exception: return None
log(ev="start", sections=SECTIONS, grid=str(dev.compute_with_storage_grid_size()), aiclk=aiclk())

calls = json.loads((R1 / "calls.json").read_text())
CK = {e["key"]: e["ckc"] for e in map(json.loads, open(R1 / "bench.jsonl")) if e.get("ev") == "sweep"}
TRUNK = dict(math_fidelity="HiFi4", fp32_dest_acc_en=True, packer_l1_acc=True, math_approx_mode=True)
OPS = dict(linear=ttnn.linear, matmul=ttnn.matmul)

def tiles(n): return math.ceil(n / 32)

def timed(name, fn, tens=()):
    try:
        r = B.time(fn)
        res = dict(var=name, us=r["us"], spread=r["spread"], mode=r["mode"], finite=r["finite"])
    except Exception as e:
        res = dict(var=name, err=f"{type(e).__name__}: {str(e).splitlines()[0][:240] if str(e) else ''}")
    for t in tens: B.free(t)
    return res

def speedups(res):
    base = next((x for x in res if x["var"] == "base" and "us" in x), None)
    for x in res:
        if base and "us" in x: x["speedup"] = base["us"] / x["us"]
    return res

def call(op, a, kw, ckc, name, i0=None, i1=None, o=None, fid=None, acc=None, pc="as"):
    """One matmul call on random tensors of the captured spec, with format / fidelity / acc / config overrides."""
    args, kws, tens = list(a), dict(kw), []
    for idx, role in ((0, i0), (1, i1)):
        args[idx] = B.mk(args[idx], DT[role] if role else None); tens.append(args[idx])
    for k2, v in list(kws.items()):
        if isinstance(v, tuple) and v[:1] == ("T",):
            kws[k2] = B.mk(v, DT[i1] if i1 else None); tens.append(kws[k2])
    kws["compute_kernel_config"] = B.ckc(ckc, fid, acc)
    if o: kws["dtype"] = DT[o]
    if pc != "as":
        kws["program_config"] = pc; kws.pop("core_grid", None)
        if kws.get("activation") == "silu" and op == "linear": kws.pop("activation")
    return timed(name, lambda: OPS[op](*args, **kws), tens)

def subblocks(pm, pn, acc):
    s = sorted({(h, w) for h in range(1, 9) for w in range(1, 9)
                if h * w <= (4 if acc else 8) and pm % h == 0 and pn % w == 0}, key=lambda s: (-s[0] * s[1], -s[1]))
    return s[:2]

def configs(pc, a, kw, acc):
    """Every legal in0_block_w (<= 12) x the two largest subblocks, at the call's own per-core split; for a call on
    the auto config, 2D mcast and 1D height splits of the padded shape."""
    s0, s1 = a[0][1], a[1][1]
    Kt = tiles(s0[-2] if kw.get("transpose_a") else s0[-1])
    ibws = [b for b in (1, 2, 3, 4, 6, 8, 12) if Kt % b == 0]
    out = []
    if pc is not None:
        j = json.loads(pc.to_json())
        for ibw in ibws:
            for h, w in subblocks(j["per_core_M"], j["per_core_N"], acc):
                try: out.append((f"ibw{ibw}_sb{h}x{w}", type(pc).from_json(json.dumps(dict(j, in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w)))))
                except Exception: pass
        return out
    if len(s1) > 2 and math.prod(s1[:-2]) > 1:
        return out
    tb = kw.get("transpose_b") is True
    Mt = tiles(math.prod(s0[:-1])); Nt = tiles(s1[-2] if tb else s1[-1])
    for lay, pm, pn in (("2d", tiles(Mt * 32 / 9), tiles(Nt * 32 / 8)), ("1d_h", tiles(Mt * 32 / 72), Nt)):
        if lay == "2d" and Nt < 8: continue
        for ibw in ibws:
            for h, w in subblocks(pm, pn, acc):
                try:
                    if lay == "2d":
                        p = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(8, 9), in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w,
                            per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None)
                    else:
                        p = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(8, 9), in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w,
                            per_core_M=pm, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=False)
                    out.append((f"{lay}_ibw{ibw}_pm{pm}_pn{pn}_sb{h}x{w}", p))
                except Exception:
                    pass
    return out

def padded(a):
    return [("T", tuple(768 if d in (730, 736) else d for d in x[1]), x[2], x[3]) if isinstance(x, tuple) else x for x in a]

# ---- pad: token axis to 768 so K = 24 tiles
if "pad" in SECTIONS:
    for rank in (10, 12, 13, 17):
        c = calls[rank]; op, a, kw = parse(c["key"]); ckc = CK.get(c["key"], TRUNK)
        acc_on = bool(ckc.get("fp32_dest_acc_en")); fp32 = a[0][2] == "FLOAT32"
        pa = padded(a); pc = kw.get("program_config")
        t0 = time.monotonic(); res = [call(op, a, kw, ckc, "base")]
        points = [("as", {}), ("noacc", dict(acc=False)), ("b8lofi", dict(i0="b8", i1="b8", o="b8", fid="LoFi", acc=False))]
        if fp32:
            points[0] = ("bf16", dict(i0="bf16", i1="bf16", o="bf16"))
        for pname, pt in points:
            acc = pt.get("acc", acc_on)
            res.append(call(op, a, kw, ckc, f"{pname}_736", **pt))
            res.append(call(op, pa, kw, ckc, f"{pname}_768_pc_as", **pt))
            for cn, p in configs(pc, pa, kw, acc):
                res.append(call(op, pa, kw, ckc, f"{pname}_768_{cn}", pc=p, **pt))
        log(ev="pad", rank=rank, op=op, key=c["key"], n_total=c["n_total"], ckc=ckc, results=speedups(res),
            work_ratio=math.prod(pa[0][1]) * pa[1][1][-1] / (math.prod(a[0][1]) * a[1][1][-1]) if not kw.get("transpose_b")
            else math.prod(pa[0][1]) * pa[1][1][-2] / (math.prod(a[0][1]) * a[1][1][-2]),
            s=time.monotonic() - t0, aiclk=aiclk())

# ---- gen: the generic_op matmul kernels through tt-bio's own entry points
ARMS = (("base", "bf16", "bf16", dict()),                               # as the fold calls them
        ("noacc", "bf16", "bf16", dict(acc=False)),
        ("lofi_noacc", "bf16", "bf16", dict(fid="LoFi", acc=False)),
        ("b8_hifi4_acc_obf16", "b8", "bf16", dict()),                   # formats alone
        ("b8_hifi2_noacc_ob8", "b8", "b8", dict(fid="HiFi2", acc=False)),
        ("b8_lofi_noacc_ob8", "b8", "b8", dict(fid="LoFi", acc=False)),
        ("b8_lofi_noacc_obf16", "b8", "bf16", dict(fid="LoFi", acc=False)))

def gen_case(name, n_calls, build):
    res = []
    for arm, fin, fout, ck in ARMS:
        ckc = B.ckc(TRUNK, ck.get("fid"), ck.get("acc"))
        try:
            fn, tens = build(DT[fin], DT[fout], ckc)
        except Exception as e:
            res.append(dict(var=arm, err=f"{type(e).__name__}: {str(e)[:200]}")); continue
        res.append(timed(arm, fn, tens))
    log(ev="gen", name=name, n_calls=n_calls, results=speedups(res), aiclk=aiclk())

def T_(shape, dt): return B.mk(("T", shape, "BFLOAT16", "DRAM"), dt)

def b_in_proj(fi, fo, ckc):
    x, w = T_((1, 736, 736, 256), fi), T_((256, 512), fi)
    def fn():
        r = mm_dualnoc.in_proj(x, w, ckc, fo, ttnn.DRAM_MEMORY_CONFIG)
        if r is None: raise RuntimeError(f"in_proj declined: {dict(mm_dualnoc.REJECTS) if hasattr(mm_dualnoc, 'REJECTS') else ''}")
        return r
    return fn, (x, w)

def b_qkvgb(fi, fo, ckc):
    x, w = T_((736, 736, 256), fi), T_((256, 1056), fi)
    cfg = (T._mm_block_for(w), tuple(T.COMPUTE_GRID_MAIN))
    def fn():
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([736, 8, 736, 32]), fo, ttnn.TILE_LAYOUT, dev,
                                               ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
        outs.append(ttnn.allocate_tensor_on_device(ttnn.Shape([736, 736, 8]), fo, ttnn.TILE_LAYOUT, dev,
                                                   ttnn.DRAM_MEMORY_CONFIG))
        G.generic_minimal_matmul(dev, x, w, outs, cfg, G.ckc_args(ckc), {"HEAD_MAJOR_MT": 23},
                                 triatt_qkv.KERNEL_DIR, n_widths=[8] * 4 + [1])
        for o in outs[1:]: B.free(o)
        return outs[0]
    return fn, (x, w)

def b_out_proj(fi, fo, ckc):
    g, w = T_((736, 8, 736, 32), fi), T_((256, 256), fi)
    return (lambda: triatt_qkv.out_proj(g, w, ckc, fo)), (g, w)

def b_tail(fi, fo, ckc):
    if fo != ttnn.bfloat16: raise RuntimeError("fused_tail writes bf16 only")
    xa, xb = T_((1, 736, 736, 256), fi), T_((1, 736, 736, 256), fi)
    wa, wb = T_((256, 256), fi), T_((256, 256), fi)
    def fn():
        r = trimul_tail.fused_tail(xa, xb, wa, wb, G.ckc_args(ckc), tuple(T.COMPUTE_GRID_MAIN))
        if r is None: raise RuntimeError(f"fused_tail declined: {dict(trimul_tail.REJECTS)}")
        return r
    return fn, (xa, xb, wa, wb)

if "gen" in SECTIONS:
    gen_case("trimul_in_proj", 2160, b_in_proj)
    gen_case("triatt_qkvgb", 1080, b_qkvgb)
    gen_case("triatt_out_proj", 1080, b_out_proj)
    gen_case("trimul_fused_tail", 1080, b_tail)

# ---- chain: layer_norm -> consumer, one traced call
def ln_params(w):
    mk = lambda v: ttnn.from_torch(torch.full((1, 1, w // 32, 32), v), dtype=ttnn.bfloat16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    return mk(1.0), mk(0.0)

if "chain" in SECTIONS:
    for name, rank in (("pwa_ln->head_out", 1), ("transition_ln->swiglu_w1", 0)):
        c = calls[rank]; op, a, kw = parse(c["key"]); ckc = CK.get(c["key"], TRUNK)
        mc = ttnn.L1_MEMORY_CONFIG if a[0][3] == "L1" else ttnn.DRAM_MEMORY_CONFIG
        x = B.mk(a[0]); x8 = B.mk(a[0], DT["b8"]); gam, bet = ln_params(a[0][1][-1])
        w16, w8 = B.mk(a[1]), B.mk(a[1], DT["b8"])
        def consumer(h, w, fid=None, acc=None, o=None):
            kws = {k: v for k, v in kw.items() if not (isinstance(v, tuple) and v[:1] == ("T",))}
            kws["compute_kernel_config"] = B.ckc(ckc, fid, acc)
            if o: kws["dtype"] = DT[o]
            return OPS[op](h, w, **kws)
        def ln(dtype=None):
            return ttnn.layer_norm(x8 if dtype == ttnn.bfloat8_b else x, weight=gam, bias=bet, memory_config=mc)
        def chain(prod, w, **ck):
            def fn():
                h = prod(); r = consumer(h, w, **ck); B.free(h); return r
            return fn
        b8ck = dict(fid="LoFi", acc=False)
        res = [timed("ln_bf16", ln), timed("ln_b8", lambda: ln(ttnn.bfloat8_b)),
               timed("typecast_b8", lambda: ttnn.typecast(x, ttnn.bfloat8_b)),
               timed("base", chain(ln, w16)),                                             # today
               timed("bf16_lofi_noacc", chain(ln, w16, **b8ck)),
               timed("ln_b8_lofi_noacc", chain(lambda: ln(ttnn.bfloat8_b), w8, **b8ck)),
               timed("ln_b8_lofi_noacc_ob8", chain(lambda: ln(ttnn.bfloat8_b), w8, o="b8", **b8ck)),
               timed("ln_cast_b8_lofi_noacc", chain(lambda: ttnn.typecast(ln(), ttnn.bfloat8_b), w8, **b8ck))]
        for t in (x, x8, gam, bet, w16, w8): B.free(t)
        log(ev="chain", name=name, rank=rank, key=c["key"], n_total=c["n_total"], results=speedups(res), aiclk=aiclk())

# ---- padcost: what producing the 768-padded einsum operands costs when the producer cannot write them padded
if "padcost" in SECTIONS:
    res = []
    for shape in ((1, 128, 736, 736), (1, 64, 736, 736)):
        x = B.mk(("T", shape, "BFLOAT16", "DRAM")); y = B.mk(("T", shape[:2] + (768, 768), "BFLOAT16", "DRAM"))
        tag = "x".join(map(str, shape))
        res.append(timed(f"pad_{tag}_to_768", lambda: ttnn.pad(x, [(0, 0), (0, 0), (0, 32), (0, 32)], 0.0)))
        res.append(timed(f"slice_{tag}_from_768", lambda: ttnn.slice(y, [0, 0, 0, 0], list(shape))))
        B.free(x); B.free(y)
    log(ev="padcost", results=res, aiclk=aiclk())

# ---- opm: z_rows at 23552 x 512 x 23552, layouts the auto config does not try
if "opm" in SECTIONS:
    c = calls[5]; op, a, kw = parse(c["key"]); ckc = CK.get(c["key"], TRUNK)
    M, K = a[0][1]; N = a[1][1][0]
    res = [call(op, a, kw, ckc, "base")]
    pre = [a[0], ("T", (K, N), a[1][2], a[1][3])]; kpre = {k: v for k, v in kw.items() if k != "transpose_b"}
    PTS = (("as", {}), ("b8lofi", dict(i0="b8", i1="b8", o="b8", fid="LoFi", acc=False)))
    for pname, pt in PTS:
        if pname != "as": res.append(call(op, a, kw, ckc, f"{pname}_tb", **pt))
        res.append(call(op, pre, kpre, ckc, f"{pname}_pre_t", **pt))
    Mt, Kt, Nt = tiles(M), tiles(K), tiles(N)
    for pname, pt in PTS:
        acc = pt.get("acc", True)
        fi = DT.get(pt.get("i0")) or ttnn.bfloat16
        w = B.mk(pre[1], fi)
        for chunk_t, pm, pn, gy in ((32, 4, 92, 8), (16, 2, 92, 8), (64, 8, 92, 8)):
            if Mt % chunk_t: continue
            xs = [B.mk(("T", (chunk_t * 32, K), "BFLOAT16", "DRAM"), fi) for _ in range(Mt // chunk_t)]
            for ibw in (1, 2, 4, 8):
                for h, sw in subblocks(pm, pn, acc)[:1]:
                    nm = f"{pname}_mchunk{chunk_t}_pm{pm}_pn{pn}_ibw{ibw}_sb{h}x{sw}"
                    try:
                        p = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(8, gy), in0_block_w=ibw, out_subblock_h=h, out_subblock_w=sw,
                            per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None)
                    except Exception as e:
                        res.append(dict(var=nm, err=str(e)[:200])); continue
                    ck_ = B.ckc(ckc, pt.get("fid"), pt.get("acc"))
                    od = DT[pt["o"]] if pt.get("o") else ttnn.bfloat16
                    def fn(p=p, ck_=ck_, od=od):
                        last = None
                        for xi in xs:
                            if last is not None: B.free(last)
                            last = ttnn.matmul(xi, w, program_config=p, compute_kernel_config=ck_, dtype=od)
                        return last
                    res.append(timed(nm, fn))
            for t in xs: B.free(t)
        B.free(w)
    # the cost of cutting M chunks out of one producer tensor, when the producer cannot emit them
    x = B.mk(a[0])
    res.append(timed("slice_one_32tile_chunk", lambda: ttnn.slice(x, [0, 0], [1024, K])))
    B.free(x)
    log(ev="opm", rank=5, key=c["key"], n_total=c["n_total"], results=speedups(res), aiclk=aiclk())

log(ev="end")
os._exit(0)
