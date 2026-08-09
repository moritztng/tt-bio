#!/usr/bin/env python3
"""W6: per-op roofline split of the NON-trimul half of a Protenix-v2 Pairformer block.

Scope: triangle attention (start + end), transition_z, the residual adds, the s-track.
W2 owns the two trimuls. Everything here is measured on the card the process opens; the
roofs are re-measured in the same process so no number is inherited from another host.

Modes
  roofs   copy/read/write and dense-matmul roofs on THIS card
  mods    uninstrumented per-sub-module wall (sync on both sides, warm, median)
  ops     per-ttnn-op wall with a sync around every op, plus bytes/FLOPs and the
          roof each op is actually bound by. Sync-per-op perturbs; `mods` is the
          honest total and the ops table is attribution only.

    TT_VISIBLE_DEVICES=1 python3 perf/attn_block/op_split.py --mode ops --n 320
"""

import argparse
import json
import time

import torch

import ttnn
from tt_bio import protenix_weights as PW
import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import PairformerLayer, get_device

CKPT = "/home/ttuser/.boltz/protenix-v2.pt"
TRI_HEAD_DIM = 32

DEV = None
REC = []
TRACK = True

_EB = {}


def elem_bytes(dt):
    if not _EB:
        _EB.update({
            ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 1,
            ttnn.uint32: 4, ttnn.int32: 4, ttnn.uint16: 2, ttnn.uint8: 1,
        })
    return _EB.get(dt, 2)


def tbytes(t):
    """Physical bytes of a ttnn tensor, tile padding included."""
    try:
        sh = [int(d) for d in t.shape]
    except Exception:
        return 0
    eb = elem_bytes(t.dtype)
    v = 1
    if len(sh) >= 2 and t.layout == ttnn.TILE_LAYOUT:
        for d in sh[:-2]:
            v *= d
        v *= ((sh[-2] + 31) // 32) * 32 * ((sh[-1] + 31) // 32) * 32
    else:
        for d in sh:
            v *= d
    return v * eb


def shp(t):
    try:
        loc = "L1" if t.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"
    except Exception:
        loc = "?"
    return f"{'x'.join(str(int(d)) for d in t.shape)}@{loc}"


def _tensors(xs):
    out = []
    for x in xs:
        if isinstance(x, ttnn.Tensor):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            out.extend(t for t in x if isinstance(t, ttnn.Tensor))
    return out


def _mm_flops(ins):
    if len(ins) < 2:
        return 0
    a, b = ins[0], ins[1]
    try:
        as_, bs = [int(d) for d in a.shape], [int(d) for d in b.shape]
        m = 1
        for d in as_[:-1]:
            m *= d
        return 2 * m * as_[-1] * bs[-1]
    except Exception:
        return 0


def _sdpa_flops(ins):
    try:
        q, k = ins[0], ins[1]
        qs, ks = [int(d) for d in q.shape], [int(d) for d in k.shape]
        b, h, sq, d = qs
        sk = ks[2]
        return 2 * (2 * b * h * sq * sk * d)
    except Exception:
        return 0


MM_OPS = {"linear", "matmul", "minimal_matmul"}


def wrap(mod, name, label=None):
    fn = getattr(mod, name)
    lab = label or name

    def w(*a, **kw):
        if not TRACK:
            return fn(*a, **kw)
        ins = _tensors(list(a) + list(kw.values()))
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        ttnn.synchronize_device(DEV)
        dt = (time.perf_counter() - t0) * 1e3
        outs = _tensors(r if isinstance(r, (tuple, list)) else [r])
        flops = _mm_flops(ins) if lab in MM_OPS else (_sdpa_flops(ins) if "dot_product" in lab else 0)
        REC.append({
            "op": lab, "ms": dt,
            "in": [shp(t) for t in ins], "out": [shp(t) for t in outs],
            "rd": sum(tbytes(t) for t in ins), "wr": sum(tbytes(t) for t in outs),
            "flops": flops,
        })
        return r

    setattr(mod, name, w)


def install():
    for n in ("layer_norm", "linear", "matmul", "add", "add_", "multiply", "multiply_",
              "permute", "transpose", "concat", "chunk", "typecast", "clone", "to_layout",
              "sigmoid", "softmax", "reshape", "unsqueeze", "squeeze", "slice"):
        if hasattr(ttnn, n):
            wrap(ttnn, n)
    for n in ("nlp_create_qkv_heads", "nlp_concat_heads", "minimal_matmul"):
        if hasattr(ttnn.experimental, n):
            wrap(ttnn.experimental, n)
    wrap(ttnn.transformer, "scaled_dot_product_attention", "sdpa")


def build_layer(ckc):
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    blk = {k[len("pairformer_stack.blocks.0."):]: v
           for k, v in sd.items() if k.startswith("pairformer_stack.blocks.0.")}
    remapped = PW.remap_pairformer_block(blk)
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    return PairformerLayer(TRI_HEAD_DIM, c_z // TRI_HEAD_DIM, 384 // 16, 16, True, remapped, ckc), c_z


def timeit(fn, warm, iters):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2], ts


def roofs(args):
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {}
    # DRAM->DRAM copy roof: one op that reads N bytes and writes N bytes and computes nothing.
    for mb in (52, 157):
        n = mb * 2 ** 20 // 2
        rows = n // 4096
        x = ttnn.from_torch(torch.randn(rows, 4096), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        med, _ = timeit(lambda: ttnn.clone(x), 3, 7)
        b = tbytes(x)
        out[f"clone_{mb}MB"] = {"ms": med, "bytes_rw": 2 * b, "GBps": 2 * b / med * 1e-6}
        ttnn.deallocate(x)
    # read roof: reduce a big tensor to a column (writes ~nothing)
    n = 157 * 2 ** 20 // 2
    x = ttnn.from_torch(torch.randn(n // 4096, 4096), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    med, _ = timeit(lambda: ttnn.sum(x, dim=-1), 3, 7)
    out["read_sum"] = {"ms": med, "bytes": tbytes(x), "GBps": tbytes(x) / med * 1e-6}
    ttnn.deallocate(x)
    # dense matmul compute roof, HiFi4 bf16, fp32 dest acc (the block's config)
    for k in (2048, 4096):
        a = ttnn.from_torch(torch.randn(k, k), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        b_ = ttnn.from_torch(torch.randn(k, k), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        med, _ = timeit(lambda: ttnn.matmul(a, b_, compute_kernel_config=ckc), 3, 7)
        out[f"matmul_{k}"] = {"ms": med, "tflops": 2 * k ** 3 / med * 1e-9}
        ttnn.deallocate(a)
        ttnn.deallocate(b_)
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


def make_inputs(N, c_z):
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.1, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    return s, z


def mods(args):
    global TRACK
    TRACK = False
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer(ckc)
    N = args.n
    s, z = make_inputs(N, c_z)
    print(f"layer built: c_z={c_z} N={N}", flush=True)

    res = {}

    def sub(name, fn):
        med, ts = timeit(fn, args.warm, args.iters)
        res[name] = {"median_ms": med, "series": [round(t, 2) for t in ts]}
        print(f"{name:22s} {med:8.2f} ms   {[round(t,1) for t in ts]}", flush=True)

    holder = {}

    def run_tm_start():
        holder["u"] = layer.triangle_multiplication_start(z, None)
    def run_tm_end():
        holder["u"] = layer.triangle_multiplication_end(z, None)
    def run_ta_start():
        holder["u"] = layer.triangle_attention_start(z, None)
    def run_ta_end():
        holder["u"] = layer.triangle_attention_end(z, None)
    def run_tz():
        holder["u"] = layer.transition_z(z)

    def timed_free(name, fn):
        def g():
            fn()
            ttnn.deallocate(holder["u"])
        sub(name, g)

    timed_free("trimul_start[W2]", run_tm_start)
    timed_free("trimul_end[W2]", run_tm_end)
    timed_free("tri_att_start", run_ta_start)
    timed_free("tri_att_end", run_ta_end)
    timed_free("transition_z", run_tz)

    zc = ttnn.clone(z)
    sub("residual_add_z", lambda: ttnn.add_(zc, z))
    ttnn.deallocate(zc)

    def run_strack():
        s_norm = ttnn.layer_norm(s, weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias,
                                 epsilon=1e-5, compute_kernel_config=ckc)
        u = layer.attention_pair_bias(s_norm, z, seq_mask=None)
        ttnn.deallocate(s_norm)
        ttnn.deallocate(u)
    sub("s_track_attn", run_strack)

    def run_ts():
        u = layer.transition_s(s)
        ttnn.deallocate(u)
    sub("transition_s", run_ts)

    def run_block():
        holder["s"], holder["z"] = layer(s, z)
    med, ts = timeit(lambda: layer(s, z), args.warm, args.iters)
    res["FULL_BLOCK"] = {"median_ms": med, "series": [round(t, 2) for t in ts]}
    print(f"{'FULL_BLOCK':22s} {med:8.2f} ms   {[round(t,1) for t in ts]}", flush=True)

    tot = sum(v["median_ms"] for k, v in res.items() if k != "FULL_BLOCK")
    print(f"sum(parts)={tot:.2f} ms  block={res['FULL_BLOCK']['median_ms']:.2f} ms "
          f"(parts exclude 4 of the 5 residual adds)")
    for k, v in res.items():
        v["pct_of_block"] = 100 * v["median_ms"] / res["FULL_BLOCK"]["median_ms"]
    if args.out:
        json.dump({"n": N, "c_z": c_z, "mods": res}, open(args.out, "w"), indent=2)


def ops(args):
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    global TRACK
    TRACK = False
    layer, c_z = build_layer(ckc)
    N = args.n
    s, z = make_inputs(N, c_z)
    # warm every path first (JIT + program cache), untracked
    for _ in range(args.warm):
        for fn in (layer.triangle_attention_start, layer.triangle_attention_end):
            ttnn.deallocate(fn(z, None))
        ttnn.deallocate(layer.transition_z(z))
    ttnn.synchronize_device(DEV)

    # what does the merged qkv-L1 config decide at this size?
    mt = (N * N) // 32
    print(f"qkv_l1_config(m_tiles={mt}, k_tiles={c_z//32}, n_tiles={3*c_z//32}) -> "
          f"{T._tri_att_qkv_l1_config(mt, c_z // 32, 3 * c_z // 32, 2)}", flush=True)

    install()
    TRACK = True
    sections = []
    for name, fn in (("tri_att_start", lambda: layer.triangle_attention_start(z, None)),
                     ("tri_att_end", lambda: layer.triangle_attention_end(z, None)),
                     ("transition_z", lambda: layer.transition_z(z))):
        REC.clear()
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(DEV)
        wall = (time.perf_counter() - t0) * 1e3
        ttnn.deallocate(out)
        sections.append({"section": name, "wall_ms": wall, "ops": list(REC)})
        print(f"\n=== {name}  instrumented wall {wall:.2f} ms, {len(REC)} ops ===", flush=True)
        agg = {}
        for r in REC:
            k = (r["op"], tuple(r["in"]), tuple(r["out"]))
            a = agg.setdefault(k, {"n": 0, "ms": 0.0, "rd": r["rd"], "wr": r["wr"], "flops": r["flops"]})
            a["n"] += 1
            a["ms"] += r["ms"]
        for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["ms"]):
            byt = (a["rd"] + a["wr"]) * a["n"]
            gbps = byt / a["ms"] * 1e-6 if a["ms"] else 0
            tf = a["flops"] * a["n"] / a["ms"] * 1e-9 if a["ms"] else 0
            ai = a["flops"] / (a["rd"] + a["wr"]) if (a["rd"] + a["wr"]) else 0
            print(f"  {a['ms']:7.3f} ms  x{a['n']:<3d} {k[0]:<24s} "
                  f"{byt/2**20:8.1f} MB  {gbps:6.1f} GB/s  {tf:7.2f} TF/s  AI={ai:6.1f}  "
                  f"in={','.join(k[1])} out={','.join(k[2])}", flush=True)
    TRACK = False
    if args.out:
        json.dump({"n": N, "c_z": c_z, "sections": sections}, open(args.out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["roofs", "mods", "ops"], required=True)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    global DEV
    DEV = get_device()
    {"roofs": roofs, "mods": mods, "ops": ops}[args.mode](args)


if __name__ == "__main__":
    main()
