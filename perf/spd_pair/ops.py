"""spd-pair op bench: the pair-stream ops between Protenix-v2's triangle kernels, at a fold's shape.

One process, one chip. Times each arm as the back-to-back slope (n calls enqueued, one sync, wall / n,
warm program cache), arms round-robin rep by rep, and scores every numerics-changing arm against a
float64 torch reference of the same math on the same bf16 inputs. AICLK sampled by a side process.

Groups (--groups):
  ln    ttnn.layer_norm on [1, N, N, c_z] bf16 under several fidelity / dest-accumulation configs
  tail  triangle attention's tail: multiply_(o, g, SIGMOID on b), the head-major out projection, add_
  add   the bare residual add_ of two pair tensors
  trans the pair transition's assembly at row height h (--trans-h): today ttnn.chunk + concat of the
        block updates + add_, against a slice per block + pair_add.add_rows into z; torch.equal checked
  tadd  the ending triangle attention's way back: pair_transpose(u) + add_ into z, against
        pair_transpose_add.transpose_add(u, z); torch.equal checked
  pln   tt_bio.pair_ln (own kernel, bf16 or bfp8 out, --pln-arms own_<fidelity>_<b16|b8>) against
        ttnn.layer_norm at normal and fast mode's configs, all scored against float64
  lnmm  layer norm + the projection that consumes it (--lnmm-n output widths): ttnn.layer_norm (bf16) or
        pair_ln HiFi3 bf16 / bfp8 into one ttnn.linear (bfp8 weights, HiFi2, fixed across arms); also the
        linear alone on a bf16 vs bfp8 input. Scored against float64 LN(x) @ W

usage: ops.py OUT [--n 736] [--cz 256] [--groups ln,tail,add] [--reps 5] [--calls 8] [--chip C]
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=736)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--groups", default="ln,tail,add")
ap.add_argument("--trans-h", default="5,32")
ap.add_argument("--ln-arms", default="fast_today,normal_today,hifi4_f32,hifi2_f32,lofi_f32,hifi3_b16,hifi2_b16,lofi_b16")
ap.add_argument("--pln-arms", default="normal_today,fast_today,own_HiFi4_b16,own_HiFi3_b16,own_HiFi2_b16,own_HiFi3_b8")
ap.add_argument("--tail-arms", default="base,fused")
ap.add_argument("--lnmm-n", default="256,1024")
ap.add_argument("--tail-variants", default="",
                help="extra fused arms, name=SIGPOLY/RNE/fidelity/f32 separated by commas, e.g. poly=1/3/HiFi3/1")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=8)
ap.add_argument("--chip", type=int, default=None)
A = ap.parse_args()

OUT = Path(A.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time()
    s = json.dumps(kw, default=str)
    LOG.write(s + "\n"); LOG.flush(); print(s[:600], flush=True)


if A.chip is not None:
    from tt_bio import runtime, worker as W
    from tt_bio.host_controller import worker_payload
    slot = runtime.build_local_workers("tenstorrent", [object()], [A.chip])[0]
    W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.triatt_qkv as TQ

dev = T.get_device()
opened = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}")
                 and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")
                 and os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1].isdigit()})
NODE = opened[0] if opened else -1
CLKF = OUT / "aiclk.tsv"; CLKF.touch()
_SAMPLER = subprocess.Popen([sys.executable, "-c", f"""
import os, time
f = open({str(CLKF)!r}, "a"); parent = os.getppid()
while os.getppid() == parent:
    try: v = open("/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read().split()[0]
    except Exception: v = "-1"
    f.write(f"{{time.monotonic()}}\\t{{v}}\\n"); f.flush(); time.sleep(0.25)
"""], stderr=open(OUT / "sampler.err", "w"))
log(ev="nodes_open", nodes=opened, grid=tuple(T.COMPUTE_GRID_MAIN), arch=str(dev.arch()), args=vars(A))


def aiclk(t0, t1, pad=0.3):
    v = [int(m) for ts, m in (l.split("\t") for l in CLKF.read_text().splitlines())
         if t0 - pad <= float(ts) <= t1 + pad and int(m) > 0]
    return (min(v), max(v), len(v)) if v else None


def slope(fn, calls):
    """ms per call, back to back. `fn` returns a tensor to free (or None)."""
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter(); m0 = time.monotonic()
    outs = [fn() for _ in range(calls)]
    ttnn.synchronize_device(dev)
    ms = 1e3 * (time.perf_counter() - t0) / calls
    for o in outs:
        if o is not None and o.is_allocated():
            ttnn.deallocate(o)
    return ms, aiclk(m0, time.monotonic())


def rel(a, ref, rows=None):
    a = a.double(); ref = ref.double()
    d = a - ref
    return dict(rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                max_abs=float(d.abs().max()), nonfinite=int((~torch.isfinite(a)).sum()))


def ckc(fid, f32):
    c = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                         fp32_dest_acc_en=f32, packer_l1_acc=False)
    return c


def run_arms(group, arms, calls):
    """arms: name -> (fn, score). Interleaved reps; one record per arm with median, spread, clock."""
    times = {a: [] for a in arms}
    clocks = {a: [] for a in arms}
    for name, (fn, _) in arms.items():      # warm (compile + cache)
        o = fn(); ttnn.synchronize_device(dev)
        if o is not None and o.is_allocated():
            ttnn.deallocate(o)
    for r in range(A.reps):
        for name, (fn, _) in arms.items():
            ms, clk = slope(fn, calls)
            times[name].append(ms); clocks[name].append(clk)
    for name, (fn, score) in arms.items():
        t = times[name]
        cl = [c for c in clocks[name] if c]
        log(ev="arm", group=group, arm=name, ms_med=round(statistics.median(t), 4), ms_min=round(min(t), 4),
            ms_max=round(max(t), 4), reps=t, aiclk_min=min(c[0] for c in cl) if cl else None,
            aiclk_max=max(c[1] for c in cl) if cl else None, **(score() if score else {}))


torch.manual_seed(0)
N, C = A.n, A.cz
groups = A.groups.split(",")

# A pair-like activation: per-channel offsets and scales, some heavy channels.
mu = torch.randn(C) * 2.0
sd = torch.exp(torch.randn(C) * 0.7)
z_host = (torch.randn(1, N, N, C) * sd + mu).bfloat16()

if "ln" in groups:
    gamma = (1 + 0.3 * torch.randn(C)).bfloat16(); beta = (0.2 * torch.randn(C)).bfloat16()
    x = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gw = ttnn.from_torch(gamma.reshape(1, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    bw = ttnn.from_torch(beta.reshape(1, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    xd = z_host.double()
    m = xd.mean(-1, keepdim=True); v = xd.var(-1, unbiased=False, keepdim=True)
    ref = (xd - m) / torch.sqrt(v + 1e-5) * gamma.double() + beta.double()
    # the bf16 floor: the reference itself rounded to bf16
    log(ev="ln_floor", **rel(ref.bfloat16(), ref))
    cfgs = {"fast_today": ("HiFi4", False), "normal_today": ("HiFi3", True), "hifi4_f32": ("HiFi4", True),
            "hifi2_f32": ("HiFi2", True), "lofi_f32": ("LoFi", True), "hifi3_b16": ("HiFi3", False),
            "hifi2_b16": ("HiFi2", False), "lofi_b16": ("LoFi", False)}
    arms = {}
    for a in A.ln_arms.split(","):
        fid, f32 = cfgs[a]
        k = ckc(fid, f32)

        def fn(k=k):
            return ttnn.layer_norm(x, weight=gw, bias=bw, epsilon=1e-5, compute_kernel_config=k)

        def score(fn=fn):
            o = fn(); r = rel(ttnn.to_torch(o).float(), ref); ttnn.deallocate(o); return r
        arms[a] = (fn, score)
    run_arms("ln", arms, A.calls)
    for t in (x, gw, bw):
        ttnn.deallocate(t)

if "pln" in groups:
    # tt_bio.pair_ln (own kernel, bf16 or bfp8 out) against ttnn.layer_norm at today's two configs.
    import tt_bio.pair_ln as PLN
    gamma = (1 + 0.3 * torch.randn(C)).bfloat16(); beta = (0.2 * torch.randn(C)).bfloat16()
    x = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gw = ttnn.from_torch(gamma.reshape(1, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    bw = ttnn.from_torch(beta.reshape(1, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    xd = z_host.double()
    m = xd.mean(-1, keepdim=True); v = xd.var(-1, unbiased=False, keepdim=True)
    ref = (xd - m) / torch.sqrt(v + 1e-5) * gamma.double() + beta.double()
    log(ev="pln_floor", bf16=rel(ref.bfloat16(), ref))
    arms = {}
    for a in A.pln_arms.split(","):
        if a in ("normal_today", "fast_today"):
            k = ckc(*{"normal_today": ("HiFi3", True), "fast_today": ("HiFi4", False)}[a])

            def fn(k=k):
                return ttnn.layer_norm(x, weight=gw, bias=bw, epsilon=1e-5, compute_kernel_config=k)
        else:                                   # own_<fidelity>_<b16|b8>[_sq0|_sq1]
            _, fid, fmt, *sq = a.split("_")
            dt = ttnn.bfloat8_b if fmt == "b8" else ttnn.bfloat16
            sq = int(sq[0][2:]) if sq else PLN.PAIR_LN_SQ

            def fn(fid=fid, dt=dt, sq=sq):
                prev, PLN.PAIR_LN_SQ = PLN.PAIR_LN_SQ, sq
                try:
                    return PLN.layer_norm(x, gw, bw, 1e-5, dt, getattr(ttnn.MathFidelity, fid))
                finally:
                    PLN.PAIR_LN_SQ = prev

        def score(fn=fn):
            o = fn(); r = rel(ttnn.to_torch(o).float(), ref); ttnn.deallocate(o); return r
        arms[a] = (fn, score)
    run_arms("pln", arms, A.calls)
    for t in (x, gw, bw):
        ttnn.deallocate(t)

if "lnmm" in groups:
    import tt_bio.pair_ln as PLN
    gamma = (1 + 0.3 * torch.randn(C)).bfloat16(); beta = (0.2 * torch.randn(C)).bfloat16()
    x = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gw = ttnn.from_torch(gamma.reshape(1, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    bw = ttnn.from_torch(beta.reshape(1, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    xd = z_host.double()
    m = xd.mean(-1, keepdim=True); v = xd.var(-1, unbiased=False, keepdim=True)
    lnref = (xd - m) / torch.sqrt(v + 1e-5) * gamma.double() + beta.double()
    lk, mk = ckc("HiFi3", True), ckc("HiFi2", False)
    xn16 = ttnn.layer_norm(x, weight=gw, bias=bw, epsilon=1e-5, compute_kernel_config=lk)
    xn8 = PLN.layer_norm(x, gw, bw, 1e-5, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi3)
    for n_out in [int(v) for v in A.lnmm_n.split(",")]:
        W = (torch.randn(C, n_out) / C ** 0.5).bfloat16()
        w = ttnn.from_torch(W, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b)
        Wq = ttnn.to_torch(w).double()                 # the weights the device multiplies by
        ref = lnref @ Wq
        lin = lambda t: ttnn.linear(t, w, dtype=ttnn.bfloat16, compute_kernel_config=mk,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def chain(norm):
            def fn():
                t = norm(); o = lin(t); ttnn.deallocate(t); return o
            return fn
        norms = {"ttnn_b16": lambda: ttnn.layer_norm(x, weight=gw, bias=bw, epsilon=1e-5, compute_kernel_config=lk),
                 "own_b16": lambda: PLN.layer_norm(x, gw, bw, 1e-5, ttnn.bfloat16, ttnn.MathFidelity.HiFi3),
                 "own_b8": lambda: PLN.layer_norm(x, gw, bw, 1e-5, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi3)}
        arms = {}
        for a, norm in norms.items():
            fn = chain(norm)

            def score(fn=fn):
                o = fn(); r = rel(ttnn.to_torch(o).float(), ref); ttnn.deallocate(o); return r
            arms[f"{a}_n{n_out}"] = (fn, score)
        arms[f"mm_in16_n{n_out}"] = (lambda: lin(xn16), None)
        arms[f"mm_in8_n{n_out}"] = (lambda: lin(xn8), None)
        run_arms("lnmm", arms, A.calls)
        ttnn.deallocate(w)
    for t in (x, gw, bw, xn16, xn8):
        ttnn.deallocate(t)

if "add" in groups:
    za = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    zb = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    run_arms("add", {"add_": (lambda: (ttnn.add_(za, zb), None)[1], None)}, A.calls)
    ttnn.deallocate(za); ttnn.deallocate(zb)

if "tail" in groups:
    H, D = C // 32, 32
    o_host = (torch.randn(N, H, N, D) * 0.5).bfloat16()
    g_host = (torch.randn(N, H, N, D) * 1.5).bfloat16()
    w_host = (torch.randn(C, C) / C ** 0.5).bfloat16()        # [in, out], as torch_to_tt lays it out
    k = ckc("HiFi3", True)
    od = ttnn.from_torch(o_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gd = ttnn.from_torch(g_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    wd = ttnn.from_torch(w_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    zd = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    # float64 reference: z + (o * sigmoid(g)) @ W, heads concatenated head-major along channels
    gated = (o_host.double() * torch.sigmoid(g_host.double())).permute(0, 2, 1, 3).reshape(1, N, N, C)
    ref = z_host.double() + gated @ w_host.double()

    def mul():
        return ttnn.multiply(od, gd, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])

    gate_t = mul()

    def proj():
        return TQ.out_proj(gate_t, wd, k, ttnn.bfloat16)

    upd = proj()

    zacc = ttnn.clone(zd)

    def addz():
        ttnn.add_(zacc, upd)       # its own accumulator: zd stays z for the scored arms
        return None

    def base_once():
        g = mul(); u = TQ.out_proj(g, wd, k, ttnn.bfloat16); ttnn.deallocate(g)
        zz = ttnn.clone(zd); u = ttnn.reshape(u, zz.shape); ttnn.add_(zz, u); ttnn.deallocate(u)
        return zz

    def score_base():
        o = base_once(); r = rel(ttnn.to_torch(o).float().reshape(1, N, N, C), ref); ttnn.deallocate(o); return r
    arms = {"mul": (mul, None), "out_proj": (proj, None), "add_": (addz, None), "clone": (lambda: ttnn.clone(zd), None),
            "base_seq": (base_once, score_base)}
    if "fused" in A.tail_arms.split(","):
        TQ.set_tail("1")
        ref_nores = (gated @ w_host.double()).reshape(N, N, C)

        def fused():
            zz = ttnn.clone(zd)
            r = TQ.gated_out_proj(od, gd, wd, k, resid=zz)
            assert r is zz, "gated_out_proj declined the residual call"
            return zz

        def fused_nores():
            r = TQ.gated_out_proj(od, gd, wd, k)
            assert r is not None, "gated_out_proj declined"
            return r

        def score_fused():
            o = fused(); r = rel(ttnn.to_torch(o).float().reshape(1, N, N, C), ref); ttnn.deallocate(o); return r

        def score_nores():
            o = fused_nores(); r = rel(ttnn.to_torch(o).float().reshape(N, N, C), ref_nores); ttnn.deallocate(o)
            return r

        def score_base_nores():
            g = mul(); u = TQ.out_proj(g, wd, k, ttnn.bfloat16); ttnn.deallocate(g)
            r = rel(ttnn.to_torch(u).float().reshape(N, N, C), ref_nores); ttnn.deallocate(u); return r
        arms["fused_seq"] = (fused, score_fused)
        for spec in filter(None, A.tail_variants.split(",")):
            name, cfg = spec.split("=")
            sp, rn, fid, f32 = cfg.split("/")
            kv = ckc(fid, f32 == "1")

            def var(sp=int(sp), rn=int(rn), kv=kv):
                TQ.TAIL_SIGPOLY, TQ.TAIL_RNE = sp, rn
                try:
                    zz = ttnn.clone(zd)
                    assert TQ.gated_out_proj(od, gd, wd, kv, resid=zz) is zz
                    return zz
                finally:
                    TQ.TAIL_SIGPOLY, TQ.TAIL_RNE = 1, 0

            def score_var(var=var):
                o = var(); r = rel(ttnn.to_torch(o).float().reshape(1, N, N, C), ref); ttnn.deallocate(o); return r
            arms[f"fused_{name}"] = (var, score_var)
        arms["fused_nores"] = (fused_nores, score_nores)
        arms["base_nores"] = (lambda: (lambda g: (TQ.out_proj(g, wd, k, ttnn.bfloat16), ttnn.deallocate(g))[0])(mul()),
                              score_base_nores)
    run_arms("tail", arms, A.calls)

if "trans" in groups:
    from tt_bio import pair_add as PA
    up_host = (torch.randn(1, N, N, C) * 0.05).bfloat16()
    for h in [int(v) for v in A.trans_h.split(",")]:
        nb = -(-N // h)
        zt = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        ub = [ttnn.from_torch(up_host[:, s:min(s + h, N)].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev,
                              dtype=ttnn.bfloat16) for s in range(0, N, h)]

        def base(zt=zt, ub=ub, nb=nb):
            for c in ttnn.chunk(zt, nb, dim=1):
                ttnn.deallocate(c)
            u = ttnn.concat(ub, dim=1)
            ttnn.add_(zt, u); ttnn.deallocate(u)

        def rows(zt=zt, ub=ub, h=h):
            for i, s in enumerate(range(0, N, h)):
                c = zt[:, s:min(s + h, N)]
                ttnn.deallocate(c)
                PA.add_rows(zt, ub[i], s)

        # bytes: both start from the same z, one runs today's join + add_, the other the in-place rows
        za = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        zb = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        u = ttnn.concat(ub, dim=1); ttnn.add_(za, u); ttnn.deallocate(u)
        for i, s in enumerate(range(0, N, h)):
            PA.add_rows(zb, ub[i], s)
        ta, tb = ttnn.to_torch(za), ttnn.to_torch(zb)
        log(ev="trans_equal", h=h, equal=bool(torch.equal(ta, tb)), ndiff=int((ta != tb).sum()),
            rel=rel(tb.float(), (z_host.double() + up_host.double()).float())["rel_rms"])
        ttnn.deallocate(za); ttnn.deallocate(zb)
        run_arms(f"trans_h{h}", {"join_add": (lambda f=base: (f(), None)[1], None),
                                 "add_rows": (lambda f=rows: (f(), None)[1], None)}, max(2, A.calls // 4))
        for t in ub + [zt]:
            ttnn.deallocate(t)

if "tadd" in groups:
    from tt_bio import pair_transpose as PT, pair_transpose_add as PTA
    up_host = (torch.randn(N, N, C) * 0.05).bfloat16()
    ud = ttnn.from_torch(up_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    zt = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    za = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    zb = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    t = PT.pair_transpose(ud); ttnn.add_(za, ttnn.reshape(t, (1, N, N, C))); ttnn.deallocate(t)
    assert PTA.eligible(ud, zb)
    PTA.transpose_add(ud, zb)
    ta, tb = ttnn.to_torch(za), ttnn.to_torch(zb)
    ref = z_host.double() + up_host.double().permute(1, 0, 2).unsqueeze(0)
    log(ev="tadd_equal", equal=bool(torch.equal(ta, tb)), ndiff=int((ta != tb).sum()),
        rel=rel(tb.float(), ref.float())["rel_rms"])
    ttnn.deallocate(za); ttnn.deallocate(zb)

    def tbase():
        t = PT.pair_transpose(ud); ttnn.add_(zt, ttnn.reshape(t, (1, N, N, C))); ttnn.deallocate(t)

    run_arms("tadd", {"transpose": (lambda: PT.pair_transpose(ud), None),
                      "transpose_add_": (lambda: (tbase(), None)[1], None),
                      "fused": (lambda: (PTA.transpose_add(ud, zt), None)[1], None)}, A.calls)
    ttnn.deallocate(ud); ttnn.deallocate(zt)

log(ev="done")
_SAMPLER.terminate()
