"""lpx-matmul: every Protenix-v2 matmul call on ONE Wormhole chip, then a format/fidelity/config sweep.

Serving-path setup copied from perf/pfm_ttfast/bench_wh.py (same lease, worker env, MSA cache, AICLK
sampler). Two phases in one process, so the chip is leased once:

1. CAPTURE: one exact fold (730 tokens, 10 recycles, 5 samples, today's defaults) with ttnn.linear,
   ttnn.matmul, ttnn.experimental.minimal_matmul and ttnn.generic_op wrapped. Per call: the tensor specs
   (shape, dtype, layout, memory config), the non-tensor kwargs (program, compute, output config), the
   tt_bio call site, and the synced wall time. Calls recorded inside a trace capture are counted once per
   execute_trace of that trace, so the diffusion steps count as they run.
2. SWEEP: each unique matmul signature, heaviest first, is rebuilt on random tensors of the same spec and
   timed as called (baseline) and under variants: activation/weight/output format, math fidelity, fp32
   dest acc, and program configs re-tuned per format. Device time is a captured trace replayed back to
   back (no host dispatch in the number), REPS reps, each long enough to swamp the sync. Each variant's
   output is read back once and checked finite and shaped right (accuracy is out of scope). Typecast
   costs for the chain (bf16 -> bfp8/bfp4 at the activation's spec, bfp8 -> bf16 at the output's spec)
   are timed the same way.

usage: bench_wh.py OUT CHIP SHARE YAML [SWEEP_BUDGET_S] [TOP_FULL]
"""
import gc, glob, json, math, os, sys, threading, time, traceback
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1]); CHIP = int(sys.argv[2]); SHARE = int(sys.argv[3]); YAML = Path(sys.argv[4])
BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 3600.0
TOP_FULL = int(sys.argv[6]) if len(sys.argv) > 6 else 16
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time()
    LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush(); print(json.dumps(kw, default=str), flush=True)

from tt_bio import runtime
if SHARE:
    os.environ.update(runtime.host_thread_cap_env(SHARE, None))
log(ev="start", chip=CHIP, share=SHARE, yaml=str(YAML), budget=BUDGET, top_full=TOP_FULL)

NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
samples = []
def _sampler():
    while True:
        row = {}
        for n in NODES:
            try: row[n] = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            except Exception: row[n] = -1
        samples.append((time.monotonic(), row)); time.sleep(0.5)
threading.Thread(target=_sampler, daemon=True).start()

from tt_bio import main as M
captured = {}
class _Stop(Exception): pass
def _grab(*a, **kw):
    captured["payload"] = a[1] if isinstance(a[0], str) else a[0]; raise _Stop
M._dispatch_run = _grab; M._dispatch_to_controller = _grab
argv = ["predict", str(YAML), "--model", "protenix-v2", "--diffusion_samples", "5",
        "--recycling_steps", "10", "--accelerator", "tenstorrent", "--output_format", "cif",
        "--msa_db_path", os.path.expanduser("~/japanfold/msa/db"), "--msa_dir", str(OUT / "msa"),
        "--out_dir", str(OUT / "cli")]
try:
    M.cli.main(argv, standalone_mode=False)
except _Stop:
    pass
cfg0 = dict(captured["payload"]["config"])
log(ev="cfg", cfg={k: v for k, v in cfg0.items() if "pass" not in k and "key" not in k})

from tt_bio import worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [CHIP])[0]
winfo = worker_payload(slot)
W._apply_tt_environment(winfo); W._bind_host_threads()
W._ensure_local_artifacts(cfg0)
import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.protenix as P
log(ev="env", worker=winfo, TT_VISIBLE_DEVICES=os.environ.get("TT_VISIBLE_DEVICES"))

# ---------------------------------------------------------------- capture hooks
DEV = {}
ON = {"v": False}            # hooks active (the fold only)
CAP = {"tid": None}          # inside begin/end_trace_capture: no syncs, record against the trace
SIGS = {}                    # key -> dict(op, spec, sites, n, times)
TRACE_SIGS = defaultdict(list)
HERE = Path(T.__file__).parent

def _tspec(t):
    return ("T", tuple(int(x) for x in t.shape), str(t.dtype), str(t.layout),
            t.memory_config() if t.storage_type() == ttnn.StorageType.DEVICE else None)

def _is_t(v):
    return isinstance(v, ttnn.Tensor)

def _key(op, a, kw):
    parts = [op]
    for i, v in enumerate(a):
        parts.append(f"a{i}=" + (repr(_tspec(v)) if _is_t(v) else (f"[{len(v)} tensors]" if isinstance(v, (list, tuple)) and v and _is_t(v[0]) else repr(v))))
    for k in sorted(kw):
        v = kw[k]
        parts.append(f"{k}=" + (repr(_tspec(v)) if _is_t(v) else repr(v)))
    return "|".join(parts)

def _site():
    f = sys._getframe(2); out = []
    while f is not None and len(out) < 3:
        p = Path(f.f_code.co_filename)
        if p.parent == HERE and p.name not in ("ops.py",):
            out.append(f"{p.name}:{f.f_lineno}:{f.f_code.co_name}")
        f = f.f_back
    return " < ".join(out)

def _hook(op, fn, replayable):
    def w(*a, **kw):
        if not ON["v"]:
            return fn(*a, **kw)
        if op == "generic_op":
            # key by site and io shapes; the program descriptor is not replayable here
            io = a[0] if a else kw.get("io_tensors", [])
            key = "generic_op|" + _site() + "|" + repr([_tspec(t)[1:3] for t in io if _is_t(t)])
        else:
            key = _key(op, a, kw)
        s = SIGS.get(key)
        if s is None:
            s = SIGS[key] = dict(op=op, replay=replayable, sites=defaultdict(int), n=0, times=[], spec=None)
            if replayable:
                s["spec"] = ([_tspec(v) if _is_t(v) else v for v in a],
                             {k: (_tspec(v) if _is_t(v) else v) for k, v in kw.items()})
        s["n"] += 1; s["sites"][_site()] += 1
        if CAP["tid"] is not None:
            TRACE_SIGS["pending"].append(key)
            return fn(*a, **kw)
        ttnn.synchronize_device(DEV["d"]); t0 = time.perf_counter()
        r = fn(*a, **kw)
        ttnn.synchronize_device(DEV["d"]); s["times"].append(time.perf_counter() - t0)
        return r
    return w

ORIG = dict(linear=ttnn.linear, matmul=ttnn.matmul, minimal_matmul=ttnn.experimental.minimal_matmul,
            generic_op=ttnn.generic_op)
ttnn.linear = _hook("linear", ORIG["linear"], True)
ttnn.matmul = _hook("matmul", ORIG["matmul"], True)
ttnn.experimental.minimal_matmul = _hook("minimal_matmul", ORIG["minimal_matmul"], True)
ttnn.generic_op = _hook("generic_op", ORIG["generic_op"], False)

_btc, _etc, _ext = ttnn.begin_trace_capture, ttnn.end_trace_capture, ttnn.execute_trace
EXEC = defaultdict(int)
def begin_tc(*a, **kw):
    tid = _btc(*a, **kw); CAP["tid"] = tid; TRACE_SIGS["pending"] = []; return tid
def end_tc(dev, tid, *a, **kw):
    r = _etc(dev, tid, *a, **kw); TRACE_SIGS[repr(tid)] = TRACE_SIGS.pop("pending", []); CAP["tid"] = None
    return r
def exec_t(dev, tid, *a, **kw):
    if ON["v"]: EXEC[repr(tid)] += 1
    return _ext(dev, tid, *a, **kw)
ttnn.begin_trace_capture, ttnn.end_trace_capture, ttnn.execute_trace = begin_tc, end_tc, exec_t

state = W._WorkerState("tenstorrent")
t = time.monotonic(); dev = DEV["d"] = T.get_device(trace="protenix"); log(ev="device_open", s=time.monotonic() - t,
                                                                           arch=T.arch_name())
opened = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1])
                 for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and
                 os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
NODE = opened[0]
log(ev="nodes_open", nodes=opened)
grid = dev.compute_with_storage_grid_size(); log(ev="grid", x=grid.x, y=grid.y)

def clk_window(t0, t1):
    c = sorted(r[NODE] for ts, r in samples if t0 <= ts <= t1)
    return dict(aiclk_n=len(c), aiclk_median=c[len(c) // 2] if c else None, aiclk_min=c[0] if c else None)

# ---------------------------------------------------------------- phase 1: one exact fold
if True:
    state.model = P.Protenix.load_from_checkpoint(cfg0["protenix_ckpt"])
    state.bind_run("lpx", dict(cfg0))
    state.model_id = cfg0["model"]; state.config_hash = W.run_config_hash(cfg0)
    m = state.model
    log(ev="build", trunk_fidelity=str(m.trunk.compute_kernel_config.math_fidelity), fast=m._fast,
        diffusion_dtype=str(m.diffusion.dtype))
    # cold fold untimed (compile), then the hooked fold
    for kind, seed in (("cold", 101), ("capture", 102)):
        rcfg = dict(cfg0, seed=seed); sdir = OUT / f"struct_{kind}"; sdir.mkdir(exist_ok=True)
        rcfg["struct_dir"] = str(sdir)
        ON["v"] = kind == "capture"; t0 = time.monotonic(); err = None
        try:
            state.predict_one(YAML, rcfg)
        except Exception:
            err = traceback.format_exc()[-3000:]
        t1 = time.monotonic(); ON["v"] = False
        log(ev="fold", kind=kind, wall_s=t1 - t0, err=err, **clk_window(t0, t1))
    for tk, keys in TRACE_SIGS.items():
        for k in keys:
            SIGS[k]["n_trace"] = SIGS[k].get("n_trace", 0) + EXEC.get(tk, 0)
    rows = []
    for k, s in SIGS.items():
        ts = sorted(s["times"]); med = ts[len(ts) // 2] if ts else None
        n_tot = len(s["times"]) + s.get("n_trace", 0)
        rows.append(dict(key=k, op=s["op"], n_eager=len(s["times"]), n_trace=s.get("n_trace", 0), n_total=n_tot,
                         med_s=med, est_total_s=(med or 0) * n_tot, sites=dict(s["sites"])))
    rows.sort(key=lambda r: -r["est_total_s"])
    (OUT / "calls.json").write_text(json.dumps(rows, indent=1, default=str))
    log(ev="captured", n_sigs=len(rows), n_calls=sum(r["n_total"] for r in rows),
        est_total_s=sum(r["est_total_s"] for r in rows), traces={k: v for k, v in EXEC.items()})
    state.model = None; gc.collect()

# ---------------------------------------------------------------- phase 2: sweep
ttnn.linear, ttnn.matmul = ORIG["linear"], ORIG["matmul"]
ttnn.experimental.minimal_matmul, ttnn.generic_op = ORIG["minimal_matmul"], ORIG["generic_op"]
ttnn.begin_trace_capture, ttnn.end_trace_capture, ttnn.execute_trace = _btc, _etc, _ext
OPS = dict(linear=ttnn.linear, matmul=ttnn.matmul, minimal_matmul=ttnn.experimental.minimal_matmul)
DT = dict(bf16=ttnn.bfloat16, b8=ttnn.bfloat8_b, b4=ttnn.bfloat4_b, f32=ttnn.float32)
FID = dict(HiFi4=ttnn.MathFidelity.HiFi4, HiFi3=ttnn.MathFidelity.HiFi3, HiFi2=ttnn.MathFidelity.HiFi2,
           LoFi=ttnn.MathFidelity.LoFi)
DTN = {str(v): k for k, v in DT.items()}

def mk(spec, dtype=None):
    _, shape, dt, layout, mc = spec
    dt = dtype or next(v for k, v in DT.items() if str(v) == dt)
    x = torch.randn(shape) * 0.25
    lay = ttnn.TILE_LAYOUT if "TILE" in layout else ttnn.ROW_MAJOR_LAYOUT
    if lay == ttnn.ROW_MAJOR_LAYOUT and dt in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        lay = ttnn.TILE_LAYOUT
    try:
        return ttnn.from_torch(x, dtype=dt, layout=lay, device=dev, memory_config=mc)
    except Exception:
        t = ttnn.from_torch(x, dtype=dt, layout=lay, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_memory_config(t, mc)

def time_fn(fn, reps=10, target_s=0.004):
    """Device time of fn() per call: trace replays back to back, else eager loop. Returns (median, spread, mode)."""
    out = fn(); ttnn.synchronize_device(dev)
    out2 = fn(); ttnn.synchronize_device(dev)
    for o in (out2,):
        try: ttnn.deallocate(o)
        except Exception: pass
    mode = "trace"
    try:
        tid = _btc(dev, cq_id=0); o = fn(); _etc(dev, tid, cq_id=0)
        run = lambda: _ext(dev, tid, cq_id=0, blocking=False)
    except Exception as e:
        mode = f"eager({type(e).__name__})"; tid = None; o = None
        run = fn
    ttnn.synchronize_device(dev); t0 = time.perf_counter(); run(); ttnn.synchronize_device(dev)
    est = max(time.perf_counter() - t0, 1e-6)
    k = max(1, min(200, math.ceil(target_s / est)))
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(k):
            r = run()
            if tid is None and r is not None:
                ttnn.deallocate(r)
        ttnn.synchronize_device(dev); per.append((time.perf_counter() - t0) / k)
    if tid is not None:
        ttnn.release_trace(dev, tid)
        try: ttnn.deallocate(o)
        except Exception: pass
    per.sort()
    return per[len(per) // 2], (per[-1] - per[0]) / per[len(per) // 2], mode, k, out

def check(out, shape):
    t = ttnn.to_torch(out).float()
    ok = bool(torch.isfinite(t).all()) and tuple(t.shape)[-2:] == tuple(shape)[-2:]
    return ok, tuple(t.shape)

def ckc_of(kw):
    c = kw.get("compute_kernel_config")
    if c is None:
        return dict(math_fidelity="None", fp32_dest_acc_en=None, packer_l1_acc=None, math_approx_mode=None)
    return dict(math_fidelity=str(c.math_fidelity).split(".")[-1], fp32_dest_acc_en=c.fp32_dest_acc_en,
                packer_l1_acc=c.packer_l1_acc, math_approx_mode=c.math_approx_mode)

def mk_ckc(base, fid=None, acc=None):
    b = ckc_of(base)
    f = fid or (b["math_fidelity"] if b["math_fidelity"] != "None" else "HiFi4")
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=FID[f], math_approx_mode=bool(b["math_approx_mode"]) if b["math_approx_mode"] is not None else True,
        fp32_dest_acc_en=bool(b["fp32_dest_acc_en"]) if acc is None else acc,
        packer_l1_acc=bool(b["packer_l1_acc"]) if b["packer_l1_acc"] is not None else True)

def pc_variants(pc, acc):
    """Re-tuned program configs: in0_block_w and out_subblock over the same per-core split."""
    if pc is None or not hasattr(pc, "to_json"):
        return []
    j = json.loads(pc.to_json()); out = []
    M_, N_ = j.get("per_core_M"), j.get("per_core_N"); kw0 = j.get("in0_block_w")
    if not (M_ and N_ and kw0):
        return []
    cap = 4 if acc else 8
    subs = sorted({(h, w) for h in range(1, 9) for w in range(1, 9)
                   if h * w <= cap and M_ % h == 0 and N_ % w == 0}, key=lambda s: -s[0] * s[1])[:3]
    for ibw in sorted({kw0, kw0 * 2, kw0 * 4, max(1, kw0 // 2)}):
        for (h, w) in subs:
            if (ibw, h, w) == (kw0, j.get("out_subblock_h"), j.get("out_subblock_w")):
                continue
            jj = dict(j, in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w)
            if "out_block_h" in jj: jj["out_block_h"] = M_
            if "out_block_w" in jj: jj["out_block_w"] = N_
            try:
                out.append((f"ibw{ibw}_sb{h}x{w}", type(pc).from_json(json.dumps(jj))))
            except Exception:
                pass
    return out

def variants(op, a, kw, full):
    """(name, in0 dtype, in1 dtype, out dtype, fidelity, acc, program config override)."""
    base_acc = ckc_of(kw)["fp32_dest_acc_en"]
    V = [("base", None, None, None, None, None, "as")]
    combos = [("b8w", None, "b8", None, None, None), ("b8b8", "b8", "b8", None, None, None),
              ("b8b8_lofi_noacc_ob8", "b8", "b8", "b8", "LoFi", False),
              ("b4b4_lofi_noacc_ob8", "b4", "b4", "b8", "LoFi", False)]
    if full:
        combos = [("hifi3", None, None, None, "HiFi3", None), ("hifi2", None, None, None, "HiFi2", None),
                  ("lofi", None, None, None, "LoFi", None), ("noacc", None, None, None, None, False),
                  ("acc", None, None, None, None, True), ("ob8", None, None, "b8", None, None),
                  ("b8w", None, "b8", None, None, None), ("b4w", None, "b4", None, None, None),
                  ("b8b8", "b8", "b8", None, None, None), ("b8b4", "b8", "b4", None, None, None),
                  ("b4b4", "b4", "b4", None, None, None),
                  ("bf16_lofi_noacc", None, None, None, "LoFi", False),
                  ("b8b8_hifi2_noacc", "b8", "b8", None, "HiFi2", False),
                  ("b8b8_lofi_noacc_ob8", "b8", "b8", "b8", "LoFi", False),
                  ("b8b4_lofi_noacc_ob8", "b8", "b4", "b8", "LoFi", False),
                  ("b4b4_lofi_noacc_ob8", "b4", "b4", "b8", "LoFi", False)]
    V += [(n, i0, i1, o, f, ac, "as") for n, i0, i1, o, f, ac in combos]
    pc = kw.get("program_config") if op != "minimal_matmul" else None
    V.append(("auto_pc", None, None, None, None, None, None))
    V.append(("b8b8_lofi_noacc_ob8_auto_pc", "b8", "b8", "b8", "LoFi", False, None))
    if full and pc is not None:
        for nm, p in pc_variants(pc, base_acc):
            V.append(("pc_" + nm, None, None, None, None, None, p))
        for nm, p in pc_variants(pc, False):
            V.append(("b8b8_lofi_noacc_ob8_pc_" + nm, "b8", "b8", "b8", "LoFi", False, p))
    return V

def run_variant(op, a, kw, v):
    name, i0, i1, o, fid, acc, pco = v
    if op == "minimal_matmul" and (i0 or i1) and (i0 or "x") != (i1 or "x"):
        return dict(var=name, skip="minimal_matmul needs in0 dtype == in1 dtype")
    args = list(a); kws = dict(kw)
    tens = []
    for idx, role in ((0, i0), (1, i1)):
        if idx < len(args) and isinstance(args[idx], tuple) and args[idx][:1] == ("T",):
            args[idx] = mk(args[idx], DT[role] if role else None); tens.append(args[idx])
    for k2 in list(kws):
        if isinstance(kws[k2], tuple) and kws[k2][:1] == ("T",):
            kws[k2] = mk(kws[k2], DT[i1] if (k2 in ("weight_tensor", "input_tensor_b") and i1) else None)
            tens.append(kws[k2])
    for idx in range(2, len(args)):
        if isinstance(args[idx], tuple) and args[idx][:1] == ("T",):
            args[idx] = mk(args[idx]); tens.append(args[idx])
    if fid or acc is not None:
        kws["compute_kernel_config"] = mk_ckc(kw, fid, acc)
    if o:
        kws["dtype"] = DT[o]
    if pco != "as":
        key = "config" if op == "minimal_matmul" else "program_config"
        if pco is None: kws.pop(key, None)
        else: kws[key] = pco
    fn = lambda: OPS[op](*args, **kws)
    t_med, spread, mode, k, out = time_fn(fn)
    ok, shp = check(out, out.shape)
    for t_ in tens + [out]:
        try: ttnn.deallocate(t_)
        except Exception: pass
    return dict(var=name, us=t_med * 1e6, spread=spread, mode=mode, k=k, finite=ok, out_shape=shp,
                in0=i0 or "as", in1=i1 or "as", out=o or "as", fid=fid or "as",
                acc="as" if acc is None else acc, pc="as" if pco == "as" else ("auto" if pco is None else name))

def chain(spec_in, spec_out):
    """Typecast costs the format introduces: activation bf16 -> b8/b4, output b8 -> bf16."""
    res = {}
    for nm, sp, src, dst in (("in_to_b8", spec_in, None, "b8"), ("in_to_b4", spec_in, None, "b4"),
                             ("out_b8_to_bf16", spec_out, "b8", "bf16")):
        if sp is None: continue
        try:
            x = mk(sp, DT[src] if src else ttnn.bfloat16)
            t_med, spread, mode, k, out = time_fn(lambda: ttnn.typecast(x, DT[dst]))
            res[nm] = dict(us=t_med * 1e6, spread=spread, mode=mode)
            ttnn.deallocate(out); ttnn.deallocate(x)
        except Exception as e:
            res[nm] = dict(err=f"{type(e).__name__}: {str(e)[:200]}")
    return res

t_sweep0 = time.monotonic(); n_done = 0
for rank, r in enumerate([r for r in rows if r["op"] in OPS]):
    if time.monotonic() - t_sweep0 > BUDGET:
        log(ev="budget_stop", done=n_done); break
    s = SIGS.get(r["key"])
    if s is None or s.get("spec") is None:
        continue
    a, kw = s["spec"]
    full = rank < TOP_FULL
    t0 = time.monotonic(); res = []
    for v in variants(r["op"], a, kw, full):
        try:
            res.append(run_variant(r["op"], a, kw, v))
        except Exception as e:
            res.append(dict(var=v[0], err=f"{type(e).__name__}: {str(e)[:300]}"))
    base = next((x for x in res if x.get("var") == "base" and "us" in x), None)
    for x in res:
        if base and "us" in x:
            x["speedup"] = base["us"] / x["us"]
    spec_in = a[0] if a and isinstance(a[0], tuple) else None
    spec_out = None
    if spec_in and len(a) > 1 and isinstance(a[1], tuple):
        spec_out = ("T", tuple(spec_in[1][:-1]) + (a[1][1][-1],), "DataType.BFLOAT8_B", spec_in[3],
                    kw.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG)
    ch = chain(spec_in, spec_out) if full else {}
    log(ev="sweep", rank=rank, op=r["op"], n_total=r["n_total"], est_total_s=r["est_total_s"],
        sites=r["sites"], key=r["key"], ckc=ckc_of(kw), results=res, chain=ch,
        s=time.monotonic() - t0, **clk_window(t0, time.monotonic()))
    n_done += 1
log(ev="end", sweep_s=time.monotonic() - t_sweep0, done=n_done)
os._exit(0)
