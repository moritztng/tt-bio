"""lpx-census: op-level device profile of one Protenix-v2 fold on ONE Wormhole chip.

Needs a Tracy-enabled tt-metal (the pip wheel has the device profiler compiled out) and
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1.
Setup is pfm-ttfast's bench_wh.py (serving-path config, MSA cache), so the fold is the one users get.

Attribution without a per-op sync: every device program carries the runtime id ttnn hands out per device
operation (ttnn::CoreIDs device_operation_id), so each top-level ttnn call records the id range [id0, id1)
it consumed, and the profiler records read every FLUSH device ops are joined back by runtime id. The fold
therefore runs at its own dispatch pace, and device-idle gaps between consecutive programs inside a flush
batch are real gaps. Programs launched outside a top-level ttnn call (Tensor methods etc.) fall in the id
holes and are recorded as "<unhooked>" at the next call's site.

Each fold in FOLDS (name:cycles:steps) is profiled the same way; run a short one first so the measured fold
is warm (program cache built, lazy caches filled).

Outputs in OUT: census.jsonl (events), sigs_<f>.jsonl (one line per distinct op signature: op, call site,
region, args/kwargs/out with shapes, dtypes, layouts, memory configs, compute/program configs),
ops_<f>.jsonl ([sig, id0, id1, cycle, step] per call), progs_<f>.jsonl (per program: runtime id, flush
batch, cores, kernel/fw/risc durations, start/end).

usage: census.py OUT CHIP YAML FOLDS     e.g. census.py r2 24 complex730.yaml warm:2:4,full:10:200
"""
import json, os, sys, threading, time, glob
from pathlib import Path

os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "5000")
OUT = Path(sys.argv[1]); CHIP = int(sys.argv[2]); YAML = Path(sys.argv[3])
FOLDS = [(f.split(":")[0], int(f.split(":")[1]), int(f.split(":")[2])) for f in sys.argv[4].split(",")]
FLUSH = [1000]
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "census.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time(); LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush()
    print(json.dumps(kw, default=str)[:400], flush=True)

from tt_bio import runtime
os.environ.update(runtime.host_thread_cap_env(31, None))
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
import ttnn.decorators as D
import tt_bio.tenstorrent as T
import tt_bio.protenix as P
import tt_bio.ops as OPS
log(ev="env", ttnn=ttnn.__file__, TT_VISIBLE_DEVICES=os.environ.get("TT_VISIBLE_DEVICES"),
    prof={k: os.environ.get(k) for k in os.environ if "PROFILER" in k})

# ---------------- region stack (module path of every op) ----------------
REG = []; CTR = {}; CUR = {"cyc": -1, "step": -1}
def wrap_method(cls, meth, key):
    f = cls.__dict__.get(meth)
    if f is None: return
    def w(*a, **k):
        REG.append(key)
        try: return f(*a, **k)
        finally: REG.pop()
    setattr(cls, meth, w)
def wrap_func(mod, name, key):
    f = getattr(mod, name, None)
    if f is None: return
    def w(*a, **k):
        REG.append(key)
        try: return f(*a, **k)
        finally: REG.pop()
    setattr(mod, name, w)
for cls, meth, key in [
        (P.Protenix, "_trunk_cond", "trunk_cond"), (P.Protenix, "_atom_feat_inputs", "atom_feats"),
        (P.AtomAttentionEncoder, "__call__", "input_aae"), (P.TrunkInput, "__call__", "trunk_input"),
        (P.Trunk, "__call__", "trunk"), (P.Trunk, "_msa", "msa"), (P.Trunk, "_template", "template"),
        (P.Protenix, "_diffusion_pair_cond", "diff_pair_cond"), (P.Protenix, "_plm_z_term", "plm_z_term"),
        (P.DiffusionModule, "_atom_cond", "atom_cond"), (P.DiffusionModule, "_dit_block_biases", "dit_biases"),
        (P.AtomTransformer, "__call__", "atom_tx"), (P.AtomTransformer, "precompute_biases", "atom_tx_bias"),
        (P.ConfidenceHead, "confidence", "confidence"), (P.ConfidenceHead, "confidence_device", "confidence"),
        (P.ConfidenceHead, "z_base_device", "conf_zbase"),
        (T.Pairformer, "__call__", "pairformer"), (T.MSA, "__call__", "msa_stack"),
        (T.DiffusionTransformer, "__call__", "dit"),
        (T.PairformerLayer, "__call__", "pf_layer"), (T.MSALayer, "__call__", "msa_layer"),
        (T.TriangleMultiplication, "__call__", "trimul"), (T.TriangleAttention, "__call__", "triatt"),
        (T.AttentionPairBias, "__call__", "apb"), (T.PairWeightedAveraging, "__call__", "pwa"),
        (T.PairWeightedAveraging, "head_weights", "pwa_w"),
        (T.OuterProductMean, "__call__", "opm"), (T.Transition, "__call__", "transition"),
        (T.DiffusionTransformerLayer, "__call__", "dit_layer")]:
    wrap_method(cls, meth, key)
wrap_func(P, "edm_sample", "sampler")
wrap_func(P, "msa_update_chunks", "upd_chunks")
_den = P.DiffusionModule.__dict__["denoise"]
def _denoise(self, *a, **k):
    CUR["step"] += 1; REG.append("denoise")
    try: return _den(self, *a, **k)
    finally: REG.pop()
P.DiffusionModule.denoise = _denoise
_rr = OPS.recycle_region
def _recycle_region(cyc, last):
    CUR["cyc"] = cyc; return _rr(cyc, last)
OPS.recycle_region = _recycle_region

# ---------------- per-op attribution by runtime id ----------------
OPID = ttnn._ttnn.get_device_operation_id
def callsite():
    f = sys._getframe(2); out = []
    while f is not None and len(out) < 3:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn:
            out.append(f"{fn.split('/tt_bio/')[1]}:{f.f_lineno}:{f.f_code.co_name}")
        f = f.f_back
    return out
def tinfo(t):
    try:
        d = dict(shape=list(t.shape), padded=list(t.padded_shape), dtype=str(t.dtype).split(".")[-1],
                 layout=str(t.layout).split(".")[-1])
        if t.storage_type() == ttnn.StorageType.DEVICE:
            d["mem"] = str(t.memory_config())
        return d
    except Exception as e:
        return {"err": type(e).__name__}
def arginfo(x, depth=0):
    if isinstance(x, ttnn.Tensor): return {"T": tinfo(x)}
    if isinstance(x, (list, tuple)) and depth < 2 and len(x) <= 16: return [arginfo(y, depth + 1) for y in x]
    if isinstance(x, (int, float, bool, str)) or x is None: return x
    r = repr(x)
    return r if len(r) < 1500 else r[:1500] + "..."
def skey(x, depth=0):
    """Cheap identity of an argument for signature dedup: tensor shape+dtype+memory layout, scalars by value."""
    if isinstance(x, ttnn.Tensor):
        try: return (tuple(x.shape), int(x.dtype.value) if hasattr(x.dtype, "value") else str(x.dtype),
                     x.is_sharded() if x.storage_type() == ttnn.StorageType.DEVICE else None)
        except Exception: return "T?"
    if isinstance(x, (list, tuple)):
        return (len(x),) + tuple(skey(y, depth + 1) for y in x[:4]) if depth < 2 else len(x)
    if isinstance(x, (int, float, bool, str)) or x is None: return x
    return type(x).__name__ + ":" + str(hash(repr(x)) if depth == 0 else "")
KEYS = ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]", "DEVICE TRISC0 KERNEL DURATION [ns]",
        "DEVICE TRISC1 KERNEL DURATION [ns]", "DEVICE TRISC2 KERNEL DURATION [ns]",
        "DEVICE BRISC KERNEL DURATION [ns]", "DEVICE NCRISC KERNEL DURATION [ns]")
SHORT = ("k", "fw", "t0", "t1", "t2", "br", "nc")
F = {"sig": None, "ops": None, "progs": None}
SIGS = {}; ST = {"on": False, "depth": 0, "last": 0, "flushed": 0, "batch": 0, "nprog": 0, "nops": 0, "missing": 0}
def drain():
    ttnn.synchronize_device(DEV["d"]); ttnn.ReadDeviceProfiler(DEV["d"])
    hi = OPID(); n = 0; seen = set()
    for chip, progs in ttnn.get_latest_programs_perf_data().items():
        for p in progs:
            r = p.program_analyses_results; u = p.program_execution_uid
            d = [u.runtime_id, ST["batch"], p.core_count]
            for k in KEYS:
                d.append(r[k].duration if k in r else None)
            kk = r.get("DEVICE KERNEL DURATION [ns]")
            d += [kk.start_timestamp, kk.end_timestamp] if kk is not None else [None, None]
            if F["progs"]: F["progs"].write(json.dumps(d) + "\n")
            seen.add(u.runtime_id); n += 1
    # every id handed out since the last flush should have come back once
    miss = sum(1 for i in range(ST["flushed"], hi) if i not in seen)
    if miss and ST["on"]:
        ST["missing"] += miss
        # a profiler buffer that overflowed drops a run of programs; shrink the batch. A stray id or two is
        # an id consumed without a launch, not an overflow.
        if miss > 0.01 * (hi - ST["flushed"]) and FLUSH[0] > 50:
            FLUSH[0] //= 2; log(ev="flush_missing", missing=miss, expected=hi - ST["flushed"], new_flush=FLUSH[0])
    ST["flushed"] = hi; ST["batch"] += 1; ST["nprog"] += n
def emit(name, a, k, id0, id1):
    site = callsite(); reg = "/".join(REG)
    key = (name, tuple(site), reg, tuple(skey(x) for x in a), tuple((kk, skey(v)) for kk, v in k.items()))
    sid = SIGS.get(key)
    if sid is None:
        sid = SIGS[key] = len(SIGS)
        F["sig"].write(json.dumps(dict(sig=sid, op=name, site=site, reg=reg, args=[arginfo(x) for x in a],
                                       kw={kk: arginfo(v) for kk, v in k.items()}), default=str) + "\n")
    F["ops"].write(f"[{sid},{id0},{id1},{CUR['cyc']},{CUR['step']}]\n"); ST["nops"] += 1
    return sid
def make(orig):
    def call(self, *a, **k):
        if not ST["on"] or ST["depth"] > 0:
            return orig(self, *a, **k)
        ST["depth"] += 1
        try:
            id0 = OPID()
            if id0 > ST["last"]:
                emit("<unhooked>", (), {}, ST["last"], id0)
            out = orig(self, *a, **k)
            id1 = OPID(); ST["last"] = id1
            if id1 > id0:
                sid = emit(self.python_fully_qualified_name, a, k, id0, id1)
                if sid not in OUTS:
                    OUTS.add(sid); F["sig"].write(json.dumps(dict(sig=sid, out=arginfo(out)), default=str) + "\n")
            if id1 - ST["flushed"] >= FLUSH[0]:
                drain()
            return out
        finally:
            ST["depth"] -= 1
    return call
OUTS = set()
D.FastOperation.__call__ = make(D.FastOperation.__call__)
D.Operation.__call__ = make(D.Operation.__call__)

state = W._WorkerState("tenstorrent")
DEV = {}
t = time.monotonic(); DEV["d"] = T.get_device(); log(ev="device_open", s=time.monotonic() - t)
opened = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
NODE = opened[0]; log(ev="nodes_open", nodes=opened, grid=str(DEV["d"].compute_with_storage_grid_size()))
CKPT = cfg0["protenix_ckpt"]
state.model = P.Protenix.load_from_checkpoint(CKPT)
state.bind_run("lpx", dict(cfg0))
state.model_id = cfg0["model"]; state.config_hash = W.run_config_hash(cfg0)
m = state.model
log(ev="build", trunk_fid=str(m.trunk.compute_kernel_config.math_fidelity), diff_dtype=str(m.diffusion.dtype),
    n_cycles=m.trunk.N_CYCLES)
drain()

for name, cyc, steps in FOLDS:
    rcfg = dict(cfg0, seed=101, recycling_steps=cyc, sampling_steps=steps)
    sdir = OUT / f"struct_{name}"; sdir.mkdir(exist_ok=True); rcfg["struct_dir"] = str(sdir)
    CUR.update(cyc=-1, step=-1)
    for kk in ("sig", "ops", "progs"):
        F[kk] = open(OUT / f"{kk}_{name}.jsonl", "w", buffering=1 << 20)
    SIGS.clear(); OUTS.clear()
    drain(); ST.update(on=True, last=OPID(), nprog=0, nops=0, missing=0, batch=0)
    n0 = len(samples); t0 = time.monotonic(); err = None
    try:
        metrics, best, feats = state.predict_one(YAML, rcfg)
    except Exception:
        import traceback; err = traceback.format_exc()[-3000:]; metrics = {}
    t1 = time.monotonic()
    ST["on"] = False; drain()
    clk = sorted(r[NODE] for s_, r in samples[n0:] if t0 <= s_ <= t1)
    for kk in F:
        F[kk].close(); F[kk] = None
    log(ev="fold", fold=name, wall_s=t1 - t0, err=err, n_ops=ST["nops"], n_sigs=len(SIGS), n_progs=ST["nprog"],
        missing=ST["missing"], flush=FLUSH[0], n_flush=ST["batch"], cycles=CUR["cyc"] + 1, steps=CUR["step"] + 1,
        aiclk_n=len(clk), aiclk_median=clk[len(clk) // 2] if clk else None, aiclk_min=clk[0] if clk else None,
        aiclk_max=clk[-1] if clk else None,
        metrics={k: metrics.get(k) for k in ("plddt", "ptm", "iptm", "n_residues", "msa_depth") if k in metrics})
log(ev="end")
os._exit(0)
