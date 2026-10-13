"""spd-gap-esmf: op-level device profile of one ESMFold2 / ESMFold2-Fast fold on ONE chip, the GAP-table workload.

spd-census's id-attributed method (perf/spd_census/census.py on wk/spd-census, attribution code copied unchanged),
built through the same `tt_baseline.build_fold` call perf/of3_4xpd/xmodel_ab.py times (cdk2x2_512 page fixture,
shipped recycles/steps, 1 sample), so the census is of exactly the fold the GAP cell is. Region labels come from
ESMFold2's own module classes (trunk / lm_encoder / msa_enc / pub / trimul / swiglu / diffusion / ...).

Needs the Tracy tt-metal build (qb1 ~/spd/spd-census/tt-metal) and TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 (perf/spd_gap_esmf/census.sh sets them).
Folds: one warm-up fold (compile, not recorded), then one profiled full fold "full". Outputs in OUT as spd-census
(census.jsonl, sig_full.jsonl, ops_full.jsonl, progs_full.jsonl); perf/spd_census/analyze.py reads them.

    python perf/spd_gap_esmf/census_esm.py --out RUN/ef2 --model esmfold2 [--fast]
"""
import argparse, json, os, resource, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--model", default="esmfold2", choices=("esmfold2", "esmfold2-fast", "openfold3", "openbind"))
ap.add_argument("--fast", action="store_true")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--recycles", type=int, default=None)
ap.add_argument("--steps", type=int, default=None)
a = ap.parse_args()
os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "5000")
OUT = a.out; OUT.mkdir(parents=True, exist_ok=True)
FLUSH = [1000]
LOG = open(OUT / "census.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time(); LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush()
    print(json.dumps(kw, default=str)[:400], flush=True)

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE)); sys.path.insert(0, str(TREE / "scripts" / "gpu_vs_tt"))
import torch, ttnn
import ttnn.decorators as D
import tt_bio.tenstorrent as T
ESM = a.model.startswith("esmfold2")
if ESM:
    import tt_bio.esmfold2 as E
    import tt_bio.esmfold2_runtime as R
    import tt_bio.esmc as C
else:
    import tt_bio.openfold3_trunk as OT, tt_bio.openfold3_diffusion_transformer as ODT
    import tt_bio.openfold3_atom_transformer as OAT, tt_bio.openfold3_diffusion_decoder as ODD
    import tt_bio.openfold3_confidence as OC, tt_bio.openfold3_msa_embedder as OM, tt_bio.openfold3_template as OTP
import tt_baseline as B
from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                         _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd: os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model) if a.recycles is None else a.recycles
B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model) if a.steps is None else a.steps
log(ev="env", ttnn=ttnn.__file__, tree=str(TREE), TT_VISIBLE_DEVICES=os.environ.get("TT_VISIBLE_DEVICES"),
    recycles=B.RECYCLING_STEPS, steps=B.SAMPLING_STEPS, model=a.model, fast=a.fast,
    prof={k: os.environ.get(k) for k in os.environ if "PROFILER" in k})

NODES = []; samples = []
def _sampler():
    while True:
        row = {}
        for n in NODES:
            try:
                v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
                if 100 <= v <= 3000: row[n] = v
            except Exception: pass
        samples.append((time.monotonic(), row)); time.sleep(0.5)
threading.Thread(target=_sampler, daemon=True).start()

# ---------------- region stack (module path of every op) ----------------
REG = []; CUR = {"cyc": -1, "step": -1}
def _wrap(f, key):
    def w(*args, **k):
        REG.append(key)
        try: return f(*args, **k)
        finally: REG.pop()
    return w
def wrap_method(cls, meth, key):
    f = cls.__dict__.get(meth)
    if f is not None: setattr(cls, meth, _wrap(f, key))
def wrap_func(mod, name, key):
    f = getattr(mod, name, None)
    if f is not None: setattr(mod, name, _wrap(f, key))
if ESM:
    for cls, meth, key in [
            (E.FoldingTrunkModel, "__call__", "trunk"), (E.PairUpdateBlock, "__call__", "pub"),
            (T.TriangleMultiplication, "__call__", "trimul"), (C.SwiGLUFFN, "__call__", "swiglu"),
            (C.SwiGLUFFN, "residual", "swiglu"), (E.TransitionLayer, "__call__", "transition"),
            (E.MSAEncoderModel, "__call__", "msa_enc"), (E.OuterProductMean, "__call__", "opm"),
            (E.MSAPairWeightedAveraging, "__call__", "pwa"), (E.LanguageModelShimModel, "__call__", "lm_shim"),
            (C.ESMCHiddenStatesModel, "__call__", "esmc"), (E.DiffusionConditioningModel, "__call__", "diff_cond"),
            (E.DiffusionModuleModel, "prepare", "diff_prep"), (E.SWAAtomTransformerModel, "__call__", "atom_tx"),
            (E.DiffusionTransformerModel, "__call__", "dit"), (E.AttentionPairBias, "__call__", "apb"),
            (E.ConditionedTransitionBlock, "__call__", "cond_trans"), (E.DistogramHead, "__call__", "disto"),
            (E.AtomEncoder, "__call__", "atom_enc"), (E.AtomDecoder, "__call__", "atom_dec")]:
        wrap_method(cls, meth, key)
    wrap_func(R, "_trunk_on_device", "lm_encoder")
    _step = E.DiffusionModuleModel.__dict__["step"]
    def _denoise(self, *args, **k):
        CUR["step"] += 1; REG.append("denoise")
        try: return _step(self, *args, **k)
        finally: REG.pop()
    E.DiffusionModuleModel.step = _denoise
    _rep = E.report_progress
    def _report(stage, step=0, total=0):
        if stage == "trunk": CUR["cyc"] = step
        return _rep(stage, step, total)
    E.report_progress = _report
else:
    # OpenFold3 / OpenBind-0: the shared pairformer ops from tenstorrent.py plus OF3's own module classes.
    for cls, meth, key in [
            (OT.OF3Trunk, "__call__", "trunk"), (T.PairformerLayer, "__call__", "pf_layer"),
            (T.TriangleMultiplication, "__call__", "trimul"), (T.TriangleAttention, "__call__", "triatt"),
            (T.Transition, "__call__", "transition"), (T.AttentionPairBias, "__call__", "apb"),
            (T.OuterProductMean, "__call__", "opm"), (T.PairWeightedAveraging, "__call__", "pwa"),
            (OM.MSAModule, "__call__", "msa"), (OTP.TemplateEmbedder, "__call__", "template"),
            (ODT._DiTBlock, "__call__", "dit"), (OAT.OF3AtomTransformer, "__call__", "atom_tx"),
            (ODD.OF3AtomAttentionDecoder, "__call__", "atom_dec"), (OC.OF3ConfidenceHead, "__call__", "confidence")]:
        wrap_method(cls, meth, key)

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
CKC = ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode", "dst_full_sync_en")
def ckc(x):
    """A compute kernel config's fields; its repr is only the object address."""
    return {"CKC": {a: str(getattr(x, a)).split(".")[-1] for a in CKC if hasattr(x, a)}}
def arginfo(x, depth=0):
    if isinstance(x, ttnn.Tensor): return {"T": tinfo(x)}
    if hasattr(x, "math_fidelity"): return ckc(x)
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
    if hasattr(x, "math_fidelity"): return str(ckc(x))
    return type(x).__name__ + ":" + str(hash(repr(x)) if depth == 0 else "")
KEYS = ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]", "DEVICE TRISC0 KERNEL DURATION [ns]",
        "DEVICE TRISC1 KERNEL DURATION [ns]", "DEVICE TRISC2 KERNEL DURATION [ns]",
        "DEVICE BRISC KERNEL DURATION [ns]", "DEVICE NCRISC KERNEL DURATION [ns]")
SHORT = ("k", "fw", "t0", "t1", "t2", "br", "nc")
F = {"sig": None, "ops": None, "progs": None}
SIGS = {}; ST = {"on": False, "depth": 0, "last": 0, "flushed": 0, "batch": 0, "nprog": 0, "nops": 0, "missing": 0}
def drain():
    ttnn.synchronize_device(DEV["d"])
    for attempt in range(4):  # the read asks ARC for the clock; on a loaded box ARC can miss its 1 s deadline
        try:
            ttnn.ReadDeviceProfiler(DEV["d"]); break
        except RuntimeError as e:
            if not any(m in str(e) for m in ("ARC to respond", "Failed to get AICLK")) or attempt == 3: raise
            log(ev="arc_retry", attempt=attempt); time.sleep(2)
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
            seen.add(u.runtime_id >> 10); n += 1   # mesh workloads carry (device op id << 10) | sub-id
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


fixdir = TREE / "perf" / "size512" / "fixtures"
tgt, a3m = fixdir / f"cdk2x2_{a.size}.yaml", fixdir / f"cdk2x2_{a.size}.a3m"
t = time.monotonic()
one_fold, meta = B.build_fold(a.model, TREE / f".msa_census_{a.model}_{a.size}", tgt, a3m, fast=a.fast)[:2]
DEV = {"d": T.get_device()}
log(ev="build", model=a.model, fast=a.fast, s=time.monotonic() - t, torch_threads=torch.get_num_threads(),
    grid=str(DEV["d"].compute_with_storage_grid_size()))
def _fdnode(fd):
    try: l = os.readlink(f"/proc/self/fd/{fd}")
    except OSError: return None
    return int(l.rsplit("/", 1)[1]) if l.startswith("/dev/tenstorrent/") else None
opened = sorted({n for n in map(_fdnode, os.listdir("/proc/self/fd")) if n is not None})
NODE = opened[0]; NODES[:] = opened; log(ev="nodes_open", nodes=opened)

t = time.monotonic(); s, m = one_fold(); log(ev="warmup", fold_s=s, wall_s=time.monotonic() - t, plddt=m.get("plddt"))
drain()
name = "full"
CUR.update(cyc=-1, step=-1)
for kk in ("sig", "ops", "progs"):
    F[kk] = open(OUT / f"{kk}_{name}.jsonl", "w", buffering=1 << 20)
SIGS.clear(); OUTS.clear()
drain(); ST.update(on=True, last=OPID(), nprog=0, nops=0, missing=0, batch=0)
n0 = len(samples); t0 = time.monotonic(); err = None; metrics = {}
try:
    fold_s, metrics = one_fold()
except Exception:
    import traceback; err = traceback.format_exc()[-3000:]
t1 = time.monotonic()
ST["on"] = False
try:
    drain()
except Exception:
    import traceback; err = (err or "") + "\ndrain: " + traceback.format_exc()[-1500:]
clk = sorted(r[NODE] for s_, r in samples[n0:] if t0 <= s_ <= t1 and NODE in r)
for kk in F:
    F[kk].close(); F[kk] = None
log(ev="fold", fold=name, wall_s=t1 - t0, err=err, n_ops=ST["nops"], n_sigs=len(SIGS), n_progs=ST["nprog"],
    missing=ST["missing"], flush=FLUSH[0], n_flush=ST["batch"], cycles=CUR["cyc"] + 1, steps=CUR["step"] + 1,
    aiclk_n=len(clk), aiclk_median=clk[len(clk) // 2] if clk else None, aiclk_min=clk[0] if clk else None,
    aiclk_max=clk[-1] if clk else None, maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
    metrics={k: metrics.get(k) for k in ("plddt", "ptm") if k in metrics})
log(ev="end")
os._exit(0)
