"""lpx-census: op-level device profile of one Protenix-v2 fold on ONE Wormhole chip.

Needs a Tracy-enabled tt-metal (the pip wheel has the device profiler compiled out) and
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1.
Setup is pfm-ttfast's bench_wh.py (serving-path config, MSA cache), so the fold is the one users get.

Two folds in one process, both at a reduced cycle/step count that the analysis scales back up by call
count (trunk cycles x N_CYCLES/CYC, denoise steps x 200/STEPS):

  A  "op":   every top-level ttnn op is bracketed by synchronize + ReadDeviceProfiler, so the device
             programs read after it are exactly that op's, joined to its args, kwargs and call site.
             Programs launched outside any ttnn op (Tensor methods etc.) are caught by the flush before
             the next op and recorded as "unhooked" at that call site.
  B  "pipe": same fold, no per-op sync; a flush every FLUSH ops. Gives device-idle gaps under the real
             dispatch pattern (start/end timestamps of consecutive programs inside one flush batch).

usage: census.py OUT CHIP YAML CYC STEPS
"""
import json, os, sys, threading, time, glob
from pathlib import Path

OUT = Path(sys.argv[1]); CHIP = int(sys.argv[2]); YAML = Path(sys.argv[3])
CYC = int(sys.argv[4]); STEPS = int(sys.argv[5]); FLUSH = 400
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

# ---------------- per-op device time ----------------
HERE = os.path.abspath(__file__)
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
KEYS = ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]", "DEVICE TRISC0 KERNEL DURATION [ns]",
        "DEVICE TRISC1 KERNEL DURATION [ns]", "DEVICE TRISC2 KERNEL DURATION [ns]",
        "DEVICE BRISC KERNEL DURATION [ns]", "DEVICE NCRISC KERNEL DURATION [ns]",
        "DEVICE KERNEL FIRST TO LAST START [ns]")
SHORT = ("k", "fw", "t0", "t1", "t2", "br", "nc", "f2l")
def drain():
    ttnn.synchronize_device(DEV["d"]); ttnn.ReadDeviceProfiler(DEV["d"])
    out = []
    for chip, progs in ttnn.get_latest_programs_perf_data().items():
        for p in progs:
            r = p.program_analyses_results; d = {"rid": p.program_execution_uid.runtime_id, "cores": p.core_count}
            for k, s in zip(KEYS, SHORT):
                if k in r: d[s] = r[k].duration
            kk = r.get("DEVICE KERNEL DURATION [ns]")
            if kk is not None: d["s"], d["e"] = kk.start_timestamp, kk.end_timestamp
            out.append(d)
    out.sort(key=lambda d: d["rid"])
    return out
NODEV = {"ttnn.deallocate", "ttnn.to_torch", "ttnn.from_device", "ttnn.is_tensor_storage_on_device",
         "ttnn.get_memory_config", "ttnn.synchronize_device"}
MODE = {"m": None, "n": 0, "depth": 0}
REC = []
def make(orig):
    def call(self, *a, **k):
        if MODE["m"] is None or MODE["depth"] > 0:
            return orig(self, *a, **k)
        name = self.python_fully_qualified_name
        MODE["depth"] += 1
        try:
            if MODE["m"] == "op" and name not in NODEV:
                site = callsite(); reg = "/".join(REG); cs = (CUR["cyc"], CUR["step"])
                pre = drain()
                if pre:
                    REC.append(dict(i=len(REC), op="<unhooked>", site=site, reg=reg, cyc=cs[0], step=cs[1], progs=pre))
                out = orig(self, *a, **k)
                progs = drain()
                REC.append(dict(i=len(REC), op=name, site=site, reg=reg, cyc=cs[0], step=cs[1],
                                args=[arginfo(x) for x in a], kw={kk: arginfo(v) for kk, v in k.items()},
                                out=arginfo(out), progs=progs))
                return out
            out = orig(self, *a, **k)
            if MODE["m"] == "pipe":
                MODE["n"] += 1
                if MODE["n"] % FLUSH == 0:
                    REC.append(dict(i=len(REC), op="<batch>", reg="/".join(REG), cyc=CUR["cyc"], step=CUR["step"],
                                    progs=drain()))
            return out
        finally:
            MODE["depth"] -= 1
    return call
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

for mode in sys.argv[6].split(","):
    rcfg = dict(cfg0, seed=101, recycling_steps=CYC, sampling_steps=STEPS)
    sdir = OUT / f"struct_{mode}"; sdir.mkdir(exist_ok=True); rcfg["struct_dir"] = str(sdir)
    REC.clear(); CUR.update(cyc=-1, step=-1); MODE.update(m=mode, n=0)
    n0 = len(samples); t0 = time.monotonic(); err = None
    try:
        metrics, best, feats = state.predict_one(YAML, rcfg)
    except Exception:
        import traceback; err = traceback.format_exc()[-3000:]; metrics = {}
    MODE["m"] = None
    REC.append(dict(i=len(REC), op="<tail>", reg="", cyc=CUR["cyc"], step=CUR["step"], progs=drain()))
    t1 = time.monotonic()
    clk = sorted(r[NODE] for s_, r in samples[n0:] if t0 <= s_ <= t1)
    with open(OUT / f"ops_{mode}.jsonl", "w") as f:
        for r in REC: f.write(json.dumps(r, default=str) + "\n")
    log(ev="fold", mode=mode, wall_s=t1 - t0, err=err, n_rec=len(REC), cycles=CUR["cyc"] + 1, steps=CUR["step"] + 1,
        aiclk_n=len(clk), aiclk_median=clk[len(clk) // 2] if clk else None, aiclk_min=clk[0] if clk else None,
        aiclk_max=clk[-1] if clk else None,
        metrics={k: metrics.get(k) for k in ("plddt", "ptm", "iptm", "n_residues", "msa_depth") if k in metrics})
log(ev="end")
os._exit(0)
