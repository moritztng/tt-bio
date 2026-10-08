"""lpx-sdpa capture: every ttnn op inside every attention module of a real Protenix-v2 fold on ONE
Wormhole chip, recorded with the exact arguments a microbench needs to replay it.

Serving-path setup copied from perf/pfm_ttfast/bench_wh.py (same lease, same MSA cache, same config).

Fold 1 (cold): RECORD. Every ttnn operation called while an attention module is on the Python stack is
logged by (attention path, op, argument spec); the first call of each distinct attention signature also
keeps its full op sequence, which is what replay.py re-runs.
Fold 2 (warm): TIME. ttnn.synchronize_device before and after every attention-module call, so each
class gets an inclusive per-call time and a call count. Nothing else is synced.

usage: capture.py OUT CHIP SHARE YAML
"""
import gc, glob, json, os, sys, threading, time
from pathlib import Path

OUT = Path(sys.argv[1]); CHIP = int(sys.argv[2]); SHARE = int(sys.argv[3]); YAML = Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "capture.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time()
    LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush(); print(json.dumps(kw, default=str)[:600], flush=True)

from tt_bio import runtime
if SHARE:
    os.environ.update(runtime.host_thread_cap_env(SHARE, None))
log(ev="start", chip=CHIP, share=SHARE, yaml=str(YAML))

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
MSA_DB = os.path.expanduser("~/japanfold/msa/db")
argv = ["predict", str(YAML), "--model", "protenix-v2", "--diffusion_samples", "5",
        "--recycling_steps", "10", "--accelerator", "tenstorrent", "--output_format", "cif",
        "--msa_db_path", MSA_DB, "--msa_dir", str(OUT / "msa"), "--out_dir", str(OUT / "cli")]
try:
    M.cli.main(argv, standalone_mode=False)
except _Stop:
    pass
cfg0 = dict(captured["payload"]["config"])
log(ev="cfg", argv=argv, cfg={k: v for k, v in cfg0.items() if "pass" not in k and "key" not in k})

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

# ---------------- argument spec ----------------
def obj_spec(o):
    """Readable, replayable description of a non-tensor argument (configs included)."""
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, (list, tuple)):
        return [obj_spec(x) for x in o][:16]
    if isinstance(o, ttnn.Tensor):
        return tensor_spec(o)
    if isinstance(o, torch.Tensor):
        return {"torch": list(o.shape), "dtype": str(o.dtype)}
    d = {"type": type(o).__name__}
    for a in dir(o):
        if a.startswith("_"):
            continue
        try:
            v = getattr(o, a)
        except Exception:
            continue
        if callable(v):
            continue
        d[a] = v if isinstance(v, (bool, int, float, str, type(None))) else str(v)[:200]
    if len(d) == 1:
        d["repr"] = str(o)[:300]
    return d

def tensor_spec(t):
    s = {"shape": list(t.shape)}
    try: s["padded"] = list(t.padded_shape)
    except Exception: pass
    for k, f in (("dtype", lambda: str(t.dtype)), ("layout", lambda: str(t.layout)),
                 ("mem", lambda: str(t.memory_config()))):
        try: s[k] = f()
        except Exception: s[k] = "?"
    return s

# ---------------- context stack + op recording ----------------
STACK = []
REC = {"on": False}
OPS = {}       # (path, op, argspec json) -> count
SEQ = {}       # attention signature -> op sequence of its first call
CUR = []       # op sequence of the attention call in flight (innermost attention only)
ATT_KEYS = ("triatt", "apb", "atom_attn")
TIMES = {}     # path -> [n, total_s]
TIME = {"on": False}
DEV = {}

def _attn_path():
    for i in range(len(STACK) - 1, -1, -1):
        if STACK[i][0] in ATT_KEYS:
            return "/".join(k for k, _ in STACK[: i + 1])
    return None

def wrap_op(holder, name, qual):
    f = getattr(holder, name)
    def w(*a, **k):
        if REC["on"] and STACK:
            path = _attn_path()
            if path is not None:
                spec = {"args": [obj_spec(x) for x in a],
                        "kwargs": {kk: obj_spec(v) for kk, v in k.items()}}
                key = (path, qual, json.dumps(spec, default=str, sort_keys=True))
                OPS[key] = OPS.get(key, 0) + 1
                CUR.append({"op": qual, **spec})
        return f(*a, **k)
    w.__wrapped_lpx__ = f
    setattr(holder, name, w)

n_wrapped = 0
for holder, prefix in ((ttnn, "ttnn"), (ttnn.transformer, "ttnn.transformer"),
                       (ttnn.experimental, "ttnn.experimental")):
    for name in list(vars(holder)):
        if name.startswith("_"):
            continue
        v = getattr(holder, name, None)
        if v is None or isinstance(v, type) or not callable(v):
            continue
        if "Operation" not in type(v).__name__ and name not in ("generic_op", "matmul", "linear"):
            continue
        try:
            wrap_op(holder, name, f"{prefix}.{name}"); n_wrapped += 1
        except Exception:
            pass
# tt-bio's own device programs reach the device through ttnn.generic_op; record their entry too
import tt_bio.sdpa_generic as SG
for name in ("sdpa",):
    if hasattr(SG, name):
        wrap_op(SG, name, f"sdpa_generic.{name}"); n_wrapped += 1
log(ev="wrapped_ops", n=n_wrapped)

def wrap_ctx(cls, meth, key):
    f = cls.__dict__.get(meth)
    if f is None:
        return False
    def w(self, *a, **k):
        tag = key
        if key == "triatt":
            tag = "triatt_end" if getattr(self, "ending", False) else "triatt_start"
        STACK.append((key, tag))
        is_att = key in ATT_KEYS
        outer = is_att and not any(s[0] in ATT_KEYS for s in STACK[:-1])
        if is_att and outer:
            CUR.clear()
        t0 = None
        if TIME["on"] and is_att and outer:
            ttnn.synchronize_device(DEV["d"]); t0 = time.perf_counter()
        try:
            return f(self, *a, **k)
        finally:
            path = "/".join(s[1] for s in STACK)
            if t0 is not None:
                ttnn.synchronize_device(DEV["d"]); dt = time.perf_counter() - t0
                tt = TIMES.setdefault(path, [0, 0.0]); tt[0] += 1; tt[1] += dt
            if REC["on"] and is_att and outer:
                ins = [tensor_spec(x) for x in a if isinstance(x, ttnn.Tensor)]
                sig = json.dumps([path, ins], sort_keys=True)
                if sig not in SEQ:
                    SEQ[sig] = {"path": path, "inputs": ins, "n": 0, "ops": list(CUR)}
                SEQ[sig]["n"] += 1
            STACK.pop()
    setattr(cls, meth, w)
    return True

CTX = [(P.Trunk, "__call__", "trunk"), (P.Trunk, "_msa", "msa"), (P.Trunk, "_template", "template"),
       (P.DiffusionModule, "denoise", "denoise"), (P.DiffusionModule, "_denoise_multiplicity", "denoise"),
       (P.AtomTransformer, "__call__", "atom_tx"), (P.AtomAttentionEncoder, "__call__", "aae"),
       (P.ConfidenceHead, "confidence", "confidence"), (P.ConfidenceHead, "confidence_device", "confidence"),
       (T.Pairformer, "__call__", "pairformer"), (T.MSA, "__call__", "msa_stack"),
       (T.DiffusionTransformer, "__call__", "dit"), (T.PairformerLayer, "__call__", "pf_layer"),
       (T.MSALayer, "__call__", "msa_layer"), (T.DiffusionTransformerLayer, "__call__", "dit_layer"),
       (T.TriangleAttention, "__call__", "triatt"), (T.AttentionPairBias, "__call__", "apb"),
       (P.AtomTransformer, "_attention", "atom_attn"), (P.AtomTransformer, "_attention_m", "atom_attn")]
log(ev="ctx", installed=[f"{c.__name__}.{m}->{k}" for c, m, k in CTX if wrap_ctx(c, m, k)])

state = W._WorkerState("tenstorrent")
t = time.monotonic(); dev = DEV["d"] = T.get_device(); log(ev="device_open", s=time.monotonic() - t)
opened = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1])
                 for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and
                 os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
NODE = opened[0]
log(ev="nodes_open", nodes=opened, grid=str(dev.compute_with_storage_grid_size()), arch=str(dev.arch()))

CKPT = cfg0["protenix_ckpt"]
state.model = P.Protenix.load_from_checkpoint(CKPT)
state.bind_run("lpx", dict(cfg0))
state.model_id = cfg0["model"]; state.config_hash = W.run_config_hash(cfg0)
m = state.model
log(ev="build", trunk_ckc=obj_spec(m.trunk.compute_kernel_config),
    diffusion_dtype=str(getattr(m.diffusion, "dtype", "?")),
    diffusion_ckc=obj_spec(m.diffusion.compute_kernel_config))

def med(v): return v[len(v) // 2] if v else None
for kind in ("record", "time"):
    REC["on"] = kind == "record"; TIME["on"] = kind == "time"
    rcfg = dict(cfg0, seed=101)
    sdir = OUT / f"struct_{kind}"; sdir.mkdir(exist_ok=True); rcfg["struct_dir"] = str(sdir)
    n0 = len(samples); t0 = time.monotonic()
    try:
        metrics, best, feats = state.predict_one(YAML, rcfg); err = None
    except Exception:
        import traceback; err = traceback.format_exc()[-3000:]; metrics = {}
    t1 = time.monotonic(); REC["on"] = TIME["on"] = False
    clk = sorted(r[NODE] for s_, r in samples[n0:] if t0 <= s_ <= t1)
    log(ev="fold", kind=kind, wall_s=t1 - t0, err=err, node=NODE, aiclk_n=len(clk), aiclk_median=med(clk),
        aiclk_min=clk[0] if clk else None, aiclk_max=clk[-1] if clk else None,
        metrics={k: metrics.get(k) for k in ("plddt", "ptm", "iptm", "n_residues", "msa_depth") if k in metrics})
    if kind == "record":
        with open(OUT / "ops.json", "w") as f:
            json.dump([{"path": p, "op": o, "n": n, **json.loads(s)} for (p, o, s), n in
                       sorted(OPS.items(), key=lambda kv: (kv[0][0], kv[0][1]))], f, indent=1, default=str)
        with open(OUT / "seq.json", "w") as f:
            json.dump(list(SEQ.values()), f, indent=1, default=str)
        log(ev="recorded", distinct_ops=len(OPS), signatures=len(SEQ),
            calls={v["path"]: v["n"] for v in SEQ.values()})
    else:
        with open(OUT / "times.json", "w") as f:
            json.dump({p: {"n": n, "total_s": s, "per_call_ms": 1e3 * s / n} for p, (n, s) in sorted(TIMES.items())},
                      f, indent=1)
        log(ev="timed", total_attn_s=sum(s for _, s in TIMES.values()))
log(ev="end")
os._exit(0)
