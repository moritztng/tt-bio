"""pfm-ttfast: Protenix-v2 precision levers on ONE Wormhole Galaxy chip, in the serving environment.

Derived from state/jgp/ours/bench_glx.py (the 535.59 s measurement): same serving-path setup
(host_thread_cap_env, build_local_workers, worker_payload, _WorkerState, predict_one), same MSA
cache, AICLK sampled DURING every fold. What it adds:

* ARMS in one process. The device stays open, so the tt-bio lease is held for the whole group and the
  agent cannot take the chip between arms. Each arm rebuilds the model with its precision settings
  and reads them back off the built model, so a log line proves what ran.
* Coordinates of all samples saved per rep (`coords_<arm>_s<seed>.pt`), for the per-sample,
  seed-matched accuracy comparison in compare.py.
* An optional CENSUS fold: call-path timers with device syncs (the pvx_ps census, extended to the
  diffusion internals), reported as a tree that sums to Protenix.fold.

usage: bench_wh.py OUT CHIP SHARE YAML PLAN
PLAN: ';'-separated  arm:seed[,seed...][:census]   e.g.  "exact:102,101,103:census;diff_bf16:102,101"
The first seed of each arm is its cold rep (compile), the second its timed warm rep.
"""
import gc, glob, json, os, re, sys, threading, time
from pathlib import Path

OUT = Path(sys.argv[1]); CHIP = int(sys.argv[2]); SHARE = int(sys.argv[3]); YAML = Path(sys.argv[4])
PLAN = [p.split(":") for p in sys.argv[5].split(";") if p]
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")
def log(**kw):
    kw["t_unix"] = time.time()
    LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush(); print(json.dumps(kw, default=str), flush=True)

# Arm -> settings. Every lever is an existing switch; nothing here edits the model.
ARMS = {
    "exact":      dict(),
    "diff_bf16":  dict(diff_fp32=False),
    "hifi2":      dict(fid="hifi2"),
    "hifi3":      dict(fid="hifi3"),
    "bias_b8":    dict(bias_b8=True),
    "fast":       dict(fast=True),
    "diff_bf16+hifi2": dict(diff_fp32=False, fid="hifi2"),
    "diff_bf16+hifi3": dict(diff_fp32=False, fid="hifi3"),
    # MSA-update placement, bit-exact by construction (every op in the update is per depth row):
    # the whole-depth path instead of 512-row chunks, and the chunked path at 4x the chunk width.
    "msa_whole":  dict(env={"TT_BIO_MSA_ROW_CHUNK_BUDGET_BYTES": str(1 << 31)}),
    "msa_c2048":  dict(env={"TT_BIO_MSA_ROW_CHUNK_SIZE": "2048"}),
    "diff_bf16+msa_whole": dict(diff_fp32=False, env={"TT_BIO_MSA_ROW_CHUNK_BUDGET_BYTES": str(1 << 31)}),
}


def lever_spec(arm):
    """An arm named `L=<set>[+lever|-lever...]` builds Protenix under that precision-lever set,
    e.g. `L=fast`, `L=fast-lofi`, `L=normal+opm_b8` (tenstorrent.LEVERS); None for other arms."""
    return T.parse_levers(arm[2:]) if arm.startswith("L=") else None


ARM_ENV = sorted({k for a in ARMS.values() for k in a.get("env", {})})

from tt_bio import runtime
if SHARE:
    os.environ.update(runtime.host_thread_cap_env(SHARE, None))
log(ev="start", chip=CHIP, share=SHARE, plan=PLAN, yaml=str(YAML))

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
log(ev="env", worker=winfo, TT_VISIBLE_DEVICES=os.environ.get("TT_VISIBLE_DEVICES"),
    torch_threads=torch.get_num_threads(), affinity=len(os.sched_getaffinity(0)))

DEVREF = {}
# ---- census: call-path timers with device syncs (pvx_ps/census.py, extended) ----
STACK, NODES_T, ON = [], {}, {"v": False}
def timed(key, fn, *a, **kw):
    if not ON["v"]:
        return fn(*a, **kw)
    ttnn.synchronize_device(DEVREF["d"]); t0 = time.perf_counter(); STACK.append(key)
    try:
        return fn(*a, **kw)
    finally:
        STACK.pop(); ttnn.synchronize_device(DEVREF["d"]); dt = time.perf_counter() - t0
        path = "/".join(STACK + [key])
        nd = NODES_T.setdefault(path, {"n": 0, "incl": 0.0, "child": 0.0}); nd["n"] += 1; nd["incl"] += dt
        if STACK:
            NODES_T.setdefault("/".join(STACK), {"n": 0, "incl": 0.0, "child": 0.0})["child"] += dt
installed = []
def wrap_method(cls, meth, key):
    f = cls.__dict__.get(meth)
    if f is None: return
    def w(self, *a, **k): return timed(key, f, self, *a, **k)
    setattr(cls, meth, w); installed.append(f"{cls.__name__}.{meth}->{key}")
def wrap_func(mod, name, key):
    f = getattr(mod, name, None)
    if f is None: return
    def w(*a, **k): return timed(key, f, *a, **k)
    setattr(mod, name, w); installed.append(f"{mod.__name__}.{name}->{key}")
for cls, meth, key in [
        (P.Protenix, "_trunk_cond", "trunk_cond"), (P.Protenix, "_atom_feat_inputs", "atom_feats"),
        (P.AtomAttentionEncoder, "__call__", "input_aae"),
        (P.Trunk, "__call__", "trunk"), (P.Trunk, "_msa", "msa"), (P.Trunk, "_template", "template"),
        (P.Protenix, "_diffusion_pair_cond", "diff_pair_cond"), (P.Protenix, "_plm_z_term", "plm_z_term"),
        (P.DiffusionModule, "denoise", "denoise"), (P.DiffusionModule, "_denoise_multiplicity", "denoise"),
        (P.AtomTransformer, "__call__", "atom_tx"),
        (P.ConfidenceHead, "confidence", "confidence"), (P.ConfidenceHead, "confidence_device", "confidence"),
        (P.ConfidenceHead, "z_base_device", "conf_zbase"), (P.ConfidenceHead, "_postprocess", "conf_post"),
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
log(ev="census_hooks", installed=installed)

if os.environ.get("PFM_DRY"):
    os._exit(0)
state = W._WorkerState("tenstorrent")
t = time.monotonic(); dev = DEVREF["d"] = T.get_device(); log(ev="device_open", s=time.monotonic() - t)
opened = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1])
                 for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and
                 os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
NODE = opened[0]
log(ev="nodes_open", nodes=opened)

CKPT = cfg0["protenix_ckpt"]
def build(arm):
    lv = lever_spec(arm)
    s = ARMS[arm] if lv is None else dict(levers=sorted(lv))
    state.model = None; gc.collect()
    os.environ["PROTENIX_DIFFUSION_FP32_DEVICE"] = "1" if s.get("diff_fp32", True) else "0"
    for k in ARM_ENV:
        os.environ.pop(k, None)
    os.environ.update(s.get("env", {}))
    T._TRUNK_MATH_FIDELITY = s.get("fid", "hifi4")
    T._TRIATT_BIAS_B8 = bool(s.get("bias_b8", False))
    T.set_fast_mode(bool(s.get("fast", False)))
    t = time.monotonic()
    state.model = P.Protenix.load_from_checkpoint(CKPT, levers=lv)
    state.bind_run("pfm", dict(cfg0, fast=bool(s.get("fast", False))))
    m = state.model
    state.model_id = cfg0["model"]; state.config_hash = W.run_config_hash(cfg0)
    def rb(f):
        try: return str(f())
        except Exception as e: return f"?{type(e).__name__}"
    got = dict(diffusion_dtype=rb(lambda: m.diffusion.dtype),
               trunk_fidelity=rb(lambda: m.trunk.compute_kernel_config.math_fidelity),
               diff_fidelity=rb(lambda: m.diffusion.compute_kernel_config.math_fidelity),
               fast=m._fast, levers=sorted(m._levers), triatt_bias_b8=T._TRIATT_BIAS_B8, triatt_b8=T._TRIATT_B8,
               msa_whole_at_1GiB=P._msa_take_whole_path(1 << 30), msa_chunk_rows=P._msa_row_chunk_size())
    log(ev="build", arm=arm, s=time.monotonic() - t, settings=s, readback=got)
    # capture coords + confidences of every fold
    orig = m.fold
    def fold(*a, **k):
        t0 = time.monotonic(); r = orig(*a, **k); LAST["fold_s"] = time.monotonic() - t0
        coords, conf = r if isinstance(r, tuple) else (r, None)
        LAST["coords"] = coords.detach().float().cpu().clone()
        LAST["conf"] = [{kk: float(vv) for kk, vv in c.items() if kk in ("plddt", "ptm", "iptm")}
                        for c in (conf if isinstance(conf, list) else [conf])] if conf is not None else None
        feats = a[0] if a else k.get("feats")
        LAST["feats"] = {kk: vv.detach().cpu().clone() for kk, vv in feats.items()
                         if kk in ("asym_id", "atom_to_token_idx", "atom_to_token", "is_protein",
                                   "atom_mask", "ref_mask", "token_index", "residue_index")
                         and hasattr(vv, "detach")}
        return r
    m.fold = fold
LAST = {}

def med(v): return v[len(v) // 2] if v else None
for item in PLAN:
    arm, seeds = item[0], [int(x) for x in item[1].split(",")]
    census = len(item) > 2 and item[2] == "census"
    build(arm)
    reps = [(sd, "cold" if i == 0 else "warm", False) for i, sd in enumerate(seeds)]
    if census:
        reps.append((seeds[1] if len(seeds) > 1 else seeds[0], "census", True))
    for sd, kind, cen in reps:
        rcfg = dict(cfg0, seed=sd, fast=bool(ARMS[arm].get("fast", False)))
        sdir = OUT / f"struct_{arm}_s{sd}_{kind}"; sdir.mkdir(exist_ok=True); rcfg["struct_dir"] = str(sdir)
        NODES_T.clear(); ON["v"] = cen; LAST.clear()
        la0 = open("/proc/loadavg").read().split()[:3]
        n0 = len(samples); t0 = time.monotonic()
        try:
            metrics, best, feats = state.predict_one(YAML, rcfg)
            err = None
        except Exception as e:  # an arm that crashes is a result, not the end of the group
            import traceback; err = traceback.format_exc()[-3000:]; metrics = {}
        t1 = time.monotonic(); ON["v"] = False
        win = [s for s in samples[n0:] if t0 <= s[0] <= t1]
        clk = sorted(r[NODE] for _, r in win)
        if "coords" in LAST:
            torch.save({"coords": LAST["coords"], "conf": LAST["conf"], "feats": LAST["feats"]},
                       OUT / f"coords_{arm}_s{sd}_{kind}.pt")
        log(ev="rep", arm=arm, seed=sd, kind=kind, wall_s=t1 - t0, fold_s=LAST.get("fold_s"), err=err,
            node=NODE, aiclk_n=len(clk), aiclk_median=med(clk), aiclk_min=clk[0] if clk else None,
            aiclk_max=clk[-1] if clk else None, loadavg_start=la0,
            loadavg_end=open("/proc/loadavg").read().split()[:3], conf=LAST.get("conf"),
            metrics={k: metrics.get(k) for k in ("plddt", "ptm", "iptm", "n_residues", "msa_depth",
                                                 "confidence_score") if k in metrics},
            census=[{"path": p, "n": d["n"], "incl_s": round(d["incl"], 4),
                     "self_s": round(d["incl"] - d["child"], 4)} for p, d in sorted(NODES_T.items())] if cen else None)
log(ev="end")
os._exit(0)
