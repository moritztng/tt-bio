"""spd-diffusion: time Protenix-v2's diffusion sampler alone, 200 steps x 5 samples, on one chip.

Two modes, one process each.

  cond OUT CHIP YAML
      Runs the serving path (tt-bio predict, protenix-v2, 10 recycles, the host MSA cache) up to the
      end of the trunk and saves the diffusion conditioning to OUT/cond.pt. Done once per target;
      every arm then samples from the same conditioning.

  diff OUT CHIP COND ARM SEEDS [census]
      Builds the model, then per seed: a fresh conditioning from COND (the per-fold hoists, DiT pair
      biases and atom terms, are inside the timed region, as in a fold), one `edm_sample` with the
      fold's own arguments, a device sync, the conditioning freed. AICLK is sampled every 0.5 s
      during each rep. The first rep is cold (program cache); later ones are warm. Coordinates are
      saved per rep so two engines can be compared on the same seed. ARM names the engine/setting
      in the records; the engine itself is whatever tt_bio is first on PYTHONPATH.
      With `census`, one extra rep runs with region timers (device-synced) on the diffusion parts.

Records: OUT/bench.jsonl. Compare: compare.py A.pt B.pt.
"""
import gc, glob, json, os, sys, threading, time
from pathlib import Path

MODE, OUT, CHIP = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time()
    LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush(); print(json.dumps(kw, default=str), flush=True)


NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
samples = []


def _sampler():
    while True:
        row = {}
        for n in NODES:
            try:
                row[n] = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            except Exception:
                row[n] = -1
        samples.append((time.monotonic(), row)); time.sleep(0.5)


threading.Thread(target=_sampler, daemon=True).start()

from tt_bio import main as M, runtime
from tt_bio import worker as W
from tt_bio.host_controller import worker_payload

captured = {}


class _Stop(Exception):
    pass


def _grab(*a, **kw):
    captured["payload"] = a[1] if isinstance(a[0], str) else a[0]; raise _Stop


def serving_cfg(yaml):
    """The config tt-bio predict would hand a worker for this target (pfm-ttfast's serving path)."""
    M._dispatch_run = _grab; M._dispatch_to_controller = _grab
    argv = ["predict", str(yaml), "--model", "protenix-v2", "--diffusion_samples", "5",
            "--recycling_steps", "10", "--accelerator", "tenstorrent", "--output_format", "cif",
            "--msa_db_path", os.path.expanduser("~/japanfold/msa/db"), "--msa_dir", str(OUT / "msa"),
            "--out_dir", str(OUT / "cli")]
    try:
        M.cli.main(argv, standalone_mode=False)
    except _Stop:
        pass
    return dict(captured["payload"]["config"])


def worker_env():
    slot = runtime.build_local_workers("tenstorrent", [object()], [CHIP])[0]
    winfo = worker_payload(slot)
    W._apply_tt_environment(winfo); W._bind_host_threads()
    return winfo


def opened_node():
    return sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1])
                   for fd in os.listdir("/proc/self/fd")
                   if os.path.exists(f"/proc/self/fd/{fd}")
                   and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})[0]


HOST_KEYS = ("s_trunk", "s_inputs", "pair_z", "c_l", "p_lm", "S", "mask_trunked")

if MODE == "cond":
    YAML = Path(sys.argv[4])
    cfg = serving_cfg(YAML)
    winfo = worker_env(); W._ensure_local_artifacts(cfg)
    import torch
    import tt_bio.protenix as P
    orig = P.Protenix._trunk_cond

    def trunk_cond(self, *a, **k):
        cond, aux = orig(self, *a, **k)
        torch.save({"cond": {k2: cond[k2] for k2 in HOST_KEYS}, "N": aux["N"], "NT": aux["NT"],
                    "ckpt": cfg["protenix_ckpt"], "yaml": str(YAML)}, OUT / "cond.pt")
        log(ev="cond_saved", N=aux["N"], NT=aux["NT"])
        raise _Stop
    P.Protenix._trunk_cond = trunk_cond
    state = W._WorkerState("tenstorrent")
    state.model = P.Protenix.load_from_checkpoint(cfg["protenix_ckpt"])
    state.bind_run("spd", cfg)
    state.model_id = cfg["model"]; state.config_hash = W.run_config_hash(cfg)
    try:
        state.predict_one(YAML, dict(cfg, seed=101, struct_dir=str(OUT / "struct")))
    except _Stop:
        pass
    log(ev="end"); os._exit(0)

# ---------------------------------------------------------------- diff
COND, ARM, SEEDS = Path(sys.argv[4]), sys.argv[5], [int(x) for x in sys.argv[6].split(",")]
CENSUS = len(sys.argv) > 7 and sys.argv[7] == "census"
saved = __import__("torch").load(COND, weights_only=False)
winfo = worker_env()
import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.protenix as P
from tt_bio.esmc import _free_ttnn_tensors

eng = Path(P.__file__).resolve().parents[1]
head = (eng / "HEAD").read_text().strip() if (eng / "HEAD").exists() else "?"
log(ev="start", mode="diff", arm=ARM, seeds=SEEDS, engine=str(eng), engine_head=head, chip=CHIP,
    N=saved["N"], NT=saved["NT"], env={k: v for k, v in os.environ.items()
                                       if k.startswith(("TT_BIO_", "PROTENIX_", "TT_VISIBLE"))})
dev = T.get_device()
t = time.monotonic(); m = P.Protenix.load_from_checkpoint(saved["ckpt"]); dm = m.diffusion
NODE = opened_node()
log(ev="build", s=time.monotonic() - t, node=NODE, diffusion_dtype=str(dm.dtype),
    diff_fidelity=str(dm.compute_kernel_config.math_fidelity),
    diff_acc=dm.compute_kernel_config.fp32_dest_acc_en, lpx=getattr(T, "LPX", None))

# region timers (census rep only): device-synced, so they add sync stalls; compare like with like
STACK, REG, ON = [], {}, {"v": False}


def timed(key, fn, *a, **kw):
    if not ON["v"]:
        return fn(*a, **kw)
    ttnn.synchronize_device(dev); t0 = time.perf_counter(); STACK.append(key)
    try:
        return fn(*a, **kw)
    finally:
        STACK.pop(); ttnn.synchronize_device(dev); dt = time.perf_counter() - t0
        path = "/".join(STACK + [key])
        nd = REG.setdefault(path, {"n": 0, "incl": 0.0, "child": 0.0}); nd["n"] += 1; nd["incl"] += dt
        if STACK:
            REG.setdefault("/".join(STACK), {"n": 0, "incl": 0.0, "child": 0.0})["child"] += dt


def wrap(cls, meth, key):
    f = cls.__dict__.get(meth)
    if f is not None:
        setattr(cls, meth, lambda self, *a, _f=f, **k: timed(key, _f, self, *a, **k))


for cls, meth, key in [(P.DiffusionModule, "_denoise_multiplicity", "denoise"),
                       (P.DiffusionModule, "_atom_cond", "atom_cond"),
                       (P.DiffusionModule, "_dit_block_biases", "dit_biases"),
                       (P.DiffusionModule, "_token_dit_device", "dit"),
                       (P.AtomTransformer, "__call__", "atom_tx"),
                       (T.AttentionPairBias, "__call__", "apb")]:
    wrap(cls, meth, key)


def fresh_cond():
    cond = {k: v.clone() for k, v in saved["cond"].items()}
    if dm.device_dit:
        cond["dit_z"] = dm._dit_z_device(cond["pair_z"])
    return cond


def med(v):
    return v[len(v) // 2] if v else None


reps = [(sd, "cold" if i == 0 else "warm", False) for i, sd in enumerate(SEEDS)]
if CENSUS:
    reps.append((SEEDS[-1], "census", True))
for sd, kind, cen in reps:
    REG.clear(); ON["v"] = cen
    ttnn.synchronize_device(dev)
    n0 = len(samples); t0 = time.monotonic()
    cond = fresh_cond()
    err = None
    try:
        x = P.edm_sample(dm, cond, saved["N"], n_step=200, multiplicity=5,
                         max_parallel_samples=P.DEFAULT_MAX_PARALLEL_SAMPLES, seed=sd)
        ttnn.synchronize_device(dev)
    except Exception:
        import traceback; err = traceback.format_exc()[-3000:]; x = None
    t1 = time.monotonic(); ON["v"] = False
    _free_ttnn_tensors(cond); del cond; gc.collect()
    clk = sorted(r[NODE] for s_, r in samples[n0:] if t0 <= s_ <= t1)
    rec = dict(ev="rep", arm=ARM, seed=sd, kind=kind, sample_s=t1 - t0, err=err, node=NODE,
               aiclk_n=len(clk), aiclk_median=med(clk), aiclk_min=clk[0] if clk else None,
               aiclk_max=clk[-1] if clk else None)
    if x is not None:
        x = x.detach().float().cpu()
        rec.update(finite=bool(torch.isfinite(x).all()), shape=list(x.shape))
        torch.save(x, OUT / f"coords_{ARM}_s{sd}_{kind}.pt")
    if cen:
        rec["regions"] = [{"path": p, "n": d["n"], "incl_s": round(d["incl"], 4),
                           "self_s": round(d["incl"] - d["child"], 4)} for p, d in sorted(REG.items())]
    log(**rec)
log(ev="end"); os._exit(0)
