"""SPD speed harness: warm Protenix-v2 fold time on ONE chip, the serving path, AICLK sampled during the fold.

One process = one arm (a flag set) on one chip, over a list of inputs. Arms run in separate processes so no
arm can be handed another arm's compiled program or module-level state. Per input: one cold rep (first of
shape, compile) then N warm reps, each on its own seed. Every rep appends one JSON record to <out>/bench.jsonl.

    python perf/spd/bench.py --out RUN/exact --chip 3 --arm exact --inputs c730,l512 --warm 3
    python perf/spd/bench.py --out RUN/lpx   --chip 3 --arm lpx:TT_BIO_LPX=1 --inputs c730
    python perf/spd/bench.py --out RUN/fast  --chip 3 --arm fast:fast --inputs c730

ARM grammar: NAME[:K=V[,K=V...]][:fast]. K=V are environment variables set before tt_bio is imported;
`fast` passes --fast. Nothing here edits the model: an arm is only switches the engine already has.

Inputs live in --data (default ~/spd-data, built by perf/spd/make_inputs.py): <data>/inputs/<name>.yaml with
the MSA cache at <data>/msa, read cache-only, so every box folds the same alignment without a search.

Record fields: git sha + dirty, tt_bio version, arm, env (every TT_BIO_/PROTENIX_/TT_METAL_ variable), argv,
host, arch, chip, opened device nodes, input, tokens, seed, kind (cold|warm), fold_s (model.fold only),
wall_s (predict_one end to end), per-sample confidences by rank (samples_conf) with the sample CIFs in
struct_dir (<input>.cif is rank 0, <input>_model_<k>.cif rank k; grade.py scores these), aiclk median/min/max/n per opened node sampled every 0.5 s DURING the rep,
loadavg, finite (coords and confidences), plddt/ptm/iptm, coords digest. Coordinates of every sample are kept
as coords_<input>_s<seed>.pt so summarize.py can quote an arm's structural deviation in Angstrom.
"""
import argparse, gc, glob, hashlib, json, os, socket, subprocess, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--chip", required=True, type=int, help="UMD index handed to TT_VISIBLE_DEVICES")
ap.add_argument("--arm", default="exact")
ap.add_argument("--inputs", default="c730")
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--seed", type=int, default=101, help="cold rep seed; warm reps use seed+1..seed+warm")
ap.add_argument("--data", type=Path, default=Path("~/spd-data").expanduser())
ap.add_argument("--share", type=int, default=0, help="host thread share (runtime.host_thread_cap_env), 0 = all")
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--recycles", type=int, default=10)
ap.add_argument("--no-coords", action="store_true")
a = ap.parse_args()


def parse_arm(spec):
    parts = spec.split(":")
    env, fast = {}, False
    for p in parts[1:]:
        if p == "fast":
            fast = True
        elif p:
            env.update(kv.split("=", 1) for kv in p.split(","))
    return parts[0], env, fast


ARM, ARM_ENV, FAST = parse_arm(a.arm)
os.environ.update(ARM_ENV)
a.out.mkdir(parents=True, exist_ok=True)
LOG = open(a.out / "bench.jsonl", "a")
REPO = Path(__file__).resolve().parents[2]


def log(**kw):
    kw["t_unix"] = time.time()
    line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


SHA, DIRTY = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
ENV = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("TT_BIO_", "PROTENIX_", "TT_METAL_"))}

from tt_bio import runtime  # noqa: E402  (after the arm's environment is in place)
if a.share:
    os.environ.update(runtime.host_thread_cap_env(a.share, None))

# AICLK of every node, from sysfs, every 0.5 s for the life of the process. A dead ARC answers 0xFFFFFFFF
# without raising, so anything outside 100..3000 MHz is dropped as a non-reading.
NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
samples = []


def _sampler():
    while True:
        row = {}
        for n in NODES:
            try:
                v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
                if 100 <= v <= 3000:
                    row[n] = v
            except Exception:
                pass
        samples.append((time.monotonic(), row)); time.sleep(0.5)


threading.Thread(target=_sampler, daemon=True).start()

INPUTS = []
for name in a.inputs.split(","):
    y = a.data / "inputs" / f"{name}.yaml"
    if not y.exists():
        sys.exit(f"no input {y}; run perf/spd/make_inputs.py")
    INPUTS.append((name, y))

# The CLI builds the run config exactly as a user's `tt-bio predict` would; we take it and fold through the
# serving worker (predict_one), which is what JapanFold runs.
from tt_bio import main as M  # noqa: E402
captured = {}


class _Stop(Exception):
    pass


def _grab(*args, **kw):
    captured["payload"] = args[1] if isinstance(args[0], str) else args[0]
    raise _Stop


M._dispatch_run = _grab; M._dispatch_to_controller = _grab
argv = ["predict", str(INPUTS[0][1]), "--model", "protenix-v2", "--diffusion_samples", str(a.samples),
        "--recycling_steps", str(a.recycles), "--accelerator", "tenstorrent", "--output_format", "cif",
        "--msa_dir", str(a.data / "msa"), "--msa_cache_only", "--out_dir", str(a.out / "cli")]
if FAST:
    argv.append("--fast")
try:
    M.cli.main(argv, standalone_mode=False)
except _Stop:
    pass
cfg0 = dict(captured["payload"]["config"])

from tt_bio import worker as W  # noqa: E402
from tt_bio.host_controller import worker_payload  # noqa: E402
slot = runtime.build_local_workers("tenstorrent", [object()], [a.chip])[0]
winfo = worker_payload(slot)
W._apply_tt_environment(winfo); W._bind_host_threads()
W._ensure_local_artifacts(cfg0)
import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
import tt_bio.protenix as P  # noqa: E402

HEAD = dict(sha=SHA, dirty=DIRTY, version=getattr(tt_bio, "__version__", None), arm=ARM, arm_env=ARM_ENV,
            fast=FAST, host=socket.gethostname(), chip=a.chip, env=ENV, samples=a.samples, recycles=a.recycles)
log(ev="start", argv=sys.argv, cli=argv, worker=winfo, TT_VISIBLE_DEVICES=os.environ.get("TT_VISIBLE_DEVICES"),
    torch_threads=torch.get_num_threads(), affinity=len(os.sched_getaffinity(0)), **HEAD)

state = W._WorkerState("tenstorrent")
t = time.monotonic(); dev = T.get_device(); t_open = time.monotonic() - t
OPENED = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1])
                 for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}")
                 and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
try:
    ARCH = str(dev.arch()).split(".")[-1].lower()
except Exception as e:
    ARCH = f"?{type(e).__name__}"
CARD = {}
for n in OPENED:
    for attr in ("tt_card_type", "tt_serial"):
        try:
            CARD.setdefault(attr, []).append(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/{attr}").read_text().strip())
        except Exception:
            pass
log(ev="device_open", s=t_open, nodes=OPENED, arch=ARCH, card=CARD)

t = time.monotonic()
state.model = P.Protenix.load_from_checkpoint(cfg0["protenix_ckpt"])
state.bind_run("spd", dict(cfg0, fast=FAST))
state.model_id = cfg0["model"]; state.config_hash = W.run_config_hash(cfg0)
m = state.model
log(ev="build", s=time.monotonic() - t, fast=getattr(m, "_fast", None))
LAST = {}
orig = m.fold


def fold(*args, **kw):
    t0 = time.monotonic(); r = orig(*args, **kw); LAST["fold_s"] = time.monotonic() - t0
    coords, conf = r if isinstance(r, tuple) else (r, None)
    LAST["coords"] = coords.detach().float().cpu().clone()
    if conf is not None:
        LAST["conf"] = [{k: float(v) for k, v in c.items() if k in ("plddt", "ptm", "iptm")}
                        for c in (conf if isinstance(conf, list) else [conf])]
    feats = args[0] if args else kw.get("feats")
    LAST["feats"] = {k: v.detach().cpu().clone() for k, v in feats.items()
                     if k in ("asym_id", "atom_to_token_idx", "atom_to_token", "atom_mask", "ref_mask",
                              "token_index", "residue_index", "is_protein") and hasattr(v, "detach")}
    return r


m.fold = fold


def clock(t0, t1):
    win = [r for ts, r in samples if t0 <= ts <= t1]
    out = {}
    for n in OPENED:
        v = sorted(r[n] for r in win if n in r)
        if v:
            out[n] = dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v))
    return out


for name, y in INPUTS:
    seeds = [(a.seed, "cold")] + [(a.seed + i, "warm") for i in range(1, a.warm + 1)]
    for sd, kind in seeds:
        rcfg = dict(cfg0, seed=sd, fast=FAST)
        sdir = a.out / f"struct_{name}_s{sd}"; sdir.mkdir(exist_ok=True); rcfg["struct_dir"] = str(sdir)
        LAST.clear(); gc.collect()
        la0 = os.getloadavg(); t0 = time.monotonic()
        try:
            metrics, _best, _feats = state.predict_one(y, rcfg); err = None
        except Exception:  # a crashing rep is a result, not the end of the run
            import traceback; err = traceback.format_exc()[-3000:]; metrics = {}
        t1 = time.monotonic()
        c = LAST.get("coords")
        finite = bool(c is not None and torch.isfinite(c).all().item()
                      and all(v == v and abs(v) != float("inf") for d in LAST.get("conf") or [] for v in d.values()))
        digest = hashlib.sha256(c.numpy().tobytes()).hexdigest()[:16] if c is not None else None
        if c is not None and not a.no_coords:
            torch.save({"coords": c, "conf": LAST.get("conf"), "feats": LAST.get("feats")},
                       a.out / f"coords_{name}_s{sd}.pt")
        tokens = metrics.get("n_tokens") or metrics.get("n_residues")
        log(ev="rep", input=name, tokens=tokens, seed=sd, kind=kind, fold_s=LAST.get("fold_s"), wall_s=t1 - t0,
            aiclk=clock(t0, t1), arch=ARCH, nodes=OPENED, finite=finite, digest=digest, err=err,
            loadavg=[round(la0[0], 2), round(os.getloadavg()[0], 2)], conf=LAST.get("conf"),
            metrics={k: metrics.get(k) for k in ("plddt", "ptm", "iptm", "msa_depth", "n_tokens", "confidence_score")
                     if k in metrics},
            samples_conf=[{k: v for k, v in r.items() if not isinstance(v, dict)} for r in metrics.get("all_runs", [])],
            struct_dir=str(sdir), **HEAD)
log(ev="end")
os._exit(0)
