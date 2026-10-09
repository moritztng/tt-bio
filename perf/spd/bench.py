"""SPD speed harness: warm fold time on ONE chip, the serving path, AICLK sampled during the fold.

One process = one arm (a flag set) on one chip, over a list of inputs. Arms run in separate processes so no
arm can be handed another arm's compiled program or module-level state. Per input: one cold rep (first of
shape, compile) then N warm reps, each on its own seed. Every rep appends one JSON record to <out>/bench.jsonl.

    python perf/spd/bench.py --out RUN/exact --chip 3 --arm exact --inputs c730,l512 --warm 3
    python perf/spd/bench.py --out RUN/lpx   --chip 3 --arm lpx:TT_BIO_LPX=1 --inputs c730
    python perf/spd/bench.py --out RUN/fast  --chip 3 --arm fast:fast --inputs c730
    python perf/spd/bench.py --out RUN/of3   --chip 3 --model openfold3 --inputs c730
    python perf/spd/bench.py --out RUN/bc2   --chip 3 --model bindcraft2 --warm 2

--model: protenix-v2 (default), opendde, openfold3 or boltz2 fold through `Worker.load_model` and
`predict_one`, the code a JapanFold worker runs, and time the model's own fold call (`fold`; Boltz-2's
`predict_step`). Protenix-v2 alone is built by hand, as before, because it is the one model that reads
the `L=` lever set. bindcraft2 is a design loop, not a fold: see "BindCraft 2" below.

ARM grammar: NAME[:K=V[,K=V...]][:L=<set>[+lever|-lever...]][:fast]. K=V are environment variables set
before tt_bio is imported; `fast` passes --fast; `L=` builds Protenix-v2 or OpenFold3 under that precision-lever set
(tenstorrent.LEVERS, e.g. `L=fast-lofi`, `L=normal+opm_b8`), otherwise the mode's own set. Nothing here edits the model: an arm is only switches the engine already has.
`L=` is refused for every other model: none of them reads tenstorrent.LEVERS, so the arm would be a no-op.

Inputs live in --data (default ~/spd-data, built by perf/spd/make_inputs.py): <data>/inputs/<name>.yaml with
the MSA cache at <data>/msa, read cache-only, so every box folds the same alignment without a search.

Record fields: git sha + dirty, tt_bio version, arm, env (every TT_BIO_/PROTENIX_/TT_METAL_ variable), argv,
host, arch, chip, opened device nodes, input, tokens, seed, kind (cold|warm), fold_s (model.fold only),
wall_s (predict_one end to end), per-sample confidences by rank (samples_conf) with the sample CIFs in
struct_dir (<input>.cif is rank 0, <input>_model_<k>.cif rank k; grade.py scores these), aiclk median/min/max/n per opened node sampled every 0.5 s DURING the rep,
loadavg, finite (coords and confidences), plddt/ptm/iptm, coords digest. Coordinates of every sample are kept
as coords_<input>_s<seed>.pt so summarize.py can quote an arm's structural deviation in Angstrom.
Records of a non-Protenix model also carry `model`; a Protenix-v2 record is unchanged.

BindCraft 2 (--model bindcraft2): one rep = one design trajectory, the unit a JapanFold campaign is
billed and capped in. It runs BindCraft 2's own campaign on its shipped PD-L1 example (examples/pdl1.json,
settings unedited) the way japanfold.bc2run does: `campaign_predictor` on this chip, validation on BindCraft
2's host JAX, one trajectory per card, `max_trajectories` = 1 + --warm. The campaign seed is --seed.
Trajectory 1 compiles (`cold`), the rest are `warm`. Per trajectory: design_s is BindCraft 2's
`run_trajectory` call (the gradient design on the card, every stage), validate_s is MPNN redesign + validation
of its sequences (host JAX, so device levers barely move it), traj_s both together; the speed number is
design_s. Design metrics are the trajectory's row in the project's trajectory CSV plus the accepted count.
A trajectory BindCraft 2 stops early (`terminated` names the stage) is shorter by design: compare completed
trajectories with completed ones.
--inputs names the binder length (default 80, `--inputs 80`); --data is unused.
"""
import argparse, gc, glob, hashlib, json, os, socket, subprocess, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--chip", required=True, type=int, help="UMD index handed to TT_VISIBLE_DEVICES")
ap.add_argument("--model", default="protenix-v2",
                choices=("protenix-v2", "opendde", "openfold3", "boltz2", "bindcraft2"))
ap.add_argument("--arm", default="exact")
ap.add_argument("--inputs", default="c730")
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--seed", type=int, default=101, help="cold rep seed; warm reps use seed+1..seed+warm")
ap.add_argument("--data", type=Path, default=Path("~/spd-data").expanduser())
ap.add_argument("--share", type=int, default=None,
                help="host thread share (runtime.host_thread_cap_env). Default: one share per TT chip on the host, "
                     "the way serving runs a worker per chip, so concurrent rows on one box do not contend. 0 = all")
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--recycles", type=int, default=None,
                help="Protenix-v2: 10. Other models: the engine's own default unless given")
ap.add_argument("--bc2", type=Path, default=Path("~/bcx_shipped/bc2").expanduser(),
                help="BindCraft 2 checkout (examples/pdl1.json); bindcraft itself must be importable")
ap.add_argument("--no-coords", action="store_true")
ap.add_argument("--dry", action="store_true", help="build the run config, log it, stop before weights and device")
a = ap.parse_args()
if a.recycles is None and a.model == "protenix-v2":
    a.recycles = 10


def parse_arm(spec):
    parts = spec.split(":")
    env, fast, lv = {}, False, None
    for p in parts[1:]:
        if p == "fast":
            fast = True
        elif p.startswith("L="):
            lv = p[2:]
        elif p:
            env.update(kv.split("=", 1) for kv in p.split(","))
    return parts[0], env, fast, lv


ARM, ARM_ENV, FAST, LEVER_SPEC = parse_arm(a.arm)
PV2 = a.model == "protenix-v2"
if LEVER_SPEC is not None and a.model == "openfold3":
    # OpenFold3 builds under `tenstorrent.mode_levers()`, which reads the set from TT_BIO_LEVERS.
    ARM_ENV["TT_BIO_LEVERS"] = LEVER_SPEC
elif LEVER_SPEC is not None and not PV2:
    sys.exit(f"L= arms build Protenix-v2 and OpenFold3 only; {a.model} reads no tenstorrent.LEVERS, so "
             f"`{a.arm}` would fold its default. Use K=V switches for {a.model}.")
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
if a.share is None:
    a.share = len(glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
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



def clock(t0, t1, nodes):
    win = [r for ts, r in samples if t0 <= ts <= t1]
    out = {}
    for n in nodes:
        v = sorted(r[n] for r in win if n in r)
        if v:
            out[n] = dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v))
    return out


def worker_env():
    """What a serving worker sets before it opens its chip: TT_VISIBLE_DEVICES, the lease, host threads."""
    from tt_bio import worker as W
    from tt_bio.host_controller import worker_payload
    winfo = worker_payload(runtime.build_local_workers("tenstorrent", [object()], [a.chip])[0])
    W._apply_tt_environment(winfo); W._bind_host_threads()
    return winfo


def opened_nodes():
    def target(fd):
        try:  # the runtime's threads open and close fds while we list them
            return os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            return ""
    return sorted({int(t.rsplit("/", 1)[1]) for t in map(target, os.listdir("/proc/self/fd"))
                   if t.startswith("/dev/tenstorrent/")})


if a.model == "bindcraft2":
    import contextlib, csv, traceback
    winfo = worker_env()
    sys.path.insert(0, str(a.bc2))
    import bindcraft.campaign as campaign  # noqa: E402
    from bindcraft.campaign_output import TRAJECTORY_STAGE, accepted_table, stage_table  # noqa: E402
    from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
    from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
    import tt_bio  # noqa: E402
    from tt_bio import bindcraft2 as bc2  # noqa: E402
    blen = int(a.inputs) if a.inputs.isdigit() else 80
    project = a.out / f"campaign_s{a.seed}"
    example = a.bc2 / "examples" / "pdl1.json"
    settings = cleaned_campaign_settings(read_settings(str(example), parse_setting_overrides(
        [f"campaign_seed={a.seed}", f"max_trajectories={1 + a.warm}", f"project_folder={project}",
         f"binder_lengths=[{blen}]"])))
    HEAD = dict(sha=SHA, dirty=DIRTY, engine=str(Path(tt_bio.__file__).parent), model=a.model, arm=ARM,
                arm_env=ARM_ENV, fast=FAST, host=socket.gethostname(), ncpu=os.cpu_count(), share=a.share,
                chip=a.chip, env=ENV, example=str(example), binder_length=blen, campaign_seed=a.seed)
    log(ev="start", argv=sys.argv, worker=winfo, TT_VISIBLE_DEVICES=os.environ.get("TT_VISIBLE_DEVICES"),
        settings={k: v for k, v in settings.items() if k != "targets"}, **HEAD)
    if a.dry:
        os._exit(0)
    traj, opened = [], []  # traj: one dict per trajectory (n, t0, design_s, validate_s, err)
    real_design, real_validate = campaign.run_trajectory, campaign.redesign_and_validate_binders

    def emit(i, t_end):
        tr = traj[i]
        rows = []
        with contextlib.suppress(OSError):
            rows = list(csv.DictReader(open(stage_table(str(project), TRAJECTORY_STAGE))))
        row = next((r for r in rows if r.get("trajectory") == str(tr["n"])), {})
        acc = 0
        with contextlib.suppress(OSError):
            acc = sum(1 for _ in csv.DictReader(open(accepted_table(str(project)))))
        nodes = opened_nodes() or opened
        log(ev="rep", input=f"pdl1_b{blen}", seed=a.seed, trajectory=tr["n"], kind="cold" if i == 0 else "warm",
            design_s=tr.get("design_s"), validate_s=tr.get("validate_s"), traj_s=t_end - tr["t0"],
            fold_s=tr.get("design_s"), aiclk=clock(tr["t0"], tr["t0"] + (tr.get("design_s") or 0), nodes),
            aiclk_traj=clock(tr["t0"], t_end, nodes), nodes=nodes, terminated=row.get("terminated") or None,
            finite=bool(row) and tr.get("err") is None, accepted_total=acc, csv_row=row,
            err=tr.get("err"), **HEAD)

    def design(*args, **kw):
        if traj:
            emit(len(traj) - 1, time.monotonic())
        traj.append(dict(n=len(traj) + 1, t0=time.monotonic()))
        t0 = time.monotonic()
        try:
            return real_design(*args, **kw)
        finally:
            traj[-1]["design_s"] = time.monotonic() - t0
            opened[:] = opened_nodes() or opened

    def validate(*args, **kw):
        t0 = time.monotonic()
        try:
            return real_validate(*args, **kw)
        finally:
            traj[-1]["validate_s"] = time.monotonic() - t0

    campaign.run_trajectory, campaign.redesign_and_validate_binders = design, validate
    weights = {k: os.environ[v] for k, v in (("af2_weights", "JAPANFOLD_BC2_AF2_WEIGHTS"),
                                            ("mpnn_weights", "JAPANFOLD_BC2_MPNN_WEIGHTS")) if os.environ.get(v)}
    try:
        # the card's trunk pool reads the same AF2 params the campaign does, not tt-bio's weights cache
        with bc2.campaign_predictor(**({"checkpoints": weights["af2_weights"]} if "af2_weights" in weights else {})):
            bc2.run_campaign(settings, str(project), trajectories_per_card=1, **weights)
    except Exception:
        if traj:
            traj[-1]["err"] = traceback.format_exc()[-3000:]
        else:
            log(ev="crash", err=traceback.format_exc()[-3000:], **HEAD)
    if traj:
        emit(len(traj) - 1, time.monotonic())
    log(ev="end")
    os._exit(0)

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
argv = ["predict", str(INPUTS[0][1]), "--model", a.model, "--diffusion_samples", str(a.samples),
        *(["--recycling_steps", str(a.recycles)] if a.recycles is not None else []),
        "--accelerator", "tenstorrent", "--output_format", "cif",
        "--msa_dir", str(a.data / "msa"), "--msa_cache_only", "--out_dir", str(a.out / "cli")]
if FAST:
    argv.append("--fast")
try:
    M.cli.main(argv, standalone_mode=False)
except _Stop:
    pass
cfg0 = dict(captured["payload"]["config"])
if a.dry:  # the run config the CLI built, before any weights or device
    log(ev="dry", model=a.model, cli=argv, cfg={k: v for k, v in cfg0.items() if "key" not in k and "pass" not in k})
    os._exit(0)

from tt_bio import worker as W  # noqa: E402
winfo = worker_env()
W._ensure_local_artifacts(cfg0)
import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
import tt_bio.protenix as P  # noqa: E402

HEAD = dict(sha=SHA, dirty=DIRTY, engine=str(Path(tt_bio.__file__).parent), arm=ARM, arm_env=ARM_ENV,
            fast=FAST, host=socket.gethostname(), ncpu=os.cpu_count(), share=a.share, chip=a.chip, env=ENV, samples=a.samples, recycles=a.recycles)
if not PV2:  # a Protenix-v2 record stays byte for byte what it was
    HEAD["model"] = a.model
log(ev="start", argv=sys.argv, cli=argv, worker=winfo, TT_VISIBLE_DEVICES=os.environ.get("TT_VISIBLE_DEVICES"),
    torch_threads=torch.get_num_threads(), affinity=len(os.sched_getaffinity(0)), **HEAD)

state = W._WorkerState("tenstorrent")
t = time.monotonic(); dev = T.get_device(); t_open = time.monotonic() - t


OPENED = opened_nodes()
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


def lever_set(spec):
    """`fast-lofi+opm_b8` -> the fast set without lofi, plus opm_b8."""
    import re
    toks = re.findall(r"([+-]?)([A-Za-z0-9_]+)", spec)
    out = T.parse_levers(toks[0][1])
    for sign, name in toks[1:]:
        out = out - T.parse_levers(name) if sign == "-" else out | T.parse_levers(name)
    return sorted(out)


LEVER_SET = None if LEVER_SPEC is None else lever_set(LEVER_SPEC)
if PV2:
    # What `Worker.load_model` does before it builds. Without it `load_from_checkpoint` reads fast mode
    # as off, takes NORMAL_LEVERS, and a `:fast` arm folds exact (same digest as `exact` at seed 101).
    T.set_fast_mode(FAST)
    # levers only when asked, so the same harness still runs a tree that predates them (main before SPD)
    state.model = P.Protenix.load_from_checkpoint(
        cfg0["protenix_ckpt"], **({} if LEVER_SET is None else dict(levers=LEVER_SET)))
    state.bind_run("spd", dict(cfg0, fast=FAST))
    state.model_id = cfg0["model"]; state.config_hash = W.run_config_hash(cfg0)
else:  # JapanFold's own load: fast mode, checkpoint and config exactly as a worker builds them
    state.load_model(dict(cfg0, fast=FAST))
    state.bind_run("spd", dict(cfg0, fast=FAST))
m = state.model
# a Protenix-family model built by the worker keeps its own fast flag and resets the global one
BUILT_FAST = T._FAST_MODE if PV2 else getattr(m, "_fast", T._FAST_MODE)
log(ev="build", s=time.monotonic() - t, fast=BUILT_FAST, levers=sorted(getattr(m, "_levers", ())))
LAST = {}
FOLD_CALL = "predict_step" if a.model == "boltz2" else "fold"
orig = getattr(m, FOLD_CALL)
CONF = {"plddt": "plddt", "ptm": "ptm", "iptm": "iptm"}


def unpack(r):
    """(coords, per-sample confidences) from whatever the model's fold call returns: Protenix-v2 and
    OpenDDE a (coords, conf) tuple, OpenFold3 a FoldResult, Boltz-2 its prediction dict."""
    if isinstance(r, dict):  # Boltz-2: per-sample tensors; complex_plddt is the scalar pLDDT
        if r.get("exception") or r.get("coords") is None:
            return None, None
        n = r["coords"].shape[0]
        keys = {"plddt": "complex_plddt", "ptm": "ptm", "iptm": "iptm"}
        return r["coords"], [{k: float(r[v][i]) for k, v in keys.items() if v in r} for i in range(n)]
    if hasattr(r, "samples"):  # OpenFold3
        return torch.stack(list(r.samples)), [{k: float(c[k]) for k in CONF if k in c} for c in r.confidence]
    coords, conf = r if isinstance(r, tuple) else (r, None)
    if conf is not None:
        conf = [{k: float(v) for k, v in c.items() if k in CONF} for c in (conf if isinstance(conf, list) else [conf])]
    return coords, conf


def fold(*args, **kw):
    t0 = time.monotonic(); r = orig(*args, **kw); LAST["fold_s"] = time.monotonic() - t0
    coords, conf = unpack(r)
    if coords is not None:
        LAST["coords"] = coords.detach().float().cpu().clone()
    if conf is not None:
        LAST["conf"] = conf
    feats = args[0] if args else kw.get("feats")
    if isinstance(feats, dict):
        LAST["feats"] = {k: v.detach().cpu().clone() for k, v in feats.items()
                         if k in ("asym_id", "atom_to_token_idx", "atom_to_token", "atom_mask", "ref_mask",
                                  "token_index", "residue_index", "is_protein") and hasattr(v, "detach")}
    return r


setattr(m, FOLD_CALL, fold)


for name, y in INPUTS:
    seeds = [(a.seed, "cold")] + [(a.seed + i, "warm") for i in range(1, a.warm + 1)]
    for sd, kind in seeds:
        rcfg = dict(cfg0, seed=sd, fast=FAST)
        sdir = a.out / f"struct_{name}_s{sd}"; sdir.mkdir(exist_ok=True); rcfg["struct_dir"] = str(sdir)
        LAST.clear(); gc.collect()
        la0 = os.getloadavg(); t0 = time.monotonic()
        try:
            metrics, _best, _feats = state.predict_one(y, rcfg); err = None
        except Exception as e:  # a crashing rep is a result, not the end of the run
            import traceback  # python frames + the message head; a ttnn message ends in a long C++ backtrace
            err = "".join(traceback.format_tb(e.__traceback__)[-12:]) + f"{type(e).__name__}: {str(e)[:600]}"
            metrics = {}
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
            aiclk=clock(t0, t1, OPENED), arch=ARCH, nodes=OPENED, finite=finite, digest=digest, err=err,
            loadavg=[round(la0[0], 2), round(os.getloadavg()[0], 2)], conf=LAST.get("conf"),
            metrics={k: metrics.get(k) for k in ("plddt", "ptm", "iptm", "msa_depth", "n_tokens", "confidence_score")
                     if k in metrics},
            samples_conf=[{k: v for k, v in r.items() if not isinstance(v, dict)} for r in metrics.get("all_runs", [])],
            struct_dir=str(sdir), **HEAD)
log(ev="end")
os._exit(0)
