"""One warm fold worker per chip.

    TT_VISIBLE_DEVICES=0 python3 demo/sc26/engine/chipworker.py --chip 0

Reads jobs from stdin, one JSON object per line ({"id", "sequence", "seed"?, "steps"?}), and
writes protocol events (demo/sc26/PROTOCOL.md) to stdout, one JSON object per line. Every
library print is moved to stderr at startup, so stdout carries events and nothing else.

--model picks OpenFold3, Boltz-2 or ESMFold2. The frames are the diffusion sampler's own
coordinates, taken through the default-off hooks in tt_bio.openfold3_fold (OpenFold3.dump_fn),
tt_bio.boltz2 (Boltz2.dump_fn) and tt_bio.esmfold2 (set_trajectory_dump). Nothing is interpolated.
The last frame's coordinates are the scored structure's, bit for bit.

OpenFold3 and Boltz-2 read a protein's MSA from --msa-dir when one was searched ahead of time (the
attract proteins, demo/sc26/engine/msa, by tt-bio's sequence hash: <hash>.a3m for OpenFold3,
<hash>.csv for Boltz-2) and fold single-sequence otherwise (a visitor's name). The booth never
touches the network.

Every fold's clock starts when the chip takes the job, so its seconds and stage times include
featurisation ("prep"), and every stage event carries its time since then.

SIGUSR1 drops the fold in progress and keeps the worker warm (a visitor preempting an attract
fold). SIGINT drops it and exits cleanly, which leaves the chip usable. SIGTERM does the same, and
a worker whose main thread is stuck inside a device call (a hung chip, where no Python signal
handler can run) prints every thread's stack and exits by itself --term-grace-s later, so the
supervisor never has to kill it.
"""
import argparse
import base64
import faulthandler
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# stdout is the event channel: keep a private copy, then point fd 1 at stderr so nothing the
# device stack prints can corrupt it.
_EVENTS = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_LOCK = threading.Lock()
_PARENT_GONE = threading.Event()


def emit(**ev):
    ev.setdefault("t_wall", round(time.time(), 3))
    line = json.dumps(ev, separators=(",", ":"))
    with _LOCK:
        try:
            _EVENTS.write(line + "\n")
        except BrokenPipeError:  # the supervisor is gone: stop like SIGINT, the device closes at exit
            _PARENT_GONE.set()


def f32(t):
    """Tensor -> base64 little-endian float32, the protocol's coordinate encoding."""
    import numpy as np
    return base64.b64encode(np.ascontiguousarray(t.detach().cpu().float().numpy(), "<f4").tobytes()).decode()


def kabsch(mobile, target, w=None):
    """Rotation R and translation t with R @ mobile_i + t ~= target_i (least squares)."""
    import torch
    w = torch.ones(mobile.shape[0]) if w is None else w
    w = w / w.sum()
    mc, tc = (w[:, None] * mobile).sum(0), (w[:, None] * target).sum(0)
    h = ((mobile - mc) * w[:, None]).T @ (target - tc)
    u, _, vt = torch.linalg.svd(h.double())
    d = torch.sign(torch.det(vt.T @ u.T))
    r = (vt.T @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.float64)) @ u.T).float()
    return r, tc - r @ mc


class Clock:
    """Samples this chip's AICLK from sysfs while a fold runs (cheap, no tt-smi)."""

    def __init__(self, node):
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk")
        self.samples, self._run = [], False

    def read(self):
        from tt_bio.runtime import aiclk_reading
        try:
            return aiclk_reading(int(self.path.read_text().split()[0]))
        except (OSError, ValueError, IndexError):
            return None

    def __enter__(self):
        self.samples, self._run = [], True
        def loop():
            while self._run:
                v = self.read()
                if v is not None:
                    self.samples.append(v)
                time.sleep(0.2)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._run = False
        self._t.join()

    def summary(self):
        s = sorted(self.samples)
        if not s:
            return None
        return {"min": s[0], "median": s[len(s) // 2], "max": s[-1], "n": len(s)}


class Aborted(Exception):
    pass


class boltz2_runner:
    """Boltz-2 kept warm on this chip, loaded and featurised exactly as `tt-bio predict` does
    (tt_bio.main.boltz2_kwargs, tt_bio.worker._WorkerState), with no output files written."""

    def __init__(self, args, emit):
        from tt_bio import main as M
        from tt_bio.worker import _WorkerState
        self.M, self.msa_dir = M, Path(args.msa_dir)
        args.loops = args.loops or M._resolve_recycling_steps(None, "boltz2")
        args.steps = args.steps or M._resolve_sampling_steps(None, "boltz2")
        cache = Path(os.environ.get("BOLTZ_CACHE", Path.home() / ".boltz"))
        conf, _ = M.boltz2_kwargs(args.loops, args.steps)
        self.st = _WorkerState("tenstorrent")
        self.st.load_model({"model": "boltz2", "conf_ckpt": str(cache / "boltz2_conf.ckpt"),
                            "conf_kwargs": conf, "mol_dir": str(cache / "mols")})
        self.model = self.st.model

    def prepare(self, seq, seed, yaml=None):
        import tempfile
        import torch
        from tt_bio._vendor.esm.models.esmfold2.output import get_element_symbol
        if yaml:
            raise ValueError("the Boltz-2 worker folds one protein chain; a YAML job is OpenFold3's")
        h = self.M.seq_hash(seq)
        searched = any((self.msa_dir / f"{h}.{x}").is_file() for x in ("a3m", "csv"))
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "fold.yaml"
            y.write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n")
            feats, _ = self.M.prepare_features(
                y, ccd=self.st._ccd, mol_dir=self.st._mol_dir, msa_dir=self.msa_dir,
                tokenizer=self.st._tokenizer, featurizer=self.st._featurizer, use_msa=False,
                msa_url="", msa_strategy="greedy", msa_user=None, msa_pass=None, api_key=None,
                max_msa=8192, single_sequence=not searched)
        mask = torch.as_tensor(feats["atom_pad_mask"]).bool()
        atoms = {
            "element": [get_element_symbol(int(z)) for z in torch.as_tensor(feats["ref_element"])[mask].argmax(-1)],
            "name": ["".join(chr(int(c) + 32) for c in n if int(c)).strip()
                     for n in torch.as_tensor(feats["ref_atom_name_chars"])[mask].argmax(-1)],
            "residue": torch.as_tensor(feats["atom_to_token"])[mask].argmax(-1).tolist(),
        }
        return {"feats": feats, "mask": mask, "atoms": atoms, "msa": searched}

    def fold(self, prep, seed, progress, dump):
        import torch
        from tt_bio.runtime import seed_everything
        seed_everything(seed)
        batch = self.M.to_batch(prep["feats"], self.st.torch_device)
        self.model.progress_fn, self.model.dump_fn = progress, dump
        try:
            with torch.no_grad():
                pred = self.model.predict_step(batch)
        finally:
            self.model.progress_fn = self.model.dump_fn = None
        if pred.get("exception"):
            raise RuntimeError("Boltz-2 predict_step: Out of Memory")
        ntok = int(prep["feats"]["token_pad_mask"].sum())
        return pred["coords"][0].cpu(), pred["plddt"][0][:ntok].tolist(), round(float(pred["ptm"][0]), 4)


class openfold3_runner:
    """OpenFold3 kept warm on this chip, its inputs built exactly as `tt-bio predict` builds them
    (tt_bio.worker._WorkerState._openfold3_inputs), with no output files written. One diffusion
    sample, predict's default."""

    def __init__(self, args, emit):
        from tt_bio import main as M
        from tt_bio import weights
        from tt_bio.worker import _WorkerState, _template_structure_dir
        self.M, self.msa_dir = M, Path(args.msa_dir)
        args.loops = args.loops if args.loops is not None else M._resolve_recycling_steps(None, "openfold3")
        args.steps = args.steps or M._resolve_sampling_steps(None, "openfold3")
        self.cfg = {"model": "openfold3", "of3_ckpt": str(weights.fetch("openfold3")),
                    "recycling_steps": args.loops, "sampling_steps": args.steps, "diffusion_samples": 1,
                    "msa_dir": str(self.msa_dir), "template_structures": _template_structure_dir(weights.cache_root())}
        self.st = _WorkerState("tenstorrent")
        self.st.load_model(self.cfg)
        self.model = self.st.model

    def prepare(self, seq, seed, yaml=None):
        """`yaml` is a whole tt-bio input (several chains, ligands); without it, one protein chain.
        Every protein chain reads its <hash>.a3m from --msa-dir if there is one; a chain without
        one gets upstream's no-MSA input (the query alone), as `--single_sequence` does."""
        import tempfile
        import numpy as np
        import torch
        from tt_bio.main import _read_bio_chains
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "fold.yaml"
            y.write_text(yaml or f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n")
            prot = [c[1] for c in _read_bio_chains(y) if c[3] == "protein"]
            searched = [(self.msa_dir / f"{self.M.seq_hash(p)}.a3m").is_file() for p in prot]
            kwargs, features, _, _ = self.st._openfold3_inputs(
                y, dict(self.cfg, seed=seed, single_sequence=not all(searched)))
        aa = features["atom_array"]
        # One index per residue (a ligand is one residue), the way the structure file numbers them.
        new = np.ones(len(aa), dtype=bool)
        new[1:] = (aa.chain_id[1:] != aa.chain_id[:-1]) | (aa.res_id[1:] != aa.res_id[:-1])
        res = torch.as_tensor(np.cumsum(new) - 1)
        atoms = {"element": [str(e).capitalize() for e in aa.element], "name": [str(n) for n in aa.atom_name],
                 "residue": res.tolist(), "chain": [str(c) for c in aa.chain_id]}
        return {"kwargs": kwargs, "mask": torch.ones(len(aa), dtype=torch.bool), "atoms": atoms,
                "res": res, "msa": all(searched)}

    def fold(self, prep, seed, progress, dump):
        import torch
        own = self.model.dump_fn
        self.model.dump_fn = dump
        try:
            with torch.no_grad():
                res = self.model.fold(**dict(prep["kwargs"], seed=seed, progress_fn=progress))
        finally:
            self.model.dump_fn = own
        conf = res.confidence[res.best_index]
        # Per-residue pLDDT: the mean over each residue's atoms (the head scores atoms).
        r, n = prep["res"], int(prep["res"].max()) + 1
        per = torch.zeros(n).index_add_(0, r, conf["plddt_atom"].float()) / torch.bincount(r, minlength=n)
        return res.coordinates, per.tolist(), round(float(conf["ptm"]), 4)


RUNNERS = {"boltz2": boltz2_runner, "openfold3": openfold3_runner}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True, help="UMD chip id this worker is pinned to")
    ap.add_argument("--model", default="esmfold2", choices=["esmfold2", "boltz2", "openfold3"])
    ap.add_argument("--steps", type=int, default=None,
                    help="diffusion steps; default the model's own (ESMFold2 20, the others tt-bio predict's)")
    ap.add_argument("--loops", type=int, default=None,
                    help="trunk recycles; default the model's own (ESMFold2 3, the others tt-bio predict's)")
    ap.add_argument("--msa-dir", default=str(Path(__file__).resolve().parent / "msa"),
                    help="OpenFold3 / Boltz-2: MSAs searched ahead of time, by tt-bio's sequence hash")
    ap.add_argument("--warm", default="", help="a JSON list of {sequence, yaml?, name?} folded once "
                    "before the chip says ready, so no shown fold carries a compile")
    ap.add_argument("--term-grace-s", type=float, default=10,
                    help="after SIGTERM, a worker still inside the device this long exits by itself")
    ap.add_argument("--workers", type=int, default=1, help="chip workers sharing this host's CPU")
    args = ap.parse_args()
    # Each worker takes its share of the host. Left at torch's default, four pools of all cores
    # spin against each other and a 20-residue fold waits 2-4 s on the host instead of 0.6 s.
    from tt_bio.runtime import host_thread_cap_env
    os.environ.update(host_thread_cap_env(args.workers))

    abort, cancel, busy = threading.Event(), threading.Event(), threading.Event()
    import signal

    def on_stop(*_):
        abort.set()
        if not busy.is_set():  # idle, blocked on stdin: leave now; mid-fold, at the next step
            raise KeyboardInterrupt

    def exit_when_stuck(grace):
        # The C signal handler writes each signal's number to the wakeup fd even while the main
        # thread is blocked in C++, so this thread hears a SIGTERM the stuck thread never will.
        r, w = os.pipe()
        os.set_blocking(w, False)
        signal.set_wakeup_fd(w)

        def watch():
            while signal.SIGTERM not in os.read(r, 64):
                pass
            time.sleep(grace)  # a healthy worker has left by now through on_stop
            print(f"[chipworker] chip {args.chip}: still inside the device {grace:.0f} s after SIGTERM; "
                  "every thread's stack follows, then exit", file=sys.stderr, flush=True)
            faulthandler.dump_traceback(all_threads=True)
            os._exit(75)
        threading.Thread(target=watch, daemon=True).start()
    exit_when_stuck(args.term_grace_s)

    def install_handlers():
        # SIGINT and SIGTERM both stop cleanly (the device closes in tt_bio's atexit). Installed
        # explicitly, because a worker started from a non-interactive `cmd &` inherits SIGINT as
        # IGNORED, and once more after the device is open in case its stack replaced them.
        signal.signal(signal.SIGINT, on_stop)
        signal.signal(signal.SIGTERM, on_stop)
        signal.signal(signal.SIGUSR1, lambda *_: cancel.set())
    install_handlers()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio.runtime import umd_index_to_dev_node
    clock = Clock(umd_index_to_dev_node().get(args.chip, args.chip))

    t0 = time.perf_counter()
    emit(type="chip", chip=args.chip, state="warming", aiclk_mhz=clock.read())
    import torch
    from tt_bio.runtime import bind_host_threads
    bind_host_threads()
    from tt_bio import esmfold2 as E
    from tt_bio.esmfold2_runtime import build_spi, load_ttnn_esmfold2
    from tt_bio.tenstorrent import get_device
    from tt_bio._vendor.esm.models.esmfold2 import ESMFold2InputBuilder
    from tt_bio._vendor.esm.models.esmfold2.output import get_element_symbol
    from tt_bio._vendor.esm.models.esmfold2.processor import _seed_context

    import ttnn
    device = get_device()
    if args.model in RUNNERS:
        run = RUNNERS[args.model](args, emit)
    else:
        model = load_ttnn_esmfold2()
        model._esmc.preload()
        builder = ESMFold2InputBuilder()
        args.steps, args.loops = args.steps or 20, args.loops or 3
    install_handlers()
    loud, quiet = emit, lambda **ev: None

    def fold(job):
        emit = quiet if job.get("quiet") else loud
        jid, seq = job["id"], job["sequence"].strip().upper()
        n_res = len(seq.replace(":", ""))   # a complex's chains are joined by ":" (gallery/build.py)
        seed, steps = int(job.get("seed", 0)), int(job.get("steps", args.steps))
        t_start = time.perf_counter()
        if args.model in RUNNERS:
            prep = run.prepare(seq, seed, job.get("yaml"))
            mask, atoms = prep["mask"], prep["atoms"]
        else:
            features, chain_infos = builder.prepare_input(build_spi([("A", seq)]), seed=seed,
                                                          device=model.device)
            mask = features["atom_attention_mask"][0].bool()
            atoms = {
                "element": [get_element_symbol(int(z)) for z in features["ref_element"][0][mask]],
                "name": ["".join(chr(int(c) + 32) for c in n if int(c)).strip()
                         for n in features["ref_atom_name_chars"][0][mask]],
                "residue": features["atom_to_token"][0][mask].tolist(),
            }
        n_atoms = int(mask.sum())
        t_prep = time.perf_counter()
        stamps = {"prep": t_prep - t_start}
        phase = {"stage": "lm" if args.model == "esmfold2" else "trunk", "t": t_prep}

        def progress(stage, step=0, total=0):
            if abort.is_set() or cancel.is_set():
                raise Aborted()
            if job.get("quiet") and time.perf_counter() - beat[0] > 5:
                # A warm-up fold is not shown, but its stages are the chip working: the engine's
                # watchdog hears them, and the lane says what the chip is compiling for.
                beat[0] = time.perf_counter()
                loud(type="chip", chip=args.chip, state="warming", warm=job.get("warm"), name=job.get("name"),
                     n_res=n_res, stage=stage)
            # ttnn queues device work and returns, so without this a trunk recycle "ends" when its
            # last op is queued, seconds before the chip finishes it, and every recycle after the
            # first is stamped at the same instant. Waiting for the chip makes each stage's time
            # the chip's. It only waits; nothing the fold computes changes.
            ttnn.synchronize_device(device)
            if stage == "diffusion":
                phase["of"] = total  # the sampler's real step count (its schedule is clipped)
                if pending:  # the noise frame waited for that count
                    emit(**{**pending.pop(), "of": total})
            if stage != phase["stage"]:
                stamps[phase["stage"]] = time.perf_counter() - phase["t"]
                phase.update(stage=stage, t=time.perf_counter())
            emit(type="stage", id=jid, chip=args.chip, stage=stage, step=step, total=total,
                 t=round(time.perf_counter() - t_start, 3))

        prev = {"ref": None}
        pending = []
        beat = [0.0]

        def stop_check(step, x, x_den):
            if abort.is_set() or cancel.is_set():
                raise Aborted()

        def dump(step, x, x_den):
            if abort.is_set() or cancel.is_set():
                raise Aborted()
            x = x.detach().float().cpu()
            x_den = x_den.detach().float().cpu() if x_den is not None else None
            raw = x[0][mask]
            ref = x_den[0][mask] if x_den is not None else raw
            # Display alignment onto ONE fixed reference, the fold's first x0 (the network's first
            # estimate of the finished structure), fitted on this step's x0, which shares the
            # step's random frame and already has the protein's shape. Aligning each frame onto
            # the previous one chained the noise of every fit into a drift. The raw coordinates
            # are sent untouched; R and t are a camera, not an edit. The app re-superposes every
            # frame onto the final structure once the fold is done (web/render/src/trajectory.js).
            if prev["ref"] is None and x_den is not None:
                prev["ref"] = x_den[0][mask].clone()
            if prev["ref"] is None:
                r, t = torch.eye(3), torch.zeros(3)
            else:
                r, t = kabsch(ref, prev["ref"])
            frame = dict(type="frame", id=jid, chip=args.chip, step=step, of=phase.get("of"),
                         t=round(time.perf_counter() - t_start, 3), xyz=f32(raw),
                         x0=f32(x_den[0][mask]) if x_den is not None else None,
                         R=[round(v, 6) for v in r.flatten().tolist()], T=[round(v, 4) for v in t.tolist()])
            if step == -1:
                pending.append(frame)
            else:
                emit(**frame)

        emit(type="fold_start", id=jid, chip=args.chip, model=args.model, sequence=seq,
             n_res=n_res, n_atoms=n_atoms, steps=steps, loops=args.loops, seed=seed, atoms=atoms,
             rg_expected=round(2.2 * n_res ** 0.38, 2), source="live",
             t=round(t_prep - t_start, 3))
        hook = dump if job.get("frames", True) else stop_check
        if args.model in RUNNERS:
            with clock:
                final, plddt, ptm = run.fold(prep, seed, progress, hook)
            final = final[mask]
        else:
            E.set_progress(progress)
            E.set_trajectory_dump(hook)
            try:
                with clock, torch.no_grad(), _seed_context(seed):
                    out = model(**features, num_loops=args.loops, num_sampling_steps=steps,
                                num_diffusion_samples=1, early_exit=False)
                res = builder.decode(out, features, chain_infos, num_diffusion_samples=1)
            finally:
                E.set_progress(None)
                E.set_trajectory_dump(None)
            final, plddt, ptm = out["sample_atom_coords"][0][mask], res.plddt.flatten().tolist(), res.ptm
        stamps[phase["stage"]] = time.perf_counter() - phase["t"]
        total = time.perf_counter() - t_start
        emit(type="fold_done", id=jid, chip=args.chip, model=args.model, n_res=n_res,
             seconds=round(total, 3), stages={k: round(v, 3) for k, v in stamps.items()},
             aiclk_mhz=clock.summary(), xyz=f32(final),
             plddt=[round(float(v), 4) for v in plddt], ptm=ptm, source="live")

    busy.set()
    try:
        for i, w in enumerate(json.loads(Path(args.warm).read_text()) if args.warm else []):
            emit(type="chip", chip=args.chip, state="warming", warm=i, name=w.get("name"))   # the watchdog sees it alive
            try:
                fold({"id": f"warm{i}", "sequence": w["sequence"], "yaml": w.get("yaml"), "name": w.get("name"),
                      "warm": i, "frames": False, "quiet": True})
            except Aborted:
                raise
            except Exception as exc:  # report it and keep warming: one bad length must not cost the chip
                emit(type="fold_error", id=f"warm{i}", chip=args.chip,
                     reason=f"{type(exc).__name__}: {exc}"[:400])
    except Aborted:
        emit(type="chip", chip=args.chip, state="stopped")
        return
    busy.clear()
    emit(type="chip", chip=args.chip, state="ready", aiclk_mhz=clock.read(),
         load_s=round(time.perf_counter() - t0, 1), model=args.model)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            if job.get("type") == "quit":
                break
            emit(type="chip", chip=args.chip, state="busy", job=job.get("id"))
            cancel.clear()
            busy.set()
            recycle = False
            try:
                fold(job)
            except Aborted:
                emit(type="fold_error", id=job.get("id"), chip=args.chip,
                     reason="stopped" if abort.is_set() else "preempted")
            except Exception as exc:  # a bad input must not take the chip down
                reason = f"{type(exc).__name__}: {exc}"[:400]
                # Device DRAM fills up over hours of mixed lengths, and once full every fold
                # fails. A fresh worker starts with empty DRAM, so leave cleanly and let the
                # engine restart this one.
                recycle = "Out of Memory" in reason
                emit(type="fold_error", id=job.get("id"), chip=args.chip,
                     reason="out_of_memory" if recycle else reason, **({"detail": reason} if recycle else {}))
            finally:
                busy.clear()
            if abort.is_set() or _PARENT_GONE.is_set() or recycle:
                break
            emit(type="chip", chip=args.chip, state="ready", aiclk_mhz=clock.read())
    except KeyboardInterrupt:
        pass
    emit(type="chip", chip=args.chip, state="stopped")


if __name__ == "__main__":
    main()
