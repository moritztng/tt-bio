"""One warm fold worker per chip.

    TT_VISIBLE_DEVICES=0 python3 demo/sc26/engine/chipworker.py --chip 0

Reads jobs from stdin, one JSON object per line ({"id", "sequence", "seed"?, "steps"?}), and
writes protocol events (demo/sc26/PROTOCOL.md) to stdout, one JSON object per line. Every
library print is moved to stderr at startup, so stdout carries events and nothing else.

The frames are the diffusion sampler's own coordinates, taken through the default-off hook in
tt_bio.esmfold2 (set_trajectory_dump). Nothing is interpolated. The last frame's coordinates are
the scored structure's, bit for bit.

SIGUSR1 drops the fold in progress and keeps the worker warm (a visitor preempting an attract
fold). SIGINT drops it and exits cleanly, which leaves the chip usable. SIGKILL does not, so the
supervisor never sends it.
"""
import argparse
import base64
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True, help="UMD chip id this worker is pinned to")
    ap.add_argument("--model", default="esmfold2", choices=["esmfold2"])
    ap.add_argument("--steps", type=int, default=20, help="diffusion steps (the model default)")
    ap.add_argument("--loops", type=int, default=3, help="trunk recycles (the model default)")
    args = ap.parse_args()

    abort, cancel, busy = threading.Event(), threading.Event(), threading.Event()
    import signal

    def on_stop(*_):
        abort.set()
        if not busy.is_set():  # idle, blocked on stdin: leave now; mid-fold, at the next step
            raise KeyboardInterrupt

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
    from tt_bio import esmfold2 as E
    from tt_bio.esmfold2_runtime import build_spi, load_ttnn_esmfold2
    from tt_bio.tenstorrent import get_device
    from tt_bio._vendor.esm.models.esmfold2 import ESMFold2InputBuilder
    from tt_bio._vendor.esm.models.esmfold2.output import get_element_symbol
    from tt_bio._vendor.esm.models.esmfold2.processor import _seed_context

    get_device()
    model = load_ttnn_esmfold2()
    model._esmc.preload()
    builder = ESMFold2InputBuilder()
    install_handlers()

    def fold(job):
        jid, seq = job["id"], job["sequence"].strip().upper()
        seed, steps = int(job.get("seed", 0)), int(job.get("steps", args.steps))
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
        t_start = time.perf_counter()
        phase = {"stage": "lm", "t": t_start}
        stamps = {}

        def progress(stage, step=0, total=0):
            if abort.is_set() or cancel.is_set():
                raise Aborted()
            if stage == "diffusion":
                phase["of"] = total  # the sampler's real step count (its schedule is clipped)
                if pending:  # the noise frame waited for that count
                    emit(**{**pending.pop(), "of": total})
            if stage != phase["stage"]:
                stamps[phase["stage"]] = time.perf_counter() - phase["t"]
                phase.update(stage=stage, t=time.perf_counter())
            emit(type="stage", id=jid, chip=args.chip, stage=stage, step=step, total=total,
                 t=round(time.perf_counter() - t_start, 3))

        prev = {"display": None}
        pending = []

        def stop_check(step, x, x_den):
            if abort.is_set() or cancel.is_set():
                raise Aborted()

        def dump(step, x, x_den):
            if abort.is_set() or cancel.is_set():
                raise Aborted()
            raw = x[0][mask]
            ref = x_den[0][mask] if x_den is not None else raw
            # Display alignment: each frame is rotated onto the one before it, so the sampler's
            # per-step random rotation does not spin the picture. The raw coordinates are sent
            # untouched; R and t are a camera, not an edit.
            if prev["display"] is None:
                r, t = torch.eye(3), torch.zeros(3)
            else:
                r, t = kabsch(ref, prev["display"])
            prev["display"] = ref @ r.T + t
            frame = dict(type="frame", id=jid, chip=args.chip, step=step, of=phase.get("of"),
                         t=round(time.perf_counter() - t_start, 3), xyz=f32(raw),
                         x0=f32(x_den[0][mask]) if x_den is not None else None,
                         R=[round(v, 6) for v in r.flatten().tolist()], T=[round(v, 4) for v in t.tolist()])
            if step == -1:
                pending.append(frame)
            else:
                emit(**frame)

        emit(type="fold_start", id=jid, chip=args.chip, model=args.model, sequence=seq,
             n_res=len(seq), n_atoms=n_atoms, steps=steps, loops=args.loops, seed=seed, atoms=atoms,
             rg_expected=round(2.2 * len(seq) ** 0.38, 2), source="live")
        E.set_progress(progress)
        E.set_trajectory_dump(dump if job.get("frames", True) else stop_check)
        try:
            with clock, torch.no_grad(), _seed_context(seed):
                out = model(**features, num_loops=args.loops, num_sampling_steps=steps,
                            num_diffusion_samples=1, early_exit=False)
            res = builder.decode(out, features, chain_infos, num_diffusion_samples=1)
        finally:
            E.set_progress(None)
            E.set_trajectory_dump(None)
        stamps[phase["stage"]] = time.perf_counter() - phase["t"]
        total = time.perf_counter() - t_start
        final = out["sample_atom_coords"][0][mask]
        emit(type="fold_done", id=jid, chip=args.chip, model=args.model, n_res=len(seq),
             seconds=round(total, 3), stages={k: round(v, 3) for k, v in stamps.items()},
             aiclk_mhz=clock.summary(), xyz=f32(final),
             plddt=[round(float(v), 4) for v in res.plddt.flatten().tolist()],
             ptm=res.ptm, source="live")
        return res

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
