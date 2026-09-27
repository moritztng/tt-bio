#!/usr/bin/env python3
"""Two BindCraft 2 design trajectories in one process, serial or interleaved on one card.

`perf/bcx_round/run_round.py` runs ONE trajectory and times its rounds. This runs two, and
the only difference between the two arms is whether they overlap:

    --interleave 0   trajectory A runs to N rounds, then trajectory B does. The control.
    --interleave 1   both run at once, each on its own thread, sharing the card through
                     `tt_bio.duotraj`'s seam lock.

One process for both arms, so the JAX compile, the weights on card and the box's load are
common-mode and cancel. The number is the amortised `wall / rounds completed by both` over
the window in which both trajectories were running, which is the same quantity
`bcx-gpuref` reports (89.93 s / 125 rounds) and the same UNIT the campaign gates on.

BindCraft 2's own campaign loop drives both: `run_campaign` claims a trajectory number out
of the file-locked campaign progress, so two of them in one project folder take trajectory
1 and trajectory 2 exactly as two worker processes would. Nothing about the design loop is
re-implemented here.
"""
import argparse
import contextlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
sys.path.insert(0, str(_ROOT / "perf" / "bcx_round"))
sys.path.insert(0, str(_ROOT / "perf" / "bcx_predictor"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import meter as M                                                      # noqa: E402
import bc2_state as B                                                  # noqa: E402
from tt_bio import duotraj                                             # noqa: E402
from tt_bio import genq as _genq                                       # noqa: E402
from tt_bio import reblock_permute as _reblock                         # noqa: E402
import bindcraft.campaign as campaign                                  # noqa: E402
import bindcraft.trajectory as trajectory                              # noqa: E402
import bindcraft.sequence_optimization as seqopt                       # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402
from tt_bio import bindcraft2                                          # noqa: E402
import bindcraft.af2 as _bc2_af2                                       # noqa: E402


@contextlib.contextmanager
def _no_cross_worker_compile_lock(compile_shape):
    """BindCraft 2's `one_worker_compiles`, with the lock taken out.

    It exists to stop several WORKER PROCESSES sharing one JAX compilation cache from
    compiling the same shape at once, and it is an `flock` on a per-shape lock file. Two
    trajectories inside ONE process is the case it was not written for, and on this path it
    deadlocks permanently rather than serialising:

      * trajectory A takes the lock and starts compiling;
      * the first device callback imports the tt-bio device stack, and `tt_bio.main`'s
        nanobind stderr filter FORKS (`tt_bio/main.py:62`);
      * the child inherits a duplicate of the locked open file description and outlives the
        `with` block, so closing the parent's fd does NOT release the lock -- measured: a
        same-process re-lock after a plain close succeeds, and the same re-lock with a forked
        duplicate alive blocks;
      * trajectory B asks for the same shape digest -- both trajectories are pinned to the
        same binder length -- and waits forever.

    Removing it changes no arithmetic and no work: it is a cross-process mutex and this is
    one process. Both arms run without it, so the comparison stays like-for-like.
    """
    yield


class DuoMeter:
    """`meter.Meter`, counted per trajectory.

    One process runs two of them, so `entries` cannot be a scalar: it is what every event in
    `meter.install` stamps as its round number, and a shared counter would interleave two
    trajectories' rounds into one numbering that belongs to neither.
    """

    def __init__(self, rounds, trajectories=1):
        self.rounds = rounds
        self.counts: dict[str, int] = {}
        #: Set when the FIRST trajectory clears its compile round, which is when a second one
        #: may start without the two of them measuring each other's compile.
        self.ready = threading.Event()
        #: Every trajectory waits here once its OWN compile round is behind it, so the warm
        #: rounds of both start together. Without it the leader is three or four rounds ahead
        #: by the time the follower finishes tracing, and it then finishes and leaves the card
        #: to the follower alone -- the common window ends up one round wide on a seven-round
        #: arm, which is not a sample.
        self.warm = threading.Barrier(trajectories)

    @property
    def entries(self) -> int:
        return self.counts.get(duotraj.slot(), 0)

    def on_sequence_gradients_enter(self):
        s = duotraj.slot()
        n = self.counts[s] = self.counts.get(s, 0) + 1
        if n > self.rounds:
            M.EVENTS.append({"kind": "round_stop", "phase": "round", "t0": time.time(),
                             "slot": s, "round": n, "reach": M._reach()})
            raise M.StopAfterRounds(f"{self.rounds} rounds collected on {s!r}")
        if n == 2:
            self.ready.set()
            try:
                self.warm.wait(timeout=1800)
            except threading.BrokenBarrierError:
                pass
        M.EVENTS.append({"kind": "round_start", "phase": "round", "t0": time.time(),
                         "slot": s, "round": n, "load1": os.getloadavg()[0],
                         "triatt_bw": M.reach(), "mm_layout": M._mm_reach(),
                         "reach": M._reach()})
        if M.DUMP:
            M.dump(*M.DUMP)


def git_head():
    return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--interleave", type=int, default=0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--binder", type=int, default=146)
    ap.add_argument("--params", default="/home/moritz/bcx_shipped/af2_params")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V")
    args = ap.parse_args()
    if args.rounds > 9:
        raise SystemExit("9 rounds is the ceiling per process, see bcx-p10-rne")

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=2",
                 f"project_folder={project}", f"binder_lengths=[{args.binder},{args.binder}]",
                 # Both trajectories must be the SAME shape, or the interleaved arm pays a
                 # second compile the serial arm does not and the comparison is of compiles.
                 "compile_next_length=0"] + args.sets
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))

    _bc2_af2.one_worker_compiles = _no_cross_worker_compile_lock

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()

    import tt_bio
    os.environ["TT_BIO_TRIATT_TAPED_SDPA"] = "0"
    os.environ["TT_BIO_SDPA_OWN_FORWARD"] = "1"
    from tt_bio import tenstorrent as _tn
    from tt_bio.af2 import AF2PairBlock
    from tt_bio import triatt_bw as _tbw
    AF2PairBlock.rne_kernel = True
    os.environ["TT_BIO_TAPED_KERNELS"] = "tri_att_sdpa_hifi,rne_add"
    os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1"
    _tn._TRIATT_FUSED_HIFI = True
    _tbw.FUSED = True

    _expect = {"genq_compact": os.environ.get("TT_BIO_GENQ_COMPACT") == "1",
               "taped_channel_move": os.environ.get("TT_BIO_TAPED_CHANNEL_MOVE") == "1",
               "mm_layout": os.environ.get("TT_BIO_MM_LAYOUT") == "1",
               "widen_add": os.environ.get("TT_BIO_WIDEN_ADD") == "1",
               "grad_fanin_l1": os.environ.get("TT_BIO_GRAD_FANIN_L1") == "1",
               "triatt_bw": True, "triatt_hifi": True, "rne_kernel": True}

    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "tt_bio_file": tt_bio.__file__, "interleave": bool(args.interleave),
             "levers_expected": _expect, "genq_compact": _genq.compact(),
             "taped_channel_move": _reblock.TAPED_MOVE,
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "commit": git_head(),
             "seed": args.seed, "binder": args.binder, "rounds_requested": args.rounds,
             "length_bucket_size": campaign_length_bucket(settings),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(),
             "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    from tt_bio import rne_add as _rne
    from tt_bio import taped_ttnn

    def _host_rss():
        """This process's resident and high-water host memory, at every round boundary.

        Leg 1 measured the CARD and the card is not the binding constraint: the first
        nine-round interleaved arm was OOM-killed by the HOST at 13.2 GB anon-rss on a 31 GB
        box with ~14 GB available, while device DRAM never went past 3.75 GB of 31.9. Two
        trajectories means two JAX programs and two sets of host activations, and that is what
        runs out first.
        """
        out = {}
        try:
            for line in open("/proc/self/status"):
                if line.startswith(("VmRSS:", "VmHWM:")):
                    out[line.split(":")[0].lower()] = int(line.split()[1]) * 1024
        except OSError:
            pass
        try:
            out["mem_available"] = next(
                int(l.split()[1]) * 1024 for l in open("/proc/meminfo")
                if l.startswith("MemAvailable:"))
        except (OSError, StopIteration):
            pass
        return out
    M.REACH.append(_host_rss)
    M.REACH.append(lambda: {
        "rne_add_served": _rne.STATS[0], "rne_add_declined": _rne.STATS[1],
        "rne_add_entry": list(taped_ttnn.KERNEL_STATS.get("rne_add", [0, 0])),
    })
    _check = M.lever_reach(_expect)
    _check()
    M.REACH.append(_check)

    # The gate's own numbers, refreshed into the stamp at every boundary. The end-of-run stamp
    # is written once and three interleaved arms have now been OOM-killed before reaching it;
    # the per-boundary dump is what actually survives.
    def _gate_now():
        g = duotraj.GATE
        if g is not None:
            stamp["gate"] = g.report()
        return {}
    M.REACH.append(_gate_now)

    out = pathlib.Path(project) / "round_events.json"
    M.DUMP = (str(out), stamp)
    mt = DuoMeter(args.rounds, 2 if args.interleave else 1)
    M.install(mt, bindcraft2, bindcraft2.design_model_class(), trajectory, seqopt)
    _digest_outputs(bindcraft2.design_model_class(), mt)

    # Every event carries the trajectory that produced it, or the device seconds of two
    # trajectories land in one column and the amortised number is unreadable.
    _ev = M.ev

    def ev(kind, phase, t0, t1, **kw):
        # A "device" event fires at the seam, on one of XLA:CPU's own pool threads, where
        # nothing on the thread says whose work it is -- so it gets no slot rather than a
        # wrong one. The device column per trajectory is the gate's own `held_s`.
        _ev(kind, phase, t0, t1,
            slot=(None if kind == "device" else duotraj.slot()), **kw)
    M.ev = ev

    def one(name):
        def go():
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                            "proteinmpnn",
                                                            "weights_neutral"),
                                  max_trajectories=2)
        return go

    t0 = time.time()
    stopped = []
    evo = None
    try:
        with bindcraft2.campaign_predictor(trunk="device", validation="device",
                                           checkpoints=args.params, extra_msa=True,
                                           template=True, exact=False) as build:
            evo = build.evoformer
            # The gate goes on BOTH arms. On the serial arm it is one thread taking an
            # uncontended lock, which costs nothing and is what makes the control able to say
            # which thread the device seam runs on -- the one fact the slot design rests on.
            with duotraj.interleave(trajectories=2 if args.interleave else 1) as gate:
                _run_pair(one, stopped, threaded=bool(args.interleave),
                          ready=mt.ready if args.interleave else None)
            stamp["gate"] = gate.report()
    finally:
        M.CLOCK.stop()
        from tt_bio import autograd, mm_layout, rne_add, tenstorrent
        stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                      "triatt_bw_stats": dict(_tbw.STATS),
                      "device_calls": dict(evo.calls) if evo else None,
                      "kernel_entry_stats": {k: list(v) for k, v in
                                             taped_ttnn.KERNEL_STATS.items()},
                      "rne_add_stats": {"served": rne_add.STATS[0],
                                        "declined": rne_add.STATS[1]},
                      "mm_layout": mm_layout.reach(),
                      "exact_softmax_stats": dict(autograd.EXACT_SOFTMAX_STATS),
                      "host_folds": dict(evo.host_folds) if evo else None,
                      "loadavg_end": os.getloadavg(),
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        M.dump(str(out), stamp)
        print(json.dumps(stamp, indent=1), flush=True)
        print(f"events -> {out}", flush=True)


def _digest_outputs(cls, mt):
    """A sha256 of every gradient round's output, per trajectory, on both arms.

    Interleaving changes no arithmetic, so each trajectory's outputs in `duo` must be
    `torch.equal` to the same trajectory run `serial`. Hashing the return of
    `sequence_gradients` (gradients, losses, aux) lets the two arms be compared round by round
    from their event logs. Both arms pay it, so it is common-mode in the timing.
    """
    import dataclasses
    import hashlib
    import numpy as np
    sg = cls.sequence_gradients

    # `StructurePrediction` and `Protein` are plain dataclasses, not registered pytrees, so
    # `jax.tree_util.tree_leaves` hands them back whole and `np.asarray` makes an object array
    # whose bytes are pointers. Walk them by field instead, with dict keys in sorted order.
    def walk(h, x):
        if dataclasses.is_dataclass(x):
            for f in dataclasses.fields(x):
                h.update(f.name.encode())
                walk(h, getattr(x, f.name))
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                walk(h, x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(h, v)
        else:
            a = np.asarray(x)
            if a.dtype == object:
                raise TypeError(f"digest: unwalkable leaf {type(x).__name__}")
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())

    def sha(x):
        h = hashlib.sha256()
        walk(h, x)
        return h.hexdigest()

    def sequence_gradients(self, *a, **kw):
        r = sg(self, *a, **kw)
        predictions, gradients, loss = r
        parts = {"pred": sha(predictions), "grad": sha(gradients), "loss": sha(loss)}
        M.EVENTS.append({"kind": "digest", "phase": "sequence_gradients", "t0": time.time(),
                         "slot": duotraj.slot(), "round": mt.entries, "parts": parts,
                         "sha256": hashlib.sha256("".join(parts.values()).encode()).hexdigest()})
        return r
    cls.sequence_gradients = sequence_gradients


def _run_pair(one, stopped, *, threaded, ready):
    """Both trajectories, either on two threads or one after the other.

    A `StopAfterRounds` is this harness finishing a trajectory on purpose, so it is recorded
    and not re-raised; anything else is a real failure and comes back out.
    """
    names = ["A", "B"]
    if threaded:
        def guard(name):
            def go():
                try:
                    one(name)()
                except M.StopAfterRounds as stop:
                    stopped.append(f"{name}: {stop}")
            return go
        duotraj.run([guard(n) for n in names], names=names, ready=ready)
        return
    for name in names:
        with duotraj.trajectory(name):
            try:
                one(name)()
            except M.StopAfterRounds as stop:
                stopped.append(f"{name}: {stop}")


if __name__ == "__main__":
    main()
