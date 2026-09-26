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


class DuoMeter:
    """`meter.Meter`, counted per trajectory.

    One process runs two of them, so `entries` cannot be a scalar: it is what every event in
    `meter.install` stamps as its round number, and a shared counter would interleave two
    trajectories' rounds into one numbering that belongs to neither.
    """

    def __init__(self, rounds):
        self.rounds = rounds
        self.counts: dict[str, int] = {}
        #: Set when the FIRST trajectory clears its compile round, which is when a second one
        #: may start without the two of them measuring each other's compile.
        self.ready = threading.Event()

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
        if n >= 2:
            self.ready.set()
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
    M.REACH.append(lambda: {
        "rne_add_served": _rne.STATS[0], "rne_add_declined": _rne.STATS[1],
        "rne_add_entry": list(taped_ttnn.KERNEL_STATS.get("rne_add", [0, 0])),
    })
    _check = M.lever_reach(_expect)
    _check()
    M.REACH.append(_check)

    out = pathlib.Path(project) / "round_events.json"
    M.DUMP = (str(out), stamp)
    mt = DuoMeter(args.rounds)
    M.install(mt, bindcraft2, bindcraft2.design_model_class(), trajectory, seqopt)

    # Every event carries the trajectory that produced it, or the device seconds of two
    # trajectories land in one column and the amortised number is unreadable.
    _ev = M.ev

    def ev(kind, phase, t0, t1, **kw):
        _ev(kind, phase, t0, t1, slot=duotraj.slot(), **kw)
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
            gate = None
            if args.interleave:
                with duotraj.interleave(bindcraft2.EvoformerOnDevice,
                                        bindcraft2.ExtraMsaOnDevice,
                                        bindcraft2.TemplateOnDevice,
                                        trajectories=2) as gate:
                    _run_pair(one, stopped, threaded=True, ready=mt.ready)
            else:
                _run_pair(one, stopped, threaded=False, ready=None)
            stamp["gate"] = gate.report() if gate is not None else None
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
