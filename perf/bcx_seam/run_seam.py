#!/usr/bin/env python3
"""BindCraft 2 gradient rounds on the card, with the host's share of CPU as the variable.

The arm is `perf/bcx_round/run_round.py` unchanged: BindCraft 2's `campaign.py` drives, the
Evoformer runs on the card for every call, `perf/bcx_round/meter.py` timestamps the seams.
This adds three things, none of which changes what the campaign computes:

* `--boost W` alternates rounds between the default cpu.weight (100) of this process's
  cgroup and W, set at the round boundary. That is a quiet host as seen from this process,
  interleaved with the loaded one in the same process, on the same card and the same
  compiled program, so the two readings differ in one thing.
* per round: the process's CPU seconds and the run-queue wait summed over its threads
  (`/proc/self/task/*/schedstat`), which is what says whether a boosted round really stopped
  waiting for a CPU.
* `--profile A,B` wraps rounds A..B in `jax.profiler` and dumps the optimised HLO, which is
  what `hostmap.py` attributes per module. XLA CPU records one event per thunk, named by
  its HLO instruction; the dump maps each instruction to its haiku name scope.
"""
import argparse
import functools
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT), str(_ROOT / "perf" / "bcx_afgrad"), str(_ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meter as M                                                      # noqa: E402
import bc2_state as B                                                  # noqa: E402
import bindcraft.af2 as bc2_af2                                        # noqa: E402
import bindcraft.campaign as campaign                                  # noqa: E402
import bindcraft.trajectory as trajectory                              # noqa: E402
import bindcraft.sequence_optimization as seqopt                       # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402
from run_round import MONOMER, git_head                                # noqa: E402


def tids():
    return [int(t) for t in os.listdir("/proc/self/task")]


def sched():
    """(CPU seconds, run-queue wait seconds) summed over every thread alive now."""
    run = wait = 0
    for t in tids():
        try:
            a, b, _ = open(f"/proc/self/task/{t}/schedstat").read().split()
        except OSError:
            continue
        run, wait = run + int(a), wait + int(b)
    return run / 1e9, wait / 1e9


def cgroup_weight_path():
    rel = open("/proc/self/cgroup").read().strip().split("::", 1)[1]
    return f"/sys/fs/cgroup{rel}/cpu.weight"


def set_weight(weight):
    """cpu.weight of this process's own cgroup (its ssh session scope).

    Per-thread nice does NOTHING here, measured: qb2 runs the cgroup-v2 cpu controller under
    user-1000.slice, so CPU is shared between session scopes by weight and a thread's nice
    only orders it against threads of its own scope. The first A/B run reniced every thread to
    -15 and read the same run-queue wait (157.8 against 157.1 thread-seconds a round). The
    scope's weight is the lever that reaches the other workers. init.scope and system.slice
    sit beside user.slice at the root, so no weight here can starve PID 1 or the watchdog.
    """
    return subprocess.run(["sudo", "-n", "tee", cgroup_weight_path()], input=f"{weight}\n",
                          text=True, capture_output=True).returncode


class SeamMeter(M.Meter):
    def __init__(self, rounds, boost, profile, trace_dir):
        super().__init__(rounds)
        self.boost, self.profile, self.trace_dir = boost, profile, trace_dir
        self.tracing = False

    def on_sequence_gradients_enter(self):
        import jax
        r = self.entries + 1
        run, wait = sched()
        M.EVENTS.append({"kind": "sched", "phase": "round", "t0": time.time(), "round": r,
                         "cpu_s": run, "wait_s": wait, "threads": len(tids())})
        if self.profile and self.tracing and r > self.profile[1]:
            jax.profiler.stop_trace()
            self.tracing = False
            M.EVENTS.append({"kind": "trace_stop", "phase": "round", "t0": time.time(),
                             "round": r})
        super().on_sequence_gradients_enter()
        # Round 1 carries the compile and is never a measurement, so boosting starts on an
        # even round: 2, 4, 6 ... boosted, 3, 5, 7 ... at the default.
        weight = self.boost if (self.boost and r % 2 == 0) else 100
        rc = set_weight(weight) if self.boost else None
        M.EVENTS.append({"kind": "nice", "phase": "round", "t0": time.time(), "round": r,
                         "nice": weight, "weight": weight, "rc": rc})
        if self.profile and r == self.profile[0]:
            jax.profiler.start_trace(self.trace_dir)
            self.tracing = True
            M.EVENTS.append({"kind": "trace_start", "phase": "round", "t0": time.time(),
                             "round": r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--boost", type=int, default=0,
                    help="cgroup cpu.weight for even rounds (default scope weight is 100); "
                         "0 leaves every round at the default")
    ap.add_argument("--profile", default="", help="A,B: jax.profiler over rounds A..B")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={project}"]
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))
    campaign.MULTIMER_POOL = MONOMER

    import ttbio_predictor as T
    campaign.AlphaFoldDesignModel = functools.partial(
        T.TTBioAlphaFoldDesignModel, trunk="device")

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()
    profile = tuple(int(x) for x in args.profile.split(",")) if args.profile else ()
    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "commit": git_head(),
             "seed": args.seed, "rounds_requested": args.rounds, "boost": args.boost,
             "profile": profile, "xla_flags": os.environ.get("XLA_FLAGS"),
             "omp": os.environ.get("OMP_NUM_THREADS"),
             "length_bucket_size": campaign_length_bucket(settings),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    import afgrad as _A
    from splice import EvoformerOnDevice, evoformer_on_device
    mt = SeamMeter(args.rounds, args.boost, profile, os.path.join(project, "trace"))
    M.install(mt, sys.modules["splice"], T.TTBioAlphaFoldDesignModel, trajectory, seqopt)

    # BindCraft 2 lowers and compiles its gradient program on EVERY call
    # (`af2.py:405-406`) and relies on JAX's caches to make that cheap. Time it, since it is
    # host time inside the round that is not AF2's arithmetic.
    real_owc = bc2_af2.one_worker_compiles

    import contextlib

    @contextlib.contextmanager
    def owc(shape):
        t0 = time.time()
        with real_owc(shape):
            yield
        M.ev("host", "lower_compile", t0, time.time(), round=mt.entries)
    bc2_af2.one_worker_compiles = owc

    t_load = time.time()
    _dm, _ = _A.load_models(_A.DEFAULT_PARAMS)
    _dev = _A.Dev(_dm.to_device())
    evo = EvoformerOnDevice(_dev, k_evo=48)
    M.ev("setup", "load_models", t_load, time.time())

    t0 = time.time()
    stopped = None
    try:
        with evoformer_on_device(evo):
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                            "proteinmpnn", "weights_neutral"),
                                  max_trajectories=1)
    except M.StopAfterRounds as stop:
        stopped = str(stop)
    finally:
        M.CLOCK.stop()
        if mt.tracing:
            import jax
            jax.profiler.stop_trace()
        if args.boost:
            set_weight(100)
        stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                      "device_calls": dict(evo.calls), "loadavg_end": os.getloadavg(),
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                      "traces": sorted(glob.glob(os.path.join(project, "trace", "**",
                                                              "*.xplane.pb"), recursive=True))})
        out = pathlib.Path(project) / "round_events.json"
        M.dump(str(out), stamp)
        print(json.dumps(stamp, indent=1), flush=True)
        print(f"events -> {out}", flush=True)


if __name__ == "__main__":
    main()
