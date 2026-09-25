#!/usr/bin/env python3
"""BindCraft 2 gradient rounds on the card, with the host's share of CPU as the variable.

The arm is `perf/bcx_round/run_round.py` unchanged: BindCraft 2's `campaign.py` drives, the
Evoformer runs on the card for every call, `perf/bcx_round/meter.py` timestamps the seams.
This adds three things, none of which changes what the campaign computes:

* `--cap N` confines this process to N logical CPUs on even rounds and releases it on odd
  ones, in the same process, on the same card and the same compiled program, so the two
  readings differ in one thing: how much CPU the host side is allowed.
* per round: the process's CPU seconds and the run-queue wait summed over its threads
  (`/proc/self/task/*/schedstat`), which is what says how much of a round
  was spent waiting for a CPU rather than on one.
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


ALL_CPUS = sorted(os.sched_getaffinity(0))


def set_cpus(n):
    """Confine every thread of THIS process to its first n logical CPUs (0 = all of them).

    This only ever narrows what this process may use; it takes nothing from any other
    process on the box. qb2 is shared with other rows, so the row does not raise its own
    CPU share to imitate a quiet host: an earlier arm that did (cgroup cpu.weight) was
    stopped and its rounds are not used. Returns how many threads were moved.
    """
    cpus = set(ALL_CPUS[:n] if n else ALL_CPUS)
    moved = 0
    for t in tids():
        try:
            os.sched_setaffinity(t, cpus)
            moved += 1
        except OSError:
            pass
    return moved


class SeamMeter(M.Meter):
    def __init__(self, rounds, cap, profile, trace_dir):
        super().__init__(rounds)
        self.cap, self.profile, self.trace_dir = cap, profile, trace_dir
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
        # Even rounds capped, odd rounds on every CPU; the compile round is dropped anyway.
        cap = self.cap if (self.cap and r % 2 == 0) else 0
        moved = set_cpus(cap) if self.cap else None
        M.EVENTS.append({"kind": "cpus", "phase": "round", "t0": time.time(), "round": r,
                         "cpus": cap or len(ALL_CPUS), "threads_moved": moved})
        if self.profile and r == self.profile[0]:
            jax.profiler.start_trace(self.trace_dir)
            self.tracing = True
            M.EVENTS.append({"kind": "trace_start", "phase": "round", "t0": time.time(),
                             "round": r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--cap", type=int, default=0,
                    help="logical CPUs this process may use on even rounds; 0 = no cap")
    ap.add_argument("--profile", default="", help="A,B: jax.profiler over rounds A..B")
    ap.add_argument("--extra-msa", action="store_true",
                    help="run the extra-MSA pair stack on the card too (bcx-extramsa's "
                         "swap, GO at 2.156x). Every other module's host share is a "
                         "different quantity with it on, so a row aimed at one of them "
                         "measures it here.")
    ap.add_argument("--template", action="store_true",
                    help="run the template pair stack on the card too (forward only)")
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
             "seed": args.seed, "rounds_requested": args.rounds, "cap": args.cap,
             "profile": profile, "xla_flags": os.environ.get("XLA_FLAGS"),
             "omp": os.environ.get("OMP_NUM_THREADS"),
             "length_bucket_size": campaign_length_bucket(settings),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    import afgrad as _A
    from splice import (EvoformerOnDevice, ExtraMsaOnDevice, TemplatePairStackOnDevice,
                        evoformer_on_device)
    mt = SeamMeter(args.rounds, args.cap, profile, os.path.join(project, "trace"))
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
    _dm, _ = _A.load_models(_A.DEFAULT_PARAMS, template=args.template)
    _dev = _A.Dev(_dm.to_device())
    evo = EvoformerOnDevice(_dev, k_evo=48)
    extra = ExtraMsaOnDevice(_dev, k_extra=4) if args.extra_msa else None
    tmpl = TemplatePairStackOnDevice(_dev, k_template=2) if args.template else None
    M.ev("setup", "load_models", t_load, time.time())

    t0 = time.time()
    stopped = None
    try:
        with evoformer_on_device(evo, extra_msa=extra, template=tmpl):
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
        if args.cap:
            set_cpus(0)
        stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                      "device_calls": dict(evo.calls),
                      "extra_msa_on": bool(args.extra_msa),
                      "extra_calls": dict(extra.calls) if extra else None,
                      "extra_swapped": extra.swapped if extra else 0,
                      "template_on": bool(args.template),
                      "template_calls": dict(tmpl.calls) if tmpl else None,
                      "template_swapped": tmpl.swapped if tmpl else 0,
                      "template_shapes": sorted(map(str, tmpl.shapes)) if tmpl else None,
                      "template_dropout_seen": sorted(tmpl.dropout_seen) if tmpl else None, "loadavg_end": os.getloadavg(),
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                      "traces": sorted(glob.glob(os.path.join(project, "trace", "**",
                                                              "*.xplane.pb"), recursive=True))})
        out = pathlib.Path(project) / "round_events.json"
        M.dump(str(out), stamp)
        print(json.dumps(stamp, indent=1), flush=True)
        print(f"events -> {out}", flush=True)


if __name__ == "__main__":
    main()
