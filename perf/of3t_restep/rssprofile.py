#!/usr/bin/env python3
"""Where the 19.0 GiB of an exactness-ON OF3T training step actually goes.

R216: `of3t-restep`'s crop-384 step was OOM-killed entering the backward, 19.0 GiB anon-rss
on a 30 GB box. The open question is the DECOMPOSITION -- the retained tape and the exact
softmax's float64 temporary are independent terms and nobody has shown which dominates.
An RSS number without a phase beside it cannot tell them apart, so this samples RSS at 5 Hz
and tags every sample with the phase of `fullstep.py` that is running.

Read it this way:
  * RSS RISES MONOTONICALLY through trunk and diffusion -> that is the retained tape,
    because nothing in the forward frees an activation the backward will need.
  * RSS SAWTOOTHS during the backward -> that is transient temporaries, and the amplitude
    is what a chunked `host_f64_softmax_values` would remove.
  * the FLOOR under the sawtooth is the tape, and it is what chunking will NOT remove.

`fullstep.py` is `of3t-stepfloor`'s and is imported, never edited: the phases come from
wrapping its module-level functions here.

SAFETY. pc has 30 GB and no swap, and the previous run's kernel OOM chose a process by score,
not by who caused it -- a sibling row's pytest suite is a plausible victim. So this watches
system MemAvailable, not just its own RSS, and hard-exits itself the moment MemAvailable
falls under --floor-gib. The JSONL is flushed per sample, so the profile survives that exit.
That makes the peak this reports a LOWER BOUND whenever it breaches, and it says so.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PAGE = os.sysconf("SC_PAGE_SIZE")
GIB = 1024.0 ** 3

_PHASE = ["import"]
_JSONL = None
_STOP = threading.Event()
_FLOOR = 2.0
_SOFTMAX = {"inflight": 0, "calls": 0, "elements": 0}


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE


def _vmhwm_bytes() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return 0


def _mem_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def _sampler(period: float):
    t0 = time.perf_counter()
    while not _STOP.is_set():
        avail = _mem_available_bytes()
        rec = {"t": round(time.perf_counter() - t0, 3), "phase": _PHASE[-1],
               "rss_gib": round(_rss_bytes() / GIB, 4),
               "avail_gib": round(avail / GIB, 4),
               "sm_inflight": _SOFTMAX["inflight"], "sm_calls": _SOFTMAX["calls"]}
        if avail < _FLOOR * GIB:
            rec["BREACH"] = (f"MemAvailable {rec['avail_gib']} GiB under the {_FLOOR} GiB floor. "
                             f"Exiting before the kernel OOM-killer picks a victim by score. "
                             f"Every peak in this profile is a LOWER BOUND.")
            _JSONL.write(json.dumps(rec) + "\n")
            _JSONL.flush()
            os.fsync(_JSONL.fileno())
            os._exit(37)
        _JSONL.write(json.dumps(rec) + "\n")
        _JSONL.flush()
        time.sleep(period)


def _phased(fn, name):
    def wrapper(*a, **k):
        _PHASE.append(name)
        try:
            return fn(*a, **k)
        finally:
            _PHASE.pop()
    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def _counted_softmax(fn):
    def wrapper(v, dim=-1):
        _SOFTMAX["inflight"] += 1
        _SOFTMAX["calls"] += 1
        try:
            n = 1
            for d in tuple(v.shape):
                n *= int(d)
            _SOFTMAX["elements"] += n
        except Exception:
            pass
        try:
            return fn(v, dim)
        finally:
            _SOFTMAX["inflight"] -= 1
    return wrapper


def start(jsonl_path, *, floor_gib=2.0, period=0.2, header=None):
    """Open the JSONL, write its header and start the 5 Hz sampler. Returns the path.

    `of3t-stepqb2` runs TWO arms in one process and so cannot use `main()`, but the guard
    and the phase tags are the instrument, not this row's: forking them would let the two
    rows' profiles drift apart. So the wiring lives in functions and `main()` is one caller.
    """
    global _JSONL, _FLOOR
    _FLOOR = floor_gib
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    _JSONL = open(jsonl_path, "w", buffering=1)
    rec = {"header": True, "argv": sys.argv[1:], "floor_gib": _FLOOR, "period_s": period,
           "page_bytes": PAGE,
           "mem_total_gib": round(os.sysconf("SC_PHYS_PAGES") * PAGE / GIB, 3),
           "avail_at_start_gib": round(_mem_available_bytes() / GIB, 3),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pid": os.getpid()}
    rec.update(header or {})
    _JSONL.write(json.dumps(rec) + "\n")
    threading.Thread(target=_sampler, args=(period,), daemon=True).start()
    return jsonl_path


def finish(rc, wall, ag, period=0.2, **extra):
    """Stop the sampler and write the footer. `vmhwm` is the process high-water mark, which
    is the one peak a self-exit cannot have understated."""
    _STOP.set()
    time.sleep(period * 2)
    rec = {"footer": True, "rc": rc, "wall_s": wall,
           "vmhwm_gib": round(_vmhwm_bytes() / GIB, 4),
           "softmax": dict(_SOFTMAX),
           "exact_training_ops": list(ag.exact_training_ops()),
           "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec.update(extra)
    _JSONL.write(json.dumps(rec) + "\n")
    _JSONL.close()
    return rec


def wire_phases():
    """Tag `fullstep.py`'s phases onto the RSS samples. Returns `(ag, F)`.

    The trunk is called twice per rep with different meaning -- the untaped no_grad prefix
    and the one taped cycle -- and they retain completely different amounts, so one tag for
    both would hide the whole question. The `taped` argument is what tells them apart.
    """
    _PHASE.append("import_engine")
    from tt_bio import autograd as ag
    from perf.of3t_stepfloor import fullstep as F
    _PHASE.pop()

    _real_trunk = F.trunk_forward

    def trunk_tagged(trunk, held, cycles, taped):
        _PHASE.append("trunk_taped" if taped else "trunk_nograd")
        try:
            return _real_trunk(trunk, held, cycles, taped)
        finally:
            _PHASE.pop()

    F.trunk_forward = trunk_tagged
    F.diffusion_train = _phased(F.diffusion_train, "diffusion")
    F.host_losses = _phased(F.host_losses, "losses")
    F.declare_all = _phased(F.declare_all, "declare_weights")
    ag.backward = _phased(ag.backward, "backward")
    ag.host_f64_softmax_values = _counted_softmax(ag.host_f64_softmax_values)

    # `capture()` runs the shipped prep -- MSA resolve, featurizer, input atom encoder -- and
    # then HOLDS the trunk's inputs for the whole step. Whether that is the step's memory or
    # the harness's is the first thing the profile has to separate, so build_fold and the fold
    # itself get their own tags rather than sitting inside one "setup".
    from perf.of3t_perf import step as S
    import tt_baseline as B
    B.build_fold = _phased(B.build_fold, "capture_build_fold")
    S.capture = _phased(S.capture, "capture_prep")

    # The first profile put 6.05 GiB between the end of capture and the first trunk call,
    # inside a span that held three unrelated things. Tag them apart: opening the device,
    # walking 3,152 weight tensors, and building AdamW's moment buffers over 381.3 M
    # elements. Since `of3t-optorder` the moments appear at the first `step()`, not here, so
    # this tag now reads ~0 and the optimizer's memory shows up under `optimizer_step`.
    import tt_bio.tenstorrent as TT
    from tt_bio.train import optim as OPT
    TT.get_device = _phased(TT.get_device, "get_device")
    F.get_device = TT.get_device
    _real_adamw_init = OPT.AdamW.__init__

    def adamw_init(self, *a, **k):
        _PHASE.append("optimizer_init")
        try:
            return _real_adamw_init(self, *a, **k)
        finally:
            _PHASE.pop()

    OPT.AdamW.__init__ = adamw_init
    OPT.AdamW.step = _phased(OPT.AdamW.step, "optimizer_step")
    return ag, F


def phase(name):
    """Push a phase tag for a `with` block, for a caller that owns spans `fullstep` has no
    name for -- an arm boundary, say."""
    class _P:
        def __enter__(self):
            _PHASE.append(name)

        def __exit__(self, *exc):
            _PHASE.pop()
    return _P()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--exact", default="on", choices=("on", "off"))
    ap.add_argument("--floor-gib", type=float, default=2.0)
    ap.add_argument("--period", type=float, default=0.2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    out_dir = REPO / "perf" / "of3t_restep" / "out"
    tag = a.tag or f"{a.tokens}_{a.exact}"
    jsonl = start(out_dir / f"rss_{tag}.jsonl", floor_gib=a.floor_gib, period=a.period,
                  header={"tag": tag})
    step_json = out_dir / f"step_{tag}.json"
    ag, F = wire_phases()

    sys.argv = ["fullstep.py", "--tokens", str(a.tokens), "--cycles", str(a.cycles),
                "--samples", str(a.samples), "--reps", str(a.reps),
                "--out", str(step_json)]
    _PHASE.append("setup")
    t0 = time.perf_counter()
    if a.exact == "on":
        rc = F.main()
    else:
        with ag.exact_training(False):
            rc = F.main()
    wall = round(time.perf_counter() - t0, 3)
    _PHASE.pop()
    foot = finish(rc, wall, ag, period=a.period)
    print(f"[rssprofile] rc={rc} wall={wall}s vmhwm={foot['vmhwm_gib']:.3f} GiB -> {jsonl}",
          flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
