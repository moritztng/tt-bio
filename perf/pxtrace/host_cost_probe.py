#!/usr/bin/env python3
"""Price what a trunk-level ttnn trace could possibly remove, before building one.

Trace replay removes HOST work: the Python in the trunk loop and the per-op enqueue path.
It removes no device work. So the whole lever is bounded above by the host CPU seconds the
trunk burns, and the part that actually shows up in the wall is bounded by how much of that
host time sits on the critical path.

Two things are measured in one fold:

  H  -- host CPU seconds consumed inside Trunk.__call__, reported per thread (the Python
        caller, which is where the enqueue path runs) and per process (which also counts
        tt-metal's spinning dispatch threads, so it OVERstates removable work).
  K  -- ttnn calls issued inside the trunk, per op name.

--inject-us D additionally spins D microseconds of host time inside every wrapped ttnn call.
Sweeping D turns the trunk wall into a line whose slope is dWall/d(host time): slope near 0
means the host already runs ahead of the device and removing host time buys nothing; slope
near K means every host microsecond is on the critical path. The numerics are untouched by
construction, so every arm must produce the same CIF sha256.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

OPS = """linear matmul add subtract multiply mul div divide layer_norm rms_norm softmax
sigmoid silu gelu relu reshape permute transpose to_layout tilize untilize slice concat pad
embedding clone copy typecast deallocate to_memory_config generic_op sum mean max min sqrt
rsqrt exp log where eq gt lt neg abs reciprocal sharded_to_interleaved
interleaved_to_sharded reallocate from_torch to_torch repeat repeat_interleave unsqueeze
squeeze""".split()

COUNTS = {}
_SPIN = [0.0]


def _wrap(fn, name):
    pc = time.perf_counter

    def w(*a, **k):
        COUNTS[name] = COUNTS.get(name, 0) + 1
        d = _SPIN[0]
        if d:
            t = pc() + d
            while pc() < t:
                pass
        return fn(*a, **k)

    w.__name__ = getattr(fn, "__name__", name)
    return w


def install(inject_us):
    import ttnn
    _SPIN[0] = inject_us / 1e6
    seen = []
    for n in dict.fromkeys(OPS):
        f = getattr(ttnn, n, None)
        if callable(f):
            setattr(ttnn, n, _wrap(f, n))
            seen.append(n)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--msa-a3m", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--inject-us", type=float, default=0.0)
    ap.add_argument("--inject-sweep", default=None,
                    help="comma-separated per-warm-fold inject_us, e.g. 0,25,200,200,25,0. "
                         "Runs every arm in ONE process, interleaved, so the arms share a "
                         "model load and a warm program cache. Overrides --inject-us; "
                         "--repeat is set from its length. The cold fold always runs at 0.")
    ap.add_argument("--keep-cif", type=Path, default=None)
    ap.add_argument("--label", default="512 aa")
    a = ap.parse_args()

    sweep = [float(x) for x in a.inject_sweep.split(",")] if a.inject_sweep else None
    if sweep:
        a.repeat = len(sweep)
    wrapped = install(0.0)          # armed per trunk call below, never during diffusion
    import tt_bio.protenix as P
    tid = time.CLOCK_THREAD_CPUTIME_ID
    pid = time.CLOCK_PROCESS_CPUTIME_ID
    rows = []
    orig = P.Trunk.__call__

    # The injected delay is armed ONLY inside Trunk.__call__. Diffusion and confidence issue
    # ttnn ops through the same wrappers, and spinning there would inflate the fold wall
    # without telling us anything about the region a trunk trace would cover.
    seq = [0.0] + (sweep if sweep else [a.inject_us] * max(a.repeat, 1))

    def timed(self, *args, **kw):
        COUNTS.clear()
        d = seq[len(rows)] if len(rows) < len(seq) else 0.0
        _SPIN[0] = d / 1e6
        w0 = time.perf_counter()
        t0 = time.clock_gettime(tid)
        p0 = time.clock_gettime(pid)
        try:
            return orig(self, *args, **kw)
        finally:
            _SPIN[0] = 0.0
            row = {"inject_us": d,
                   "wall": time.perf_counter() - w0,
                   "thread_cpu": time.clock_gettime(tid) - t0,
                   "proc_cpu": time.clock_gettime(pid) - p0,
                   "calls": dict(sorted(COUNTS.items(), key=lambda kv: -kv[1]))}
            row["n_calls"] = sum(row["calls"].values())
            rows.append(row)

    P.Trunk.__call__ = timed

    import tt_baseline as TB
    TB.measure("protenix-v2", a.repeat, Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser(),
               a.out, a.target, a.msa_a3m, a.label, fast=False, keep_cif=a.keep_cif,
               trace=False)

    d = json.loads(a.out.read_text())
    d["host_cost_probe"] = {"inject_us": a.inject_us, "inject_sweep": sweep,
                            "wrapped_ops": wrapped, "trunk": rows}
    a.out.write_text(json.dumps(d, indent=2))
    for i, r in enumerate(rows):
        print("trunk[%d] D=%6.1fus wall=%.3fs thread_cpu=%.3fs proc_cpu=%.3fs calls=%d"
              % (i, r["inject_us"], r["wall"], r["thread_cpu"], r["proc_cpu"], r["n_calls"]))
    print("top ops:", list(rows[-1]["calls"].items())[:12] if rows else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
