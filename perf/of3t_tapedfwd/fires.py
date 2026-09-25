#!/usr/bin/env python3
"""Which fused forwards fire in a TAPED OpenFold3 forward, counted at runtime.

The suspicion this row was given: "one taped call switched fused triangle attention off for
the whole process". That defect is real and was fixed (69a0a6bdc). What a code read misses is
the LARGER thing the same grep exposes -- `ops.taping()` is not asked by eleven fused kernels,
it is asked by every tape-gated route in the engine, including L1 residency and eltwise
fusion that have nothing to do with `generic_op`. So this harness does not count kernels. It
counts CALL SITES, by replacing `tt_bio.ops.taping` with a function that records its caller's
frame and returns whatever the arm wants.

That replacement also gives the clean counterfactual. Three arms, same captured inputs, same
cycle count, interleaved so drift lands on all three:

  A  untaped, `taping()` -> False   the inference forward, every lever live
  B  untaped, `taping()` -> True    EXACTLY the route set a taped forward takes, with none of
                                    the tape's own recording cost. A minus B is the seconds
                                    the tape-gated levers cost, with no confound
  C  taped                          the real thing. C minus B is the tape's own overhead

A/B is the number this row owes. Running B untaped is what separates "the levers are off" from
"recording a tape is slow", which a taped-vs-untaped A/B cannot do at all.

    fires.py --tokens 384 --cycles 1 --reps 3 --out out/fires_384.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_perf import step as S                                  # noqa: E402

#: Every `[served, declined]` (or stats-dict) counter a fused forward keeps, as
#: `module.NAME`. Taken from `scripts/lever_census.py::LEVERS` rather than retyped, plus the
#: three counters that table has no row for. The census script runs the CLI as a subprocess
#: and sums per-process dumps; this row needs the counters IN this process, between two
#: forwards, so the table is imported and the reading is done here.
EXTRA_COUNTERS = [
    "tt_bio.triatt_qkv.QKVG_STATS", "tt_bio.triatt_qkv.QKVGB_STATS",
    "tt_bio.swiglu_fused.STATS", "tt_bio.page_copy.STATS",
    "tt_bio.softmax_generic.SSTATS", "tt_bio.tenstorrent.TRIMUL_TAIL_L1_STATS",
    "tt_bio.tenstorrent.PWA_DEPTH_STATS", "tt_bio.tenstorrent.OPM_ROW_STATS",
    # `tape()` installs a HOST float64 softmax and layer norm for the whole taped forward
    # (`autograd._EXACT_OPS`, opened by `taped_ttnn.tape` through `_training_exact`). That is
    # an accuracy decision, not a route, and it is invisible to every counter above -- which
    # is exactly how a 248 s arm C can look like "the tape is slow".
    "tt_bio.autograd.EXACT_SOFTMAX_STATS", "tt_bio.tenstorrent.HOST_F64_SOFTMAX_STATS",
    "tt_bio.autograd.EXACT_LAYER_NORM_STATS",
    # Every refusal LATCH, read as a set length. These are the caches the recorded suspicion
    # is about: if a taped call still poisons one, the arm that follows it serves less. Read
    # by `read_counters` as `len(set)`, so a latch that grows during arm B or C is visible
    # without knowing what it holds.
    "tt_bio.tenstorrent._TRIATT_HIFI_OVER_L1", "tt_bio.tenstorrent._SDPA_QK_OVER_L1",
    "tt_bio.tenstorrent._SDPA_Q_CHUNK_OVER_L1", "tt_bio.tenstorrent._L1_OUT_REFUSED",
    "tt_bio.tenstorrent._TRANSPOSE_L1_REFUSED", "tt_bio.tenstorrent._BMM_CFG_REFUSED",
    "tt_bio.triatt_sdpa._PM_OVER_L1", "tt_bio.triatt_sdpa._GATE_OVER_L1",
]


def counter_attrs():
    sys.path.insert(0, str(REPO / "scripts"))
    import lever_census as LC
    attrs = {}
    for flag, _mod, _a, spec, _how in LC.LEVERS:
        if not spec:
            continue
        attrs.setdefault(spec.split(":")[0], []).append(flag)
    for a in EXTRA_COUNTERS:
        attrs.setdefault(a, []).append("(no census row)")
    return attrs


def read_counters(attrs):
    """Snapshot every counter by name. Lists and dicts of ints only."""
    out = {}
    for attr in attrs:
        mod, _, name = attr.rpartition(".")
        m = sys.modules.get(mod)
        v = getattr(m, name, None) if m is not None else None
        if isinstance(v, list):
            out[attr] = list(v)
        elif isinstance(v, dict) and all(isinstance(x, int) for x in v.values()):
            out[attr] = dict(v)
        elif isinstance(v, (set, frozenset)):
            out[attr] = len(v)
    return out


def delta(before, after):
    d = {}
    for k, a in after.items():
        b = before.get(k)
        if isinstance(a, list):
            v = [x - y for x, y in zip(a, b or [0] * len(a))]
        elif isinstance(a, dict):
            v = {kk: a[kk] - (b or {}).get(kk, 0) for kk in a}
            v = {kk: vv for kk, vv in v.items() if vv}
        else:
            v = a - (b or 0)
        if (isinstance(v, list) and any(v)) or (isinstance(v, dict) and v) or \
           (isinstance(v, int) and v):
            d[k] = v
    return d


class TapingProbe:
    """`tt_bio.ops.taping` replaced by a counting stub.

    Every gate in the engine reaches this through the MODULE (`ops.taping()`), never as a
    rebound local, so one substitution covers all of them -- verified by grep across
    `tt_bio/*.py` before it was written. The caller's frame is what gets recorded: a gate that
    asks is a route the tape decides, whether or not that route keeps a served/declined
    counter, and most of them do not.
    """

    def __init__(self):
        import tt_bio.ops as ops
        self.ops = ops
        self.real = ops.taping
        self.answer = None          # None = defer to the real function
        self.sites = Counter()

    def install(self):
        def stub():
            f = sys._getframe(1)
            name = f.f_code.co_filename
            name = name.split("/tt_bio/")[-1] if "/tt_bio/" in name else name
            self.sites[f"{name}:{f.f_lineno}"] += 1
            return self.real() if self.answer is None else self.answer
        self.ops.taping = stub
        return self

    def restore(self):
        self.ops.taping = self.real

    def take(self):
        s = dict(self.sites)
        self.sites.clear()
        return s


def digest(t):
    """A cheap, order-stable fingerprint of a device tensor, for the A/B equivalence note."""
    import numpy as np
    import ttnn
    from tt_bio import autograd as ag
    raw = ag._unwrap(t) if isinstance(t, ag.Tensor) else t
    a = ttnn.to_torch(raw).float().numpy().astype(np.float64).ravel()
    return {"n": int(a.size), "mean": float(a.mean()), "absmax": float(np.abs(a).max()),
            "l2": float(np.sqrt((a * a).sum()))}


def run_arm(trunk, held, cycles, arm, probe, attrs, out):
    """One forward. Returns (seconds, counter delta, taping sites, digests, exit cost)."""
    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    probe.answer = {"A": False, "B": True, "C": None}[arm]
    probe.take()
    before = read_counters(attrs)
    # The context is opened HERE rather than by `S.cycle_once`, so the clock can stop while
    # the tape is still open. `cycle_once` exits its own context before returning, and exiting
    # a tape frees every retained activation -- at 128 tokens that teardown read as 9.1 s of a
    # 9.5 s "forward", which is not a forward cost at all. A training step keeps the tape open
    # from the forward into the backward, so the teardown belongs to neither.
    from tt_bio import autograd as ag
    snap_args, snap_kwargs = held["trunk_snap"]
    args = S._rehydrate(snap_args, dev)
    kwargs = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items() if k != "progress_fn"}
    trunk.num_cycles = cycles
    ttnn.synchronize_device(dev)
    with (ag.tape() if arm == "C" else ag.no_grad()):
        t0 = time.perf_counter()
        r = trunk(*args, **kwargs)
        ttnn.synchronize_device(dev)
        secs = time.perf_counter() - t0
        dig = [digest(x) for x in r] if isinstance(r, tuple) else [digest(r)]
        d = delta(before, read_counters(attrs))
        sites = probe.take()
        t1 = time.perf_counter()
    teardown = time.perf_counter() - t1
    probe.answer = None
    del r
    return secs, d, sites, dig, teardown


def declare_trunk(trunk, out):
    """Register the trunk's device weights as tape leaves, the way a training step does.

    THE ONE STRUCTURAL DIFFERENCE between this harness and `of3t-stepfloor`'s `fullstep.py`,
    which measured the 3.415 s taped cycle this row prices against. That harness calls
    `declare_all` before its taped forward, so 2531 weights are `ag.parameter()` leaves; this
    one declared none, and its taped arm read 257 s for what should be the same work. 75x is
    not a card difference, so the difference is in the harnesses and this flag is what tells
    the two apart. Same walk (`step.walk_weights`), deduped by tensor IDENTITY for the reason
    `fullstep.py` gives: the tape keys a leaf on the raw handle, so one tensor reachable by
    two paths would be declared twice.
    """
    from tt_bio import autograd as ag
    found, stats = S.walk_weights(trunk, prefix="trunk.")
    by_id, n = set(), 0
    for name, (owner, key, t) in sorted(found.items()):
        if id(t) in by_id:
            continue
        by_id.add(id(t))
        ag.parameter(t)
        n += 1
    out["declared"] = {"weights": n, "found": len(found),
                       "walk_depth": stats["max_depth"], "truncated": stats["truncated"]}
    print(f"[declare] {n} trunk weights registered as tape leaves", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default="ABC")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile the LAST arm and report the top cumulative callees. One "
                         "run names the cost instead of binary-searching hypotheses.")
    ap.add_argument("--declare", action="store_true",
                    help="register the trunk's weights as tape leaves before the arms, the "
                         "way fullstep.py does. The arm-C discrepancy hangs on this.")
    ap.add_argument("--out", type=Path, default=Path("perf/of3t_tapedfwd/out/fires.json"))
    a = ap.parse_args()

    out = {"doc": "Which tape-gated routes a taped OF3 forward gives up, counted at runtime.",
           "argv": sys.argv[1:], "arms": {
               "A": "untaped, taping()->False -- the inference forward",
               "B": "untaped, taping()->True  -- the taped forward's route set, no tape cost",
               "C": "taped -- the real thing"}}
    out["env"] = {"host": socket.gethostname(),
                  "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
                  "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                           capture_output=True, text=True).stdout.strip(),
                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "loadavg": os.getloadavg()}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    clk = during(period=2.0)
    try:
        with clk:
            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            attrs = counter_attrs()
            out["counters_watched"] = len(attrs)
            out["declared"] = None
            if a.declare:
                declare_trunk(trunk, out)
            probe = TapingProbe().install()

            # Burn JIT off on every arm before any arm is timed. A first-call compile is
            # 3-9 s at this size and it lands on whichever arm runs first, which would read
            # as that arm's cost.
            for arm in a.arms:
                t0 = time.perf_counter()
                run_arm(trunk, held, a.cycles, arm, probe, attrs, out)
                print(f"[warm {arm}] {time.perf_counter() - t0:.3f}s", flush=True)

            if a.profile:
                import cProfile
                import pstats
                arm = a.arms[-1]
                pr = cProfile.Profile()
                pr.enable()
                run_arm(trunk, held, a.cycles, arm, probe, attrs, out)
                pr.disable()
                st = pstats.Stats(pr).sort_stats("cumulative")
                rows = []
                for fn, (cc, nc, tt_, ct, _) in st.stats.items():
                    rows.append({"fn": f"{fn[0].split('/')[-1]}:{fn[1]}({fn[2]})",
                                 "ncalls": nc, "tottime": round(tt_, 3),
                                 "cumtime": round(ct, 3)})
                rows.sort(key=lambda r: -r["tottime"])
                out["profile_arm"] = arm
                out["profile_top_tottime"] = rows[:25]
                print(f"[profile] arm {arm}: top self-time")
                for r in rows[:15]:
                    print(f"   {r['tottime']:9.3f}s self  {r['ncalls']:8d} calls  {r['fn']}",
                          flush=True)

            reps = []
            for r in range(a.reps):
                for arm in a.arms:
                    secs, d, sites, dig, td = run_arm(trunk, held, a.cycles, arm,
                                                      probe, attrs, out)
                    reps.append({"rep": r, "arm": arm, "s": round(secs, 4),
                                 "context_exit_s": round(td, 4),
                                 "counters": d, "taping_sites": sites, "digest": dig})
                    print(f"[rep {r} arm {arm}] {secs:8.4f}s  "
                          f"{len(d)} counters moved, {len(sites)} taping sites asked",
                          flush=True)
            out["reps"] = reps
            probe.restore()

            med = {arm: statistics.median([x["s"] for x in reps if x["arm"] == arm])
                   for arm in a.arms}
            out["median_s"] = {k: round(v, 4) for k, v in med.items()}
            if "A" in med and "B" in med:
                out["lever_cost_s"] = round(med["B"] - med["A"], 4)
                out["lever_cost_x"] = round(med["B"] / med["A"], 4)
            if "B" in med and "C" in med:
                out["tape_overhead_s"] = round(med["C"] - med["B"], 4)
            out["cycles"] = a.cycles
    except Exception:
        out["error"] = traceback.format_exc()
        print(out["error"], file=sys.stderr, flush=True)
    out["clock"] = clk.summary()
    out["clock_line"] = clk.line(0)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=False))
    print(out.get("clock_line", ""), flush=True)
    print(f"wrote {a.out}", flush=True)
    return 1 if "error" in out else 0


if __name__ == "__main__":
    raise SystemExit(main())
