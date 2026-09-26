#!/usr/bin/env python3
"""The exactness-OFF backward's verb split, at STEP scope, per rep.

`of3t-bwattrib` already took an OFF-arm split (`perf/of3t_bwattrib/out/hist_384_noexact.json`,
33.63 s) and it is not the perf path's decomposition, for two reasons that this file fixes and
does not re-argue:

  * IT IS ONE BACKWARD FROM A COLD PROGRAM CACHE. Its own table says so: `rsqrt` on
    `[1,128,384,1]` reads 114,993 us/call over 6 calls while `rsqrt` on `[1,64,384,1]` reads
    2,288 us/call over 312. Same op class, 98 KB of operand, 50x apart on call count alone.
    That is a JIT compile amortised over a handful of calls, not a cost a training step pays
    every step. A steady-state decomposition has to come off a rep whose programs are already
    built, which needs more than one backward in the process.
  * IT IS THE TRUNK AT `cycles=1`, not the step. The step differentiates the trunk cycle, the
    diffusion samples and the loss heads' device tail together, and `of3t-stepqb2` times that
    at 20.718 s of a 39.886 s step.

So: `fullstep.py` is the step, imported not forked, exactness OFF, and the recorder is
installed around `ag.backward` ONLY. Each rep gets its own `Rec`, so rep 0 (cold) and the
steady reps are separate tables and the difference between them is the one-time cost.

The instrument is `of3t-bwattrib`'s, imported rather than vendored, so the numbers stay
comparable with what is already banked. Two of its columns are void and stay that way:
`gb`/`gb_s` bill both ends of a crossing and size `DataType.FLOAT32` at 2 bytes (`_itemsize`
never matches the uppercase name), so this file drops them from what it publishes rather than
republishing a number the campaign has already voided. The seconds columns stand.

WHAT IS ADDED, and why each is not cosmetic:

  * THE DRAIN, TIMED SEPARATELY. ttnn dispatch is asynchronous, so the per-call wall this
    recorder sums is host time that may be hiding device time behind it. The
    `synchronize_device` that ends each backward is timed on its own: drain near zero means
    the host was already blocking and the verb attribution is real device work, a large drain
    means the backward is host-dispatch-bound and the per-verb ranking is smeared.
  * THE CLOCK OFF THE CARD THIS PROCESS ACTUALLY OPENED. qb1's logical card and device node
    differ (card0=node1, card1=node2, card2=node3, card3=node0), so a sysfs class node built
    from `TT_VISIBLE_DEVICES` reads a different chip's clock. This reads `/proc/self/fd` after
    the device is open and samples `tenstorrent!<that node>`, 1 Hz, into a flushed JSONL from
    a daemon thread, so the clock survives a run that never reaches the last line of `main()`.

    stepbw.py --tokens 384 --reps 3 --out perf/of3t_p10bwd/out/stepbw_384.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.of3t_bwattrib import bwprof as BW                              # noqa: E402

CLASS = Path("/sys/class/tenstorrent")
RECS: list = []          # one dict per ag.backward call
_SUS_PREV: dict = {}


def open_nodes() -> list:
    """The device nodes THIS process holds open, read off /proc. See the docstring."""
    out = []
    try:
        for fd in Path("/proc/self/fd").iterdir():
            try:
                t = os.readlink(fd)
            except OSError:
                continue
            if t.startswith("/dev/tenstorrent/"):
                n = t.rsplit("/", 1)[1]
                if n.isdigit() and n not in out:
                    out.append(n)
    except OSError:
        pass
    return sorted(out)


def clock_thread(node: str, stop: threading.Event, path: Path, period=1.0):
    p = CLASS / f"tenstorrent!{node}"
    with path.open("a", buffering=1) as fh:
        while not stop.is_set():
            try:
                mhz = int((p / "tt_aiclk").read_text().strip())
            except (OSError, ValueError):
                mhz = None
            fh.write(json.dumps({"aiclk_mhz": mhz, "node": node,
                                 "in_backward": bool(IN_BW[0]),
                                 "load1": round(os.getloadavg()[0], 2),
                                 "t": time.time()}) + "\n")
            stop.wait(period)


IN_BW = [False]


def clock_summary(path: Path) -> dict:
    import statistics
    allv, bwv = [], []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("aiclk_mhz") is None:
                continue
            allv.append(r["aiclk_mhz"])
            if r.get("in_backward"):
                bwv.append(r["aiclk_mhz"])

    def red(v):
        return None if not v else {"n": len(v), "min": min(v),
                                   "median": statistics.median(v), "max": max(v)}
    return {"source": "sysfs tt_aiclk on the class node of the device node this process "
                      "opened, 1 Hz, sampled DURING the work",
            "whole_run": red(allv), "during_backward": red(bwv)}


def _verbs(rec):
    """`by_verb` without the two void columns."""
    agg = {}
    for (name, _s, _d, _l), (n, ns, _m, _nb, sns) in rec.verb.items():
        a = agg.setdefault(name, [0, 0, 0])
        a[0] += n
        a[1] += ns
        a[2] += sns
    out = [{"verb": k, "n": v[0], "self_s": round(v[2] / 1e9, 3),
            "incl_s": round(v[1] / 1e9, 3),
            "us_per_call": round(v[2] / v[0] / 1e3, 1)} for k, v in agg.items()]
    out.sort(key=lambda r: -r["self_s"])
    return out


def _shapes(rec, top):
    out = []
    for (name, shp, dt, lay), (n, ns, mx, _nb, sns) in rec.verb.items():
        out.append({"verb": name, "shape": list(shp), "dtype": dt.replace("DataType.", ""),
                    "layout": lay.replace("Layout.", ""), "n": n,
                    "self_s": round(sns / 1e9, 4), "us_per_call": round(sns / n / 1e3, 1),
                    "max_us": round(mx / 1e3, 1)})
    out.sort(key=lambda r: -r["self_s"])
    return out[:top]


def install(ag, dev_box):
    """Recorder around `ag.backward`, per call. `instrument` first: it wraps `_Node.__init__`,
    so it has to be in place before the tape builds a single node.

    `recompute_scope()` is entered BEFORE the recorder installs, and that ordering is the whole
    reason the first attempt at this run died 578 s in. A checkpointed segment recomputes itself
    from inside the backward and asks `taped_ttnn` for the shim to do it; the recorder rebinds
    `mod.ttnn` in every tt_bio module, so if the recorder goes in first the shim's own `_swap`
    finds the recorder already sitting in the slot and the recompute reaches raw `ttnn` holding
    an `autograd.Tensor` (`ttnn.layer_norm(): incompatible function arguments`, at
    `tenstorrent.py:7791`). Entering the scope first makes `_SHIMMED` true, so the recorder
    wraps the shim rather than replacing it and every inner `recompute_scope()` is a no-op.
    This is what `of3t-bwattrib` does and it is not optional.

    The rec is banked in a `finally`, so a backward that raises still publishes the verbs it
    reached instead of costing the whole run.
    """
    import ttnn
    from tt_bio import taped_ttnn as TT
    BW.instrument(ag)
    real = ag.backward
    saved: list = []

    def backward(roots, cotangents=None, *a, **k):
        rec = BW.Rec()
        prev = BW._use(rec)
        sus0 = dict(BW.SUS)
        IN_BW[0] = True
        # Drain BEFORE the clock starts. For a nested call this is the recompute forward's
        # device tail, which is the number that splits a checkpointed segment's cost into
        # "run the block again" and "differentiate it"; without it both land in one drain at
        # the end and the segment is one opaque number.
        t_pre = time.perf_counter()
        ttnn.synchronize_device(dev_box[0])
        t0 = t1 = time.perf_counter()
        rs = TT.recompute_scope()
        rs.__enter__()
        BW._swap(True, saved)
        try:
            return real(roots, cotangents, *a, **k)
        finally:
            BW._swap(False, saved)
            rs.__exit__(None, None, None)
            t1 = time.perf_counter()
            ttnn.synchronize_device(dev_box[0])
            t2 = time.perf_counter()
            IN_BW[0] = False
            BW._use(prev)
            RECS.append({"wall_s": t1 - t0, "drain_s": t2 - t1, "pre_drain_s": t0 - t_pre,
                         "rec": rec,
                         "suspects": {kk: BW.SUS[kk] - sus0.get(kk, 0) for kk in BW.SUS}})

    ag.backward = backward

    # NOT wrapped: `node.group`. A multi-output segment's members share ONE group callable
    # and `_backward` fires it once after the last member on the walk. Wrapping it per member
    # gives the members different objects, the group splits, and the segment recomputes twice
    # -- measured: 105 nested backwards instead of 54, and a 45.18 s backward instead of
    # 24.50 s. The pre-drain below gets the same split without touching the tape.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--top", type=int, default=45)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    clk_path = a.out.with_suffix(".clock.jsonl")

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": os.getloadavg()},
        "config": {"crop": a.tokens, "cycles": a.cycles, "samples": a.samples,
                   "reps": a.reps, "exact_training": False, "batch": 1}}
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))   # noqa: E731
    dump()

    stop = threading.Event()
    try:
        from tt_bio import autograd as ag
        from tt_bio.tenstorrent import get_device
        from perf.of3t_stepfloor import fullstep as F

        dev_box = [get_device()]
        nodes = open_nodes()
        out["env"]["device_nodes_open"] = nodes
        out["env"]["arch"] = str(dev_box[0].arch())
        if nodes:
            threading.Thread(target=clock_thread, args=(nodes[0], stop, clk_path),
                             daemon=True).start()
        else:
            out["env"]["clock_note"] = "no /dev/tenstorrent fd found; clock not sampled"

        install(ag, dev_box)
        step_out = a.out.parent / (a.out.stem + "_fullstep.json")
        sys.argv = ["fullstep.py", "--tokens", str(a.tokens), "--cycles", str(a.cycles),
                    "--samples", str(a.samples), "--reps", str(a.reps),
                    "--out", str(step_out)]
        t0 = time.perf_counter()
        with ag.exact_training(False):
            out["env"]["exact_training_ops"] = list(ag.exact_training_ops())
            rc = F.main()
        out["fullstep_rc"] = rc
        out["fullstep_wall_s"] = round(time.perf_counter() - t0, 2)
        if step_out.exists():
            d = json.loads(step_out.read_text())
            out["step_reps"] = [{k: v for k, v in r.items()
                                 if k.endswith("_s") or k in ("rep", "cold", "tape_nodes",
                                                              "params_with_grad")}
                                for r in d.get("reps", [])]
            out["step_medians"] = d.get("medians")
    except Exception:                                                    # noqa: BLE001
        out["error"] = traceback.format_exc()[-6000:]

    reps = []
    for i, e in enumerate(RECS):
        wall, drain, rec, sus = e["wall_s"], e["drain_s"], e["rec"], e["suspects"]
        reps.append({"rep": i, "cold": i == 0,
                     "backward_s": round(wall, 3), "drain_s": round(drain, 3),
                     "pre_drain_s": round(e["pre_drain_s"], 3),
                     "verb_calls": rec.calls, "outer_verb_calls": rec.outer_calls,
                     "verb_self_s": round(rec.self_ns / 1e9, 3),
                     "verb_share": round(rec.self_ns / 1e9 / wall, 4) if wall else None,
                     "us_per_call": round(rec.self_ns / rec.calls / 1e3, 1)
                     if rec.calls else None,
                     "distinct_shape_classes": len(rec.verb),
                     "suspects": sus,
                     "by_verb": _verbs(rec),
                     "by_shape_class": _shapes(rec, a.top),
                     "by_closure": BW._bynode(rec)[:20]})
    out["backward_reps"] = reps
    if len(reps) >= 2:
        cold, warm = reps[0], reps[-1]
        cv = {r["verb"]: r["self_s"] for r in cold["by_verb"]}
        delta = [{"verb": r["verb"], "cold_s": cv.get(r["verb"], 0.0), "warm_s": r["self_s"],
                  "one_time_s": round(cv.get(r["verb"], 0.0) - r["self_s"], 3)}
                 for r in warm["by_verb"]]
        delta.sort(key=lambda r: -r["one_time_s"])
        out["cold_minus_warm_by_verb"] = delta[:25]
        out["one_time_total_s"] = round(cold["verb_self_s"] - warm["verb_self_s"], 3)
    stop.set()
    time.sleep(1.2)
    out["env"]["aiclk"] = clock_summary(clk_path)
    dump()

    for r in reps:
        print(f"rep {r['rep']}{' COLD' if r['cold'] else ''}: backward {r['backward_s']} s, "
              f"drain {r['drain_s']} s, {r['verb_calls']} verbs, "
              f"self {r['verb_self_s']} s ({r['verb_share']} of wall)", flush=True)
    print(json.dumps(out["env"].get("aiclk"), indent=1), flush=True)
    print(f"-> {a.out}", flush=True)
    return 0 if reps and "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
