#!/usr/bin/env python3
"""Settle the 11.7x: one commit, one card, one quiet host, and the sample axis beside it.

`of3t-stepfloor` banked 466.70 s and `of3t-stepqb2` banked 39.886 s for what both artifacts
declare as the same step. This runs both argv in ONE process on ONE tree, then sweeps
`--samples`, so the three readings that could make 39.886 s not a step are decided in the same
card visit as the reproduction.

NOTHING HERE IS A SECOND STEP IMPLEMENTATION. `of3t-stepqb2`'s `steprun.py` does not contain a
step: it calls `of3t-restep`'s `steparms.arm`, which calls `of3t-stepfloor`'s `fullstep.main`.
The two "harnesses" are one harness, so the comparison that matters is the one this file runs --
`fullstep`'s own argv against `steprun`'s argv, on the same tree, with exactness OFF on both.
Exactness has no environment switch (`autograd.py`: "There is no environment variable"), so a
bare `fullstep.py` on this tree runs the host float64 softmax and layer norm; `steparms.arm`
opens `exact_training(False)` and every arm here goes through it.

WHAT EACH ARM ANSWERS

  qb2repro    steprun's argv (--cycles 4 --samples 4), quiet host. Reproduces 39.886 s or not.
  floorrepro  stepfloor's argv verbatim (--cycles 1 --samples 4 --renorm-per-rep 0,0,1,0).
              Same tree as qb2repro, so any gap left is the argv and not the commit.
  s1 s2 s8    diffusion_s must GROW with --samples. A phase that reads 10.858 s cold and
              0.283 s steady is either program-cache warmup or a phase that stopped working,
              and the slope is what tells them apart.
  s48         upstream's own axis. Expected to exceed the board's DRAM; the arm is kept so the
              failure is a recorded number rather than an estimate.

The gradient census rides along: after every backward this records WHICH declared weights
carry a gradient, so `2,674 of 3,152` stops being a count and becomes 478 names.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "perf" / "of3t_p10axis" / "out"
GIB = 1024.0 ** 3

# arm -> [{rep, with_grad, without_grad: [names]}], filled by the backward wrapper below.
CENSUS: dict = {}
_LIVE = {"params": None, "arm": None, "rep": 0}


def stash_parameters():
    """Hold the `Parameters` fullstep declares, so the census reads the real object.

    `declare_all` builds it inside `fullstep.main` and never publishes it; wrapping the
    constructor is the only way to see it without forking the harness.
    """
    from tt_bio.train import lora
    real = lora.Parameters.__init__

    def w(self, *a, **k):
        real(self, *a, **k)
        _LIVE["params"] = self
    lora.Parameters.__init__ = w


def census_backward(ag):
    """Name the weights the backward did NOT reach, once per rep.

    Wraps whatever `ag.backward` is NOW, so it sits outside `steprun.timed`'s clock and adds
    nothing to `backward_s`.
    """
    real = ag.backward

    def w(*a, **k):
        try:
            return real(*a, **k)
        finally:
            p = _LIVE["params"]
            if p is not None and _LIVE["arm"]:
                miss = sorted(n for n, t in p.items() if getattr(t, "grad", None) is None)
                CENSUS.setdefault(_LIVE["arm"], []).append(
                    {"rep": _LIVE["rep"], "declared": len(p),
                     "with_grad": len(p) - len(miss), "without_grad": len(miss),
                     "missing": miss})
                _LIVE["rep"] += 1
    ag.backward = w


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--card", default=os.environ.get("TT_VISIBLE_DEVICES", "0"))
    ap.add_argument("--floor-gib", type=float, default=1.5)
    ap.add_argument("--period", type=float, default=0.25)
    ap.add_argument("--arms", default="qb2repro,floorrepro,s1,s2,s6,s8,s48")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or str(a.tokens)
    OUT.mkdir(parents=True, exist_ok=True)

    import perf.of3t_stepqb2.steprun as SR
    import perf.of3t_restep.rssprofile as R

    jsonl = R.start(OUT / f"rss_axis_{tag}.jsonl", floor_gib=a.floor_gib, period=a.period,
                    header={"tag": tag, "row": "of3t-p10axis",
                            "host_facts": SR.board_facts(a.card)})
    clk_stop = threading.Event()
    threading.Thread(target=SR.aiclk_thread,
                     args=(SR.class_node(a.card), clk_stop, R._JSONL), daemon=True).start()

    import perf.of3t_restep.steparms as A
    ag, F = R.wire_phases()
    verbs: list = []
    SR.timed(R, ag, F, verbs)
    stash_parameters()
    census_backward(ag)

    T = ["--tokens", str(a.tokens)]
    PLAN = {
        # name          argv after --tokens                                       reps
        "qb2repro":   T + ["--cycles", "4", "--samples", "4", "--reps", "3"],
        "floorrepro": T + ["--cycles", "1", "--samples", "4", "--reps", "3",
                           "--renorm-per-rep", "0,0,1,0"],
        "s1":         T + ["--cycles", "4", "--samples", "1", "--reps", "2"],
        "s2":         T + ["--cycles", "4", "--samples", "2", "--reps", "2"],
        "s8":         T + ["--cycles", "4", "--samples", "8", "--reps", "2"],
        "s6":         T + ["--cycles", "4", "--samples", "6", "--reps", "2"],
        "s48":        T + ["--cycles", "4", "--samples", "48", "--reps", "1"],
    }
    facts = SR.board_facts(a.card)
    ran, rc = [], 0
    for name in [w.strip() for w in a.arms.split(",") if w.strip()]:
        path = OUT / f"arm_{name}_{tag}.json"
        _LIVE["arm"], _LIVE["rep"] = name, 0
        t0 = time.perf_counter()
        try:
            arc = A.arm(name, False, PLAN[name], path)
        except BaseException as e:          # an arm that dies is a reading, not the run's end
            arc = 2
            path.write_text(json.dumps({"arm": {"name": name, "died": repr(e)}}, indent=1))
        rc |= arc
        d = json.loads(path.read_text())
        d["host_facts"] = facts
        d["avail_at_start_gib"] = round(R._mem_available_bytes() / GIB, 3)
        d["grad_census"] = CENSUS.get(name, [])
        path.write_text(json.dumps(d, indent=1, default=str))
        ran.append({"arm": name, "rc": arc, "wall_s": round(time.perf_counter() - t0, 1),
                    "cold_s": d.get("cold_s"),
                    "steady_median_s": (d.get("steady") or {}).get("median_s"),
                    "error": (d.get("error") or "").splitlines()[-1:] or None})
        print(f"[axis] {name} rc={arc} cold={d.get('cold_s')} "
              f"steady={(d.get('steady') or {}).get('median_s')}", flush=True)
        # The census reference pins the arm's gradients on the card, so it is
        # dropped before the collection rather than after the next arm allocates.
        _LIVE["params"] = None
        with R.phase("arm_boundary"):
            gc.collect()
        # A run that dies here loses every arm after it, so the roll-up is written each time.
        (OUT / f"AXIS_{tag}.json").write_text(json.dumps(
            {"row": "of3t-p10axis", "host_facts": facts, "arms": ran,
             "aiclk": SR.clock_summary(jsonl), "verbs": verbs,
             "jsonl": str(jsonl.relative_to(REPO))}, indent=1, default=str))

    clk_stop.set()
    time.sleep(1.5)
    foot = R.finish(rc, 0.0, ag, period=a.period, host_facts=facts)
    (OUT / f"AXIS_{tag}.json").write_text(json.dumps(
        {"row": "of3t-p10axis", "host_facts": facts, "arms": ran, "footer": foot,
         "aiclk": SR.clock_summary(jsonl), "verbs": verbs,
         "jsonl": str(jsonl.relative_to(REPO))}, indent=1, default=str))
    print(json.dumps(SR.clock_summary(jsonl), indent=1), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
