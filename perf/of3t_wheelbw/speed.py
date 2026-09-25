#!/usr/bin/env python3
"""Per-node cost of the J4 substitutions, arms interleaved, AICLK sampled DURING.

What this times is the BACKWARD CLOSURE of each op, composed against wheel, at the shapes
those nodes carry in the crop-384 OF3T step. Nothing else: the forward is untouched by every
substitution in this row, so a step-level A/B would spend 466 s to read a difference the
closures own entirely.

Arms are interleaved A B A B and an A/A floor runs first, because a drifting host is the
failure mode a block-of-A-then-block-of-B schedule cannot see. Seconds off the step come
from this per-node delta multiplied by the node census in `census.py`, and both halves are
reported rather than only their product.
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np
import ttnn
from tt_bio import autograd as ag
from tt_bio.tenstorrent import get_device

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from perf.clocksample import TT_SMI, during                           # noqa: E402


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                          text=True).stdout.strip() or None


def _t(v, dt, dev, layout=ttnn.TILE_LAYOUT):
    return ttnn.Tensor(np.ascontiguousarray(v.astype(np.float32)), dt).to(layout).to(dev)


# Shapes are the ones these nodes carry in the crop-384 step: the pair track at
# [1, 384, 384, c], the single track at [1, 384, c]. `census.py` resolves which op sits on
# which; this file only has to time a node of each kind at a size it really runs at.
def build(op, dev, dt, rng, shape):
    x = rng.standard_normal(shape)
    g = rng.standard_normal(shape)
    tx, tg = _t(x, dt, dev), _t(g, dt, dev)
    ty = None
    if op == "mul":
        tb = _t(rng.standard_normal(shape), dt, dev)
        return {
            "composed": lambda: (ttnn.multiply(tg, tb), ttnn.multiply(tg, tx)),
            "wheel": lambda: ttnn.mul_bw(tg, tx, tb),
        }
    if op == "relu":
        ty = ttnn.relu(tx)
        return {
            "composed": lambda: ttnn.multiply(tg, ttnn.gtz(ty)),
            "wheel": lambda: ttnn.relu_bw(tg, ty),
        }
    if op == "sigmoid":
        ty = ttnn.sigmoid(tx)
        return {
            "composed": lambda: ttnn.multiply(tg, ttnn.multiply(ty, ttnn.rsub(ty, 1.0))),
            "wheel": lambda: ttnn.sigmoid_bw(tg, tx),
        }
    if op == "silu":
        def composed():
            sig = ttnn.sigmoid(tx)
            d = ttnn.multiply(sig, ttnn.add(ttnn.multiply(tx, ttnn.rsub(sig, 1.0)), 1.0))
            return ttnn.multiply(tg, d)
        return {"composed": composed, "wheel": lambda: ttnn.silu_bw(tg, tx)}
    if op == "narrow":
        # the gradient arrives at the SLICE's shape and the backward restores the source
        n = shape[-2]
        sl = list(shape); sl[-2] = n // 2
        tgs = _t(rng.standard_normal(sl), dt, dev)
        before = n // 4
        after = n - before - n // 2
        z = lambda k: _t(np.zeros(shape[:-2] + [k] + shape[-1:]), dt, dev)
        zb, za = z(before), z(after)
        pad = [(0, 0)] * len(shape)
        pad[-2] = (before, after)
        return {
            "composed": lambda: ttnn.concat([zb, tgs, za], dim=len(shape) - 2),
            "wheel": lambda: ttnn.pad(tgs, padding=pad, value=0.0),
        }
    if op == "concat":
        n = shape[-2]
        half = list(shape); half[-2] = n // 2
        ta, tb = _t(rng.standard_normal(half), dt, dev), _t(rng.standard_normal(half), dt, dev)
        ax = len(shape) - 2
        def composed():
            s0 = [0] * len(shape); e0 = list(shape); e0[ax] = n // 2
            s1 = [0] * len(shape); s1[ax] = n // 2; e1 = list(shape)
            return ttnn.slice(tg, s0, e0), ttnn.slice(tg, s1, e1)
        return {"composed": composed, "wheel": lambda: ttnn.concat_bw(tg, ta, tb, ax)}
    raise KeyError(op)


def timed(fn, dev, iters):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--ops", default="mul,relu,sigmoid,silu,narrow,concat")
    ap.add_argument("--shape", default="1,384,384,128")
    ap.add_argument("--reps", type=int, default=5, help="interleaved A B pairs")
    ap.add_argument("--iters", type=int, default=20, help="closure calls per timed block")
    ap.add_argument("--out", default="/tmp/of3t/of3t-wheelbw/speed.json")
    a = ap.parse_args()
    shape = [int(v) for v in a.shape.split(",")]
    dt = {"float32": ttnn.float32, "bfloat16": ttnn.bfloat16}[a.dtype]

    dev = get_device()
    clk = during(period=1.0)
    clk.__enter__()
    # R209: an artifact without the commit it ran on cannot be checked for expiry, and the
    # campaign quoted a four-day-stale step because the one it had only recorded provenance.
    # `exact_training_ops()` is READ, not asserted -- the sprint grades on `exact_training(False)`
    # (pass 450) and an arm whose setting is not written down is comparable to nothing.
    out = {"host": os.uname().nodename, "card": a.card, "board": None,
           "dtype": a.dtype, "shape": shape, "reps": a.reps, "iters": a.iters, "ops": {},
           "env": {"commit": _git("rev-parse", "HEAD"),
                   "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                   "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "exact_training": {"ops": list(ag.exact_training_ops()),
                              "on": bool(ag.exact_training_ops())}}
    try:
        try:
            devs = json.loads(subprocess.run([TT_SMI, "-s"], capture_output=True, text=True,
                                             timeout=30).stdout)["device_info"]
            i = min(int(a.card), len(devs) - 1)
            out["board"] = devs[i]["board_info"]["board_type"]
        except Exception as e:
            out["board"] = f"unread: {e}"[:120]

        for op in a.ops.split(","):
            rng = np.random.default_rng(11)
            try:
                arms = build(op, dev, dt, rng, list(shape))
            except Exception as e:
                out["ops"][op] = {"build_error": f"{type(e).__name__}: {e}"[:300]}
                print(op, "BUILD ERROR", out["ops"][op]["build_error"][:160], flush=True)
                continue
            # warm both arms so neither pays the other's JIT
            try:
                for f in arms.values():
                    timed(f, dev, 2)
            except Exception as e:
                out["ops"][op] = {"warm_error": f"{type(e).__name__}: {e}"[:300]}
                print(op, "WARM ERROR", out["ops"][op]["warm_error"][:160], flush=True)
                continue
            aa, ab = [], []
            floor = []
            try:
                for _ in range(a.reps):            # A/A floor first, same schedule
                    floor.append(timed(arms["composed"], dev, a.iters))
                    floor.append(timed(arms["composed"], dev, a.iters))
                for _ in range(a.reps):            # then A B A B
                    aa.append(timed(arms["composed"], dev, a.iters))
                    ab.append(timed(arms["wheel"], dev, a.iters))
            except Exception as e:
                out["ops"][op] = {"run_error": f"{type(e).__name__}: {e}"[:300]}
                print(op, "RUN ERROR", out["ops"][op]["run_error"][:160], flush=True)
                continue
            f0 = statistics.median(floor[0::2]); f1 = statistics.median(floor[1::2])
            m_a, m_b = statistics.median(aa), statistics.median(ab)
            out["ops"][op] = {
                "composed_s": m_a, "wheel_s": m_b,
                "delta_s_per_node": m_a - m_b,
                "ratio_x": (m_a / m_b) if m_b else None,
                "aa_floor_s": [f0, f1], "aa_floor_rel": abs(f0 - f1) / max(f0, 1e-12),
                "composed_all": aa, "wheel_all": ab,
            }
            print(f"{op:8s} composed {m_a*1e3:8.3f} ms  wheel {m_b*1e3:8.3f} ms  "
                  f"delta {(m_a-m_b)*1e3:8.3f} ms  A/A floor "
                  f"{out['ops'][op]['aa_floor_rel']*100:.2f} %", flush=True)
    finally:
        clk.__exit__()
        out["aiclk_during"] = clk.summary()
        out["aiclk_line"] = clk.line()
        # `get_device` owns the handle and the lease for the process;
        # process exit releases both.
    print(out["aiclk_line"], flush=True)
    p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
