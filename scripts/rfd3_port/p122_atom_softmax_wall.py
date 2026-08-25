#!/usr/bin/env python3
"""p122 -- S1 of the RFD3 fusion programme: the denominator L5b is scored against.

p49x measured the atom regions at 87.9 + 41.7 = 129.6 unsynced wall ms/step while its own
wrapped rows summed to only 56.0 + 22.3 = 78.3. An itemiser that UNDER-counts is pointing at
calls it does not wrap (`state/rfd3-fusion-programme.md` §2), and in the atom path the unwrapped
calls are exactly the three L5b touches: the fused softmax kernel (`ttnn.generic_op`), the
`ttnn.empty` that allocates its 294.3 MB output, and the PV matmul that reads it back.

This wraps those three and nothing else, so the rest of the step runs at its normal speed and
the inflation is bounded by ~4 syncs per call rather than p49's per-op syncing.

Acceptance, pre-committed in §4 of the state doc: **if `softmax_into` is not at least 25 ms/step
on the atom shape, §2's inference is wrong and L5b is re-priced before it is built.**

S2 (`perf/p123/pv_korder.json`) already timed the same three calls in ISOLATION: 4.7867 ms/call
softmax + 0.7971 ms/call PV over 9 calls = 50.3 ms/step. Isolated timing oversyncs and inflates
(`tt-bio-isolated-op-timing-oversync-inflates-cost`), so the in-fold number here is the honest
denominator and is expected to come in AT OR BELOW 50.3, not above it.

    ~/.coworker/scripts/benchlock.sh rfd3-fusion-programme-p1 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:rfd3-fusion-programme-p1 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p122_atom_softmax_wall.py
"""
import collections
import json
import os
import pathlib
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                      # noqa: E402
from tt_bio.rfd3 import model as M                                 # noqa: E402
from tt_bio import softmax_generic                                 # noqa: E402

FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p122/atom_softmax_wall.json")
STEPS = 8
WARM_STEPS = 3
ATOM_K = 6080          # the dense atom key axis; anything narrower is a DiT / local-attention call

STEP = [0]
ACC = collections.defaultdict(lambda: [0.0, 0])    # (call, bucket) -> [seconds, n]
STEP_WALL = []


def _bucket(width):
    """Atom path or not. L5b is only about the dense 6080-wide chain."""
    return "atom_6080" if int(width) == ATOM_K else "other_w%d" % int(width)


def _timed(call, bucket, fn):
    """Sync on both sides, but only once counting has started."""
    if STEP[0] < WARM_STEPS:
        return fn()
    dev = M.get_device()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        ttnn.synchronize_device(dev)
        e = ACC[(call, bucket)]
        e[0] += time.perf_counter() - t0
        e[1] += 1


def main():
    # --- the three L5b touches, and nothing else -------------------------------------------
    _softmax_into = softmax_generic.softmax_into

    def softmax_into(device, x, out, *a, **k):
        return _timed("softmax_into", _bucket(x.shape[-1]),
                      lambda: _softmax_into(device, x, out, *a, **k))
    softmax_generic.softmax_into = softmax_into

    _empty = ttnn.empty

    def empty(shape, *a, **k):
        return _timed("ttnn.empty", _bucket(list(shape)[-1]), lambda: _empty(shape, *a, **k))
    ttnn.empty = empty

    # model.py did `from ..tenstorrent import attn_value_matmul`, so the binding that the fold
    # actually calls lives on the MODEL module. Patching tenstorrent's would miss every call.
    _pv = M.attn_value_matmul

    def attn_value_matmul(attn, v, ckc, dtype):
        return _timed("attn_value_matmul", _bucket(attn.shape[-1]),
                      lambda: _pv(attn, v, ckc, dtype))
    M.attn_value_matmul = attn_value_matmul

    # --- step counter + the unsynced step wall ----------------------------------------------
    dm_cls = M.RFD3DiffusionModule
    _call = dm_cls.__call__

    def stepped(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return _call(self, *a, **k)
        finally:
            dt = time.perf_counter() - t0
            if STEP[0] >= WARM_STEPS:
                STEP_WALL.append(dt)
            STEP[0] += 1
    dm_cls.__call__ = stepped

    specs = json.loads(FIXTURE.read_text())
    out_dir = "/tmp/rfd3_p122"
    os.system("rm -rf %s" % out_dir)

    t0 = time.perf_counter()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=8,
                           verbose=True)
    wall = time.perf_counter() - t0
    counted = STEP[0] - WARM_STEPS

    rows = []
    for (call, bucket), (tot, n) in sorted(ACC.items(), key=lambda kv: -kv[1][0]):
        rows.append({"call": call, "bucket": bucket,
                     "ms_per_step": round(1000 * tot / counted, 3),
                     "calls_per_step": round(n / counted, 1),
                     "ms_per_call": round(1000 * tot / n, 4)})

    atom = {r["call"]: r for r in rows if r["bucket"] == "atom_6080"}
    chain = round(sum(r["ms_per_step"] for r in atom.values()), 3)
    softmax_ms = atom.get("softmax_into", {}).get("ms_per_step", 0.0)
    step_ms = round(1000 * sorted(STEP_WALL)[len(STEP_WALL) // 2], 1) if STEP_WALL else 0.0

    res = {"fixture": str(FIXTURE), "atoms": 6051, "tokens": 685, "batch": 1,
           "num_timesteps": STEPS, "counted_steps": counted, "host": "qb2", "card": 1,
           "ttnn": "0.68.0", "torch": torch.__version__, "total_wall_s": round(wall, 1),
           "step_wall_ms": step_ms, "rows": rows,
           "atom_chain_ms_per_step": chain,
           "softmax_into_ms_per_step": softmax_ms,
           "softmax_served": softmax_generic.SSTATS[0]}

    print("\n%-22s %-12s %10s %11s %11s" % ("call", "bucket", "ms/step", "calls/step", "ms/call"))
    for r in rows:
        print("%-22s %-12s %10.3f %11.1f %11.4f"
              % (r["call"], r["bucket"], r["ms_per_step"], r["calls_per_step"], r["ms_per_call"]))

    print("\natom-path L5b chain: %.1f ms/step of a %.1f ms step (%.1f %%)"
          % (chain, step_ms, 100 * chain / step_ms if step_ms else 0))
    print("softmax_into on the atom shape: %.1f ms/step" % softmax_ms)

    # The gate, stated before the run and evaluated here rather than in prose afterwards.
    res["acceptance_gate"] = "softmax_into >= 25 ms/step on atom_6080"
    res["acceptance_pass"] = bool(softmax_ms >= 25.0)
    print("ACCEPTANCE (%s): %s"
          % (res["acceptance_gate"], "PASS" if res["acceptance_pass"] else
             "FAIL -- re-price L5b before building it"))

    # The prize, re-stated against the number this run measured rather than against the
    # isolated one. COST-MODEL ESTIMATE, and it stays one until the fold A/B reports.
    # L5b deletes the empty, the kernel's 294.3 MB pack and the PV matmul's read of it; what
    # survives is the kernel's own score read and the PV arithmetic, which moves inside it.
    deletable = round(chain - softmax_ms * (1.509 / 4.06), 3)   # score read stays, at the roof split
    res["deletable_ms_per_step_estimate"] = deletable
    res["prize_s_per_design_estimate"] = {
        "at_110pct": round(-1.10 * deletable * 200 / 1000, 3),
        "at_75pct": round(-0.75 * deletable * 200 / 1000, 3),
        "at_41pct": round(-0.41 * deletable * 200 / 1000, 3),
    }
    print("deletable %.1f ms/step -> %s s/design (COST-MODEL ESTIMATE, not a fold A/B)"
          % (deletable, res["prize_s_per_design_estimate"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
