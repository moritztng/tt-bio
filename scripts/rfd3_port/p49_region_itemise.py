"""Itemise the three device regions nobody has looked at: the DiT, the decoder, the atom encoder.

P3.2 attributed the step's 503.6 ms of exposed device time to four regions by which drain waited for
it. The token encoder (242.6 ms) is itemised and roofed already (p46, p48). This does the other
three -- LocalTokenTransformer 112.6, CompactStreamingDecoder 93.2, LocalAtomTransformer 44.8, 250.6
ms/step between them -- with p46's method: `ttnn.synchronize_device` on both sides of every op, but
only while the call stack is inside one of the three regions, so the rest of the step runs at its
normal speed and one run covers all three.

Read the output the way p46's docstring says to read its own: the synced sum overshoots the region's
true wall (`tt-bio-isolated-op-timing-oversync-inflates-cost`), so use the RANKING and the RATIOS,
and reconcile against the unsynced region wall printed beside each region -- which is measured here
too, on the same run, by timing the region entry point without inner syncs on the warmup calls.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p49_region_itemise.py
"""
import collections
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                      # noqa: E402
from tt_bio.rfd3 import model as M                                 # noqa: E402

FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p49/region_itemise.json")
STEPS = 8
WARM_STEPS = 3          # steps 0-2 discarded; they also carry the unsynced region walls

TT_OPS = ("linear", "matmul", "add", "add_", "multiply", "multiply_", "subtract", "softmax",
          "typecast", "scatter", "embedding", "reshape", "permute", "pad", "to_layout", "concat",
          "rms_norm", "layer_norm", "sigmoid", "clone", "full", "from_torch", "to_torch",
          "transpose", "repeat", "sum", "mean", "exp", "where", "zeros", "arange", "silu")

REGION = [None]
STEP = [0]
ACC = collections.defaultdict(lambda: [0.0, 0])     # (region, label) -> [s, n]
WALL = collections.defaultdict(list)                # region -> unsynced walls (warmup only)
SYNCED = collections.defaultdict(list)              # region -> synced walls (counted only)


def _sync():
    ttnn.synchronize_device(M.get_device())


def _wrap_op(name):
    fn = getattr(ttnn, name, None)
    if fn is None or not callable(fn):
        return
    def w(*a, **k):
        if REGION[0] is None:
            return fn(*a, **k)
        fr = sys._getframe(1)
        label = "%s@%d" % (name, fr.f_lineno)
        _sync()
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _sync()
            dt = time.perf_counter() - t0
            if STEP[0] >= WARM_STEPS:
                e = ACC[(REGION[0], label)]
                e[0] += dt
                e[1] += 1
    setattr(ttnn, name, w)


def _wrap_region(cls, meth, region):
    fn = getattr(cls, meth)
    def w(self, *a, **k):
        counted = STEP[0] >= WARM_STEPS
        REGION[0] = region if counted else None    # warmup runs unsynced, for the true wall
        _sync()
        t0 = time.perf_counter()
        try:
            return fn(self, *a, **k)
        finally:
            _sync()
            dt = time.perf_counter() - t0
            REGION[0] = None
            (SYNCED if counted else WALL)[region].append(dt)
    setattr(cls, meth, w)


def main():
    for n in TT_OPS:
        _wrap_op(n)
    _wrap_region(M.LocalTokenTransformer, "run_device", "token_dit")
    _wrap_region(M.CompactStreamingDecoder, "run_full_device", "decoder")
    _wrap_region(M.LocalAtomTransformer, "run_device", "atom_encoder")

    specs = json.loads(FIXTURE.read_text())
    out_dir = "/tmp/rfd3_p49"
    os.system("rm -rf %s" % out_dir)

    # step counter: one RFD3DiffusionModule.__call__ is one sampler step
    dm_cls = M.RFD3DiffusionModule
    call = dm_cls.__call__
    def stepped(self, *a, **k):
        try:
            return call(self, *a, **k)
        finally:
            STEP[0] += 1
    dm_cls.__call__ = stepped

    t0 = time.perf_counter()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=8,
                           verbose=True)
    wall = time.perf_counter() - t0
    counted_steps = STEP[0] - WARM_STEPS

    result = {"fixture": str(FIXTURE), "atoms": 6051, "tokens": 685, "batch": 1,
              "num_timesteps": STEPS, "counted_steps": counted_steps,
              "host": "qb2", "card": 0, "ttnn": "0.68.0", "torch": torch.__version__,
              "total_wall_s": round(wall, 1), "regions": {}}

    for region in ("token_dit", "decoder", "atom_encoder"):
        rows = []
        for (r, label), (tot, n) in ACC.items():
            if r != region:
                continue
            rows.append({"op": label, "n_per_step": round(n / counted_steps, 1),
                         "ms_per_step": round(1000 * tot / counted_steps, 3)})
        rows.sort(key=lambda r: -r["ms_per_step"])
        synced = sum(r["ms_per_step"] for r in rows)
        n_calls = len(WALL[region]) / max(1, WARM_STEPS)
        unsynced = 1000 * statistics.median(WALL[region]) * n_calls if WALL[region] else 0.0
        print("\n=== %s: %d ops, synced sum %.1f ms/step, unsynced wall %.1f ms/step, "
              "inflation %.2fx ===" % (region, len(rows), synced, unsynced,
                                       synced / unsynced if unsynced else 0))
        print("%-28s %10s %11s %7s" % ("op@line", "ms/step", "calls/step", "%region"))
        for r in rows[:18]:
            if r["ms_per_step"] < 0.05:
                continue
            print("%-28s %10.3f %11.1f %6.1f%%"
                  % (r["op"], r["ms_per_step"], r["n_per_step"],
                     100 * r["ms_per_step"] / synced if synced else 0))
        result["regions"][region] = {"rows": rows, "synced_sum_ms_per_step": round(synced, 3),
                                     "unsynced_wall_ms_per_step": round(unsynced, 3),
                                     "calls_per_step": n_calls}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
