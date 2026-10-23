"""Per-family device kernel time of RFD4 trunk passes, for census v2 and the elementwise bet.

    perf/rfd4/e2e/elt_prof.sh <out_dir> [card]    (runs this under `python -m tracy` on the Tracy build)

Runs the engine's own `Trunk` eagerly at 128 tokens, batch 1, on random inputs of the design's shapes
(kernel time does not depend on the values) with an all-ones token mask, which is what a 128-token
design passes. One warm-up call, then one measured call of `--cycles` passes, so the second and later
passes include the recycle projections. Every sublayer is bracketed by `B|<family>` / `E|<family>`
signposts (monkeypatched here, the trunk code is untouched) and elt_prof_table.py charges each op to the
innermost open family. Only DEVICE KERNEL DURATION is read; wall under the profiler is not a measurement.
"""
from __future__ import annotations

import argparse
import functools
import json
import os

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402
from tracy import signpost  # noqa: E402

import clocksample  # noqa: E402


def _bracket(fn, name):
    @functools.wraps(fn)
    def w(*a, **k):
        signpost(f"B|{name}")
        try:
            return fn(*a, **k)
        finally:
            signpost(f"E|{name}")
    return w


class _Bracketed:
    """A bracketed module call that still reads as the module (trunk.py reads tri.took_residual after the call)."""

    def __init__(self, mod, name):
        self._mod, self._call = mod, _bracket(mod.__call__, name)

    def __call__(self, *a, **k):
        return self._call(*a, **k)

    def __getattr__(self, k):
        return getattr(self._mod, k)


def instrument(trunk):
    from tt_bio.rfd4 import trunk as T
    T._as = _bracket(T._as, "cast")
    T._residual = _bracket(T._residual, "residual")
    T._masked_copy = _bracket(T._masked_copy, "masked_copy")
    T.pair_biases = _bracket(T.pair_biases, "pair_bias")
    # The fused programs PairUpdate.run calls directly: without these they fell into
    # pass_glue (census v6: 8.35 s a 512 design with no sub-family).
    T.outer_residual.outer_residual = _bracket(T.outer_residual.outer_residual, "outer_residual")
    for cls, meth, name in ((T.AdaptiveNorm, "__call__", "adaln"), (T.AdaptiveScale, "__call__", "ada_scale"),
                            (T.Transition, "__call__", "token_transition"),
                            (T.PairBiasAttention, "__call__", "attention"),
                            (T.Trunk, "_group_biases", "pair_bias"), (T.Trunk, "_recycle", "recycle"),
                            (T.PairUpdate, "_write", "pair_write"), (T.PairUpdate, "_transition", "pair_transition"),
                            (T.PairUpdate, "_factors", "pair_factors"),
                            (T.PairUpdate, "_transition_block", "pair_relu"),
                            (T.Trunk, "run_pass", "pass")):
        setattr(cls, meth, _bracket(getattr(cls, meth), name))
    for pu in trunk.pair_updates.values():
        pu.tri_out = _Bracketed(pu.tri_out, "tri_mult_out")
        pu.tri_in = _Bracketed(pu.tri_in, "tri_mult_in")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/ttuser/rfd4-bundle")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio.rfd4 import engine as E
    from tt_bio.tenstorrent import upload

    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev, trunk = engine._device, engine.trunk
    instrument(trunk)
    n = a.n
    g = torch.Generator().manual_seed(0)
    tok, pair = int(trunk.blocks[0].mhba.mha.node_norm_w.shape[-1]), int(trunk.z_proj.shape[-1])
    cond = int(trunk.blocks[0].mhba.adaln.gamma_w.shape[0])
    masks = trunk.masks_for(torch.ones(1, n), dev)
    up = lambda t, dt: upload(t, dtype=dt, device=dev)  # noqa: E731

    def call():
        s = up(torch.randn(1, n, tok, generator=g), ttnn.float32)
        z = up(torch.randn(1, n, n, pair, generator=g), ttnn.float32)
        cs = [up(torch.randn(1, n, cond, generator=g), ttnn.bfloat16) for _ in range(a.cycles)]
        so, zo = trunk(s, z, cs, masks)
        ttnn.synchronize_device(dev)
        for t in (s, z, so, zo, *cs):
            ttnn.deallocate(t)

    call()
    ttnn.ReadDeviceProfiler(dev)
    signpost("elt_prof")
    with clocksample.during(period=0.5) as clk:
        call()
    signpost("elt_end")
    ttnn.synchronize_device(dev)
    json.dump({"n": n, "cycles": a.cycles, "tok": tok, "pair": pair, "cond": cond, "aiclk": clk.summary(),
               "load": os.getloadavg()[0], "card": os.environ.get("TT_VISIBLE_DEVICES")}, open(a.out, "w"), indent=2)
    ttnn.ReadDeviceProfiler(dev)


if __name__ == "__main__":
    main()
