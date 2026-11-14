"""One whole RFD4 network call under the device profiler: the pair built on the card, the trunk's entry copies,
every pass and the s readback, the path `_CardTrunk.launch` + `collect` runs once per sampler step.

    (under tracy, see call_prof.sh) call_prof.py --n 512 --out run.json

elt_prof.py profiles `Trunk.__call__` on uploaded random z; this goes through `Trunk.launch` with the engine's
`PairOnCard` builder on synthetic `PairInputs` of the design's shapes (the card does the same work for any values),
so the census also sees what a step costs outside the trunk call. Families: elt_prof's sublayers plus
`pair_build` (PairOnCard), `trunk_entry` (the masked copies at the top of Trunk.__call__), `collect` (readback).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import torch  # noqa: E402
import ttnn  # noqa: E402
import clocksample  # noqa: E402
from tracy import signpost  # noqa: E402
from elt_prof import _bracket, instrument  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/ttuser/rfd4-bundle")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import pair as P
    from tt_bio.rfd4 import trunk as T

    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev, trunk, builder = engine._device, engine.trunk, engine.pair
    assert builder is not None, "this engine does not build the pair on the card"
    instrument(trunk)
    P.PairOnCard.__call__ = _bracket(P.PairOnCard.__call__, "pair_build")
    T.Trunk.collect = _bracket(T.Trunk.collect, "collect")
    call = T.Trunk.__call__

    def entry_bracketed(self, s_init, z_init, conds, masks):
        # the two masked copies at the top of __call__, bracketed by running them under the family name
        orig = T._masked_copy
        T._masked_copy = _bracket(orig, "trunk_entry")
        try:
            return call(self, s_init, z_init, conds, masks)
        finally:
            T._masked_copy = orig
    T.Trunk.__call__ = entry_bracketed

    n, g = a.n, torch.Generator().manual_seed(0)
    tok, pair = int(trunk.blocks[0].mhba.mha.node_norm_w.shape[-1]), int(trunk.z_proj.shape[-1])
    cond = int(trunk.blocks[0].mhba.adaln.gamma_w.shape[0])
    nb = builder.split.nbins
    const = P.PairConstant(torch.randn(1, n, n, pair, generator=g), torch.rand(1, n, n, nb, generator=g) + 0.5)
    mask = torch.ones(1, n)

    def step():
        p = P.PairInputs(const, torch.randint(0, nb, (1, n, n), generator=g), torch.rand(1, n, n, generator=g) + 0.5,
                         torch.randn(1, pair, generator=g), torch.randn(1, pair, generator=g))
        s = torch.randn(1, n, tok, generator=g)
        cs = [torch.randn(1, n, cond, generator=g) for _ in range(a.cycles)]
        out = trunk.collect(trunk.launch(s, p, cs, mask, dev, pair=False, z_builder=builder))
        ttnn.synchronize_device(dev)
        return out

    step()  # compile; also uploads the design's constant, which a design does once
    ttnn.ReadDeviceProfiler(dev)
    signpost("elt_prof")
    with clocksample.during(period=0.5) as clk:
        step()
    signpost("elt_end")
    ttnn.synchronize_device(dev)
    json.dump({"n": n, "cycles": a.cycles, "tok": tok, "pair": pair, "cond": cond, "aiclk": clk.summary(),
               "load": os.getloadavg()[0], "card": os.environ.get("TT_VISIBLE_DEVICES")}, open(a.out, "w"), indent=2)
    ttnn.ReadDeviceProfiler(dev)


if __name__ == "__main__":
    main()
