#!/usr/bin/env python3
"""Leg 1: what one composed round holds on the card, and what a second trajectory would add.

Two trajectories in flight means two sets of activations resident at once, so before any
speed question this row has to answer whether the card has room. The answer is two numbers
and not one: the RESIDENT floor (weights, masks, anything alive at a round boundary), which
two trajectories SHARE, and the PEAK above that floor, which they do not.

Sampled at block granularity rather than at the seam, because the peak is inside the seam:
`_Trunk.evoformer` runs 48 blocks and the backward recomputes them, and a probe at the seam
boundary reads the round's floor twice and never its high-water. `ttnn.get_memory_view`
behaves like a pipeline drain (`tenstorrent.dram_peak`'s own note, 12.0 s -> 28.8 s on a
117-aa fold), so this arm's seconds are the probe's and are never quoted as a round time.

Runs `perf/bcx_round/run_round.py` unchanged underneath; everything here is a wrapper the
timed arms do not load.
"""
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
sys.path.insert(0, str(_ROOT / "perf" / "bcx_round"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SAMPLES: list = []
OUT = os.environ.get("DUOTRAJ_FOOTPRINT_OUT", "footprint.json")


def _view():
    import ttnn
    from tt_bio.tenstorrent import get_device
    mv = ttnn.get_memory_view(get_device(), ttnn.BufferType.DRAM)
    banks = int(mv.num_banks)
    total = int(mv.total_bytes_per_bank) * banks
    free = int(mv.total_bytes_free_per_bank) * banks
    lcf = mv.largest_contiguous_bytes_free_per_bank
    if isinstance(lcf, (list, tuple)):
        lcf = min(lcf)
    return total, total - free, int(lcf) * banks


def sample(tag):
    try:
        total, used, lcf = _view()
    except Exception as exc:                       # a probe must never kill the round
        SAMPLES.append([time.time(), tag, -1, -1, -1, repr(exc)])
        return
    SAMPLES.append([time.time(), tag, used, total, lcf])


class _Probe:
    """One Evoformer / extra-MSA block with a DRAM read after it.

    `__getattr__` passes `_residual` through, which the extra-MSA loop calls on the block
    before calling the block itself.
    """

    def __init__(self, blk, tag):
        self._blk, self._tag = blk, tag

    def __getattr__(self, name):
        return getattr(self._blk, name)

    def __call__(self, *a, **kw):
        out = self._blk(*a, **kw)
        sample(self._tag)
        return out


def install():
    from tt_bio import bindcraft2
    import meter as M

    trunk_init = bindcraft2._Trunk.__init__

    def __init__(self, *a, **kw):
        trunk_init(self, *a, **kw)
        self.model.device_evoformer = [_Probe(b, f"evo{i}")
                                       for i, b in enumerate(self.model.device_evoformer)]
        self.model.device_extra_msa = [_Probe(b, f"xmsa{i}")
                                       for i, b in enumerate(self.model.device_extra_msa)]
        sample("trunk_loaded")
    bindcraft2._Trunk.__init__ = __init__

    # The seams themselves, so a sample exists even on a round whose peak is outside a block
    # (the upload, the cotangent seed, the readback).
    for module, cls in (("evo", bindcraft2.EvoformerOnDevice),
                        ("xmsa", bindcraft2.ExtraMsaOnDevice),
                        ("tmpl", bindcraft2.TemplateOnDevice)):
        for name in ("_primal", "_taped", "_backward"):
            orig = getattr(cls, name)

            def make(orig, tag):
                def wrapper(self, *a, **kw):
                    sample(tag + ":enter")
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        sample(tag + ":exit")
                return wrapper
            setattr(cls, name, make(orig, f"{module}{name}"))

    # A round boundary marker, so the samples can be cut into rounds.
    M.REACH.append(lambda: (sample("round_boundary"), {"dram_samples": len(SAMPLES)})[1])


def main():
    install()
    import run_round
    try:
        run_round.main()
    finally:
        with open(OUT, "w") as fh:
            json.dump({"samples": SAMPLES}, fh)
        print(f"dram samples -> {OUT} ({len(SAMPLES)})", flush=True)


if __name__ == "__main__":
    main()
