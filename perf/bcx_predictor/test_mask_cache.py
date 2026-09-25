"""Two masks of the same shape must not be confused for each other.

`EvoformerOnDevice` caches the uploaded MSA mask so a fold does not re-upload it every
call. That cache was keyed by shape, on the reasoning that the shape changes when the
binder length does -- which is true, and does not give what the key needs. BindCraft 2
buckets the token axis to 32, so on the shipped `examples/pdl1.json` binder 146 and binder
164 against the same 115-residue target both pad to 288: same shape, different real-residue
count, different mask. Shape-keyed, the first mask to arrive is uploaded once and then
folded into every later call of that shape, with no exception anywhere to say so.

No card and no weights: the cache is host bookkeeping and this exercises it with a stub.
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    if p not in sys.path:
        sys.path.insert(0, p)

splice = pytest.importorskip("splice")


class StubDev:
    """`up` is the only thing the mask cache touches. Each call returns a fresh object, so
    "was this cached" is object identity rather than array equality."""

    device = None

    def __init__(self):
        self.uploads = 0

    def up(self, tensor):
        self.uploads += 1
        return ("uploaded", self.uploads)


def bucketed_mask(real, padded=288):
    """`seq_mask` as BindCraft 2 hands it over: ones for real residues, zeros for the pad."""
    mask = np.zeros((padded,), dtype=np.float32)
    mask[:real] = 1.0
    return mask


def evo():
    return splice.EvoformerOnDevice(StubDev(), k_evo=48)


def test_same_mask_is_uploaded_once():
    e = evo()
    mask = bucketed_mask(261)
    assert e._mask(mask) is e._mask(mask.copy())
    assert e.dev.uploads == 1


def test_two_binder_lengths_that_share_a_padded_shape_do_not_share_a_mask():
    """146 + 115 = 261 and 164 + 115 = 279 both bucket to 288 -- the two draws the first
    shipped-pool run actually made (`pdl1_denovo_l146`, `pdl1_denovo_l164`)."""
    e = evo()
    l146, l164 = bucketed_mask(261), bucketed_mask(279)
    assert l146.shape == l164.shape
    assert e._mask(l146) is not e._mask(l164)
    assert e.dev.uploads == 2
    assert e.mask_shape_collisions["msa"] == 1


def test_a_collision_is_counted_once_per_new_content():
    e = evo()
    for real in (261, 279, 288, 261, 279):
        e._mask(bucketed_mask(real))
    assert e.dev.uploads == 3
    assert e.mask_shape_collisions["msa"] == 2


def test_a_shape_seen_once_is_not_a_collision():
    e = evo()
    e._mask(bucketed_mask(261))
    e._mask(bucketed_mask(146, padded=160))
    assert e.mask_shape_collisions["msa"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
