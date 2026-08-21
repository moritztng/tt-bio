"""RF3's three lever defaults, and the one that must stay off.

All three were exported by the pass-9 perf ladder (`perf/rf3/p9_qb1.sh`) and all three were
default-OFF in the shipped code, which is why RF3's ladder cells and its perf-page cell
disagreed by 3.24x. Two of them then passed a whole-fold accuracy screen and are on; one
failed it badly and is not. That split is the thing this file pins, because it is a perf lever
worth 2.107x of a 512 aa fold and the next perf pass will want to flip it.

Host-only: no device, no fold. The live halves are `perf/rf3/gln_1024_check.py` (the OOM) and
`perf/rf3/fold_fix_ab.py` (the RMSD).
"""
from __future__ import annotations

import inspect

import pytest

from tt_bio import tenstorrent as T
from tt_bio.rf3 import confidence_head as CH
from tt_bio.rf3 import remap as R


# --- the lever that must stay OFF ------------------------------------------------------

def test_rf3_does_not_opt_into_the_fused_triangle_attention():
    """It costs 1.9335 A CA RMSD on a whole cdk2x2_298 fold, so it is not a default.

    Measured: 2.1833 A all-atom, max |dxyz| 17.03 A over 2397 atoms, plDDT 81.7155 -> 78.8675,
    pTM 0.9005 -> 0.8338. An op-level fidelity screen had graded the same path 1.25-2.49x
    CLOSER to fp64 than the materialised softmax it replaces above S=128, so per-op fidelity
    is not evidence for this flip and re-quoting it is not grounds to remove this test.
    """
    assert "triatt_fused_hifi" not in R.PAIRFORMER_FLAGS


def test_the_fused_path_is_reachable_but_only_deliberately():
    """Rejected as a default, kept as a lever: the seam has to still exist."""
    sig = inspect.signature(T.TriangleAttention.__init__)
    assert "fused_hifi" in sig.parameters
    # None, not False: unset must fall back to TT_BIO_TRIATT_FUSED_HIFI so the env flag keeps
    # switching every model at once, which is what the perf ladder scripts rely on.
    assert sig.parameters["fused_hifi"].default is None
    for cls in (T.PairformerLayer, T.Pairformer):
        p = inspect.signature(cls.__init__).parameters
        assert "triatt_fused_hifi" in p, cls.__name__
        assert p["triatt_fused_hifi"].default is None, cls.__name__


# --- the confidence head's global layer norm, which is ON because of a hard OOM ---------

def test_the_global_norm_row_fold_is_on_by_default():
    """Not a perf nicety: the alternative cannot allocate at 1024 aa."""
    assert CH._GLN_ROW_FOLD is True


def test_the_one_row_flatten_asks_for_32x_the_tensor():
    """The arithmetic behind that default, so the reason survives as a number.

    `global_layer_norm`'s off-path reshapes to (1, 1, 1, n) and TILE_LAYOUT pads that single
    row up to a full 32-row tile. At 1024 aa the pair rep is [1, 1024, 1024, 128] bf16, and
    the allocator was measured asking for exactly this and failing with TT_FATAL: Out of
    Memory across 8 banks (perf/rf3/accuracy/gln_1024_check.json).
    """
    pair_bytes = 1024 * 1024 * 128 * 2
    assert pair_bytes == 268_435_456
    assert pair_bytes * 32 == 8_589_934_592


def test_the_row_fold_keeps_an_escape_hatch(monkeypatch):
    """Non-bit-exact against the flatten, so it must stay switchable off."""
    import importlib
    monkeypatch.setenv("TT_BIO_RF3_GLN_ROW_FOLD", "0")
    assert importlib.reload(CH)._GLN_ROW_FOLD is False
    monkeypatch.delenv("TT_BIO_RF3_GLN_ROW_FOLD")
    assert importlib.reload(CH)._GLN_ROW_FOLD is True


# --- the outer product's small-depth reassociation, ON but self-limiting ----------------

def test_the_outer_product_small_depth_seam_exists_and_defaults_to_the_env():
    sig = inspect.signature(T.OuterProductMean.__init__)
    assert "small_depth" in sig.parameters
    assert sig.parameters["small_depth"].default is None


def test_the_small_depth_gate_stays_at_eight_rows():
    """Break-even is ~9 rows: 3.03 ms/row reassociated against 27.43 ms materialised per
    block at 512 aa. `state/rf3-perf.md`'s "break-even at MSA depth ~32" is the estimate that
    was wrong, not the gate, so raising this would make deep MSAs slower, not faster.
    """
    assert T.OPM_SMALL_DEPTH_MAX == 8


@pytest.mark.parametrize("depth", [1, 4, 8])
def test_it_fires_on_shallow_msas(depth):
    assert depth <= T.OPM_SMALL_DEPTH_MAX


@pytest.mark.parametrize("depth", [9, 35, 128])
def test_it_declines_on_the_published_workload(depth):
    """The perf page's fixture carries a 35-row a3m, where this lever served 0 of 80 calls,
    so it cannot regress the published number in either direction."""
    assert depth > T.OPM_SMALL_DEPTH_MAX
