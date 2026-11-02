"""A recycling cycle DRAM refuses is re-run on inputs that are still allocated.

`Trunk.__call__` re-runs a refused cycle from the pair and single it was given, so the attempt has
to keep them: a cycle frees its input pair as soon as it has read it unless told `retry`. The
device-picked-rows attempt did not pass it, and 8eud-assembly1 / 8k7x-assembly1 (819 and 856 aa
with ligands) threw `Buffer is not allocated` in the re-run's first layer norm on Wormhole.

Host-only: the tensors are stand-ins with an allocated flag, no device is opened.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("ttnn")

from tt_bio import protenix as P  # noqa: E402

REFUSAL = ("Out of Memory: Not enough space to allocate 1610612736 B DRAM buffer across 12 banks, "
           "where each bank needs to store 134217728 B, but bank size is 1073741792 B (allocated: "
           "949108160 B, free: 124633632 B, largest free block: 111896800 B)")


class _T:
    def __init__(self):
        self.alive = True

    def is_allocated(self):
        return self.alive


def _cycle(refuse):
    """A cycle that, like `Trunk.cycle`, frees its input pair unless `retry`, then maybe refuses."""
    def run(carry, retry=False):
        z3, s = carry
        carry.clear()
        if not retry:
            z3.alive = False
        if refuse:
            raise RuntimeError(f"TT_FATAL @ bank_manager.cpp:439: {refuse}")
        return [_T(), _T()]
    return run


def test_refused_cycle_returns_its_inputs_allocated():
    carry = [_T(), _T()]
    z3, s = carry
    out, inputs = P.refusable_cycle(_cycle(REFUSAL), carry)
    assert out is None
    assert inputs[0] is z3 and inputs[1] is s
    assert all(t.is_allocated() for t in inputs)


def test_a_cycle_that_fits_returns_its_carry():
    out, inputs = P.refusable_cycle(_cycle(None), [_T(), _T()])
    assert inputs is None and len(out) == 2


def test_other_errors_propagate():
    with pytest.raises(RuntimeError, match="Buffer is not allocated"):
        P.refusable_cycle(_cycle("Buffer is not allocated"), [_T(), _T()])


def test_trunk_guards_every_cycle_through_it():
    # A hand-rolled guard is how the device-rows attempt came to leave `retry` off.
    src = inspect.getsource(P.Trunk.__call__)
    assert "is_alloc_refusal" not in src
    assert src.count("refusable_cycle(") == 3
