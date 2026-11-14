"""The work a design ran is counted where it runs, and its output carries a digest, without a card.

`Trunk.launch` / `collect` around a stub trunk call stand in for the device: what is asserted is the
accounting (a pass counts once a design, only when its output is read back, and a call rerun after a
device refusal counts once), which is host code. The fair harness checks the same counts against the
recipe on hardware (`perf/rfd4/fair/run.py::work_mismatch`).
"""

import pytest
import torch

T = pytest.importorskip("tt_bio.rfd4.trunk")
from tt_bio.rfd4.engine import DesignOutput  # noqa: E402


class _Out:
    def __init__(self, b):
        self.shape = (b, 1, 1)

    def is_allocated(self):
        return False


def _trunk(monkeypatch, passes, refusals=0):
    """A Trunk whose call runs `passes` passes of 24 blocks; the first `refusals` calls teach a new plan."""
    tr = object.__new__(T.Trunk)
    tr.residual_dtype = tr.dtype = None
    learned = {"n": 0, "left": refusals}

    def call(s, z, cs, masks):
        tr._ran = [0, 0]
        for _ in range(passes):
            tr._ran[0] += 1
            tr._ran[1] += 24
        if learned["left"]:
            learned["left"] -= 1
            learned["n"] += 1
        return _Out(s.shape[0]), _Out(s.shape[0])

    monkeypatch.setattr(T.Trunk, "__call__", lambda self, *a: call(*a))
    monkeypatch.setattr(T, "refusals_learned", lambda: learned["n"])
    monkeypatch.setattr(T, "upload", lambda t, dtype, device: _Out(t.shape[0]))
    monkeypatch.setattr(T, "download", lambda t: torch.zeros(t.shape))
    monkeypatch.setattr(T.ttnn, "deallocate", lambda t: None)
    tr.masks_for = lambda m, device: None
    tr.pair_dtype = lambda masks: None
    return tr


def _go(tr, b, n=32, depth=3):
    s, z = torch.zeros(b, n, 4), torch.zeros(b, n, n, 2)
    conds = [torch.zeros(b, n, 4)] * depth
    return tr.launch(s, z, conds, torch.ones(b, n), device=None, pair=False)


def test_a_stand_in_trunk_that_never_counted_still_launches_and_collects(monkeypatch):
    tr = _trunk(monkeypatch, passes=0)
    del T.Trunk.__call__  # a stub trunk subclassing nothing of ours: no __call__ of ours ever ran
    T.Trunk.__call__ = lambda self, *a: (_Out(1), _Out(1))
    tr.collect(_go(tr, b=1))
    assert tr.counted == {"passes": 0, "blocks": 0}


def test_a_pass_counts_once_a_design_and_only_when_read_back(monkeypatch):
    tr = _trunk(monkeypatch, passes=3)
    pending = _go(tr, b=4)
    assert tr.counted == {"passes": 0, "blocks": 0}  # enqueued, not read: nothing ran yet as far as a design knows
    tr.collect(pending)
    assert tr.counted == {"passes": 12, "blocks": 288}


def test_a_call_rerun_after_a_refusal_counts_once(monkeypatch):
    tr = _trunk(monkeypatch, passes=3, refusals=1)
    tr.collect(_go(tr, b=2))
    assert tr.counted == {"passes": 6, "blocks": 144}


def test_the_digest_sees_every_design_and_nothing_else():
    g = torch.Generator().manual_seed(0)
    o = DesignOutput(coords_nm=torch.randn(3, 8, 14, 3, generator=g), residue_type=torch.zeros(3, 8, dtype=torch.long),
                     atom_mask=torch.ones(3, 8, 14, dtype=torch.bool), insertion_logits=None)
    d = o.digests()
    assert len(d) == 3 and len(set(d)) == 3 and o.digests() == d
    o.coords_nm[1, 0, 0, 0] += 1e-6
    d2 = o.digests()
    assert d2[0] == d[0] and d2[2] == d[2] and d2[1] != d[1] and o.digest() != DesignOutput(
        coords_nm=o.coords_nm[[0, 2, 1]], residue_type=o.residue_type, atom_mask=o.atom_mask,
        insertion_logits=None).digest()
