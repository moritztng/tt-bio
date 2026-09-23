"""Matmul calibration must DECLINE when its scratch does not fit, and never hold both screens.

Calibration is scratch: a reference output per screen plus a full random copy of each operand,
all live alongside the model's own working set. Holding the random screen's reference and the
live screen's reference at the same time is what ended the first 1536-residue rfd3 design -- the
throw was at `rref`, asking for a 2717908992 B DRAM buffer with 2.45 GB free on the card
(j10glx02 card 30, 2026-09-23, mgx-design-ceiling). A shape too big to calibrate is not a shape
too big to RUN: the configs calibration returns are bitwise equal to the default it falls back
to, so declining costs speed and can never change a result.

Host-only. The device calls are faked, because what is under test is the control flow around
them: which tensors are alive when, and what happens when one of them refuses to allocate.
"""
from __future__ import annotations

import pytest
import ttnn

from tt_bio.rfd3 import model as M


class FakeTensor:
    def __init__(self, name, live):
        self.name = name
        self.alive = True
        self.padded_shape = (1, 32, 32)
        live.append(self)

    def __repr__(self):
        return f"<{self.name} {'alive' if self.alive else 'dead'}>"


class Fixture:
    """Enough of ttnn to run `_calibrate_linear`, with an allocation that can be made to refuse."""

    def __init__(self, refuse=None, times=None):
        self.refuse = refuse            # "rref" / "ref" / "timed", or None: which allocation raises
        self.times = list(times or [])  # `_mm_time` results, last one repeats
        self.live = []
        self.overlaps = []              # reference-output names alive together, per allocation
        self.timing = False             # inside `_mm_time`, where every output is transient

    # -- the pieces model.py calls ------------------------------------------------------------
    def linear(self, a, b, **kw):
        if self.timing and self.refuse == "timed":
            raise RuntimeError("Out of Memory: Not enough space to allocate the timed default")
        kept = "program_config" not in kw and not self.timing
        name = {"rx": "rref", "x": "ref"}.get(a.name, "cand") if kept else "cand"
        if kept:
            if name == self.refuse:
                raise RuntimeError(f"Out of Memory: Not enough space to allocate {name}")
            self.overlaps.append(sorted(t.name for t in self.live
                                        if t.alive and t.name in ("ref", "rref")) + [name])
        return FakeTensor(name, self.live)

    def deallocate(self, t):
        t.alive = False

    def random_like(self, t, seed):
        return FakeTensor("rx" if seed == 0 else "rw", self.live)

    def time(self, fn):
        self.timing = True
        try:
            fn()
        finally:
            self.timing = False
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]

    def candidates(self, x, w, grid):
        yield "pc-a"
        yield "pc-b"

    def install(self, mp):
        mp.setattr(ttnn, "linear", self.linear)
        mp.setattr(ttnn, "deallocate", self.deallocate)
        mp.setattr(M, "_mm_random_like", self.random_like)
        mp.setattr(M, "_mm_time", self.time)
        mp.setattr(M, "_mm_candidates", self.candidates)
        mp.setattr(M, "_mm_maxabs", lambda a, b: 0.0)
        mp.setattr(M, "get_device", lambda: type("D", (), {
            "compute_with_storage_grid_size": lambda self: None})())
        return self

    def leaked(self):
        """Calibration's own scratch, still alive. `x`/`w` are the caller's and a candidate
        output is a Python temporary that real ttnn frees on refcount, so neither is a leak."""
        return [t for t in self.live if t.alive and t.name in ("rx", "rw", "rref", "ref")]


def calibrate(fx, x_name="x"):
    return M._calibrate_linear(FakeTensor(x_name, fx.live), FakeTensor("w", fx.live), {}, None)


def test_the_two_screens_never_hold_both_reference_outputs(monkeypatch):
    """`ref` and `rref` are each a full output. The peak is |out| + |x| + |w|, not 2|out| + ..."""
    fx = Fixture(times=[0.010, 0.001]).install(monkeypatch)
    calibrate(fx)
    assert ["rref"] in fx.overlaps and ["ref"] in fx.overlaps
    assert not any(len(o) > 1 for o in fx.overlaps), fx.overlaps


def test_a_random_screen_that_does_not_fit_declines_and_frees_what_it_built(monkeypatch):
    fx = Fixture(refuse="rref", times=[0.010]).install(monkeypatch)
    assert calibrate(fx) is None
    assert fx.leaked() == []


def test_a_live_screen_that_does_not_fit_declines(monkeypatch):
    """The random screen can fit and the live one still not: the model grows between them."""
    fx = Fixture(refuse="ref", times=[0.010]).install(monkeypatch)
    assert calibrate(fx) is None
    assert fx.leaked() == []


def test_a_shape_under_the_time_floor_allocates_nothing(monkeypatch):
    fx = Fixture(times=[M._TUNE_MIN_MS / 1e3 / 2]).install(monkeypatch)
    assert calibrate(fx) is None
    assert [t.name for t in fx.live if t.name not in ("x", "w", "cand")] == []


def test_a_candidate_that_beats_the_budget_is_still_chosen(monkeypatch):
    """The decline paths must not cost the win: a faster exact candidate still comes back."""
    fx = Fixture(times=[0.010, 0.001]).install(monkeypatch)
    assert calibrate(fx) == "pc-a"
    assert fx.leaked() == []


def test_no_survivor_from_the_random_screen_means_the_live_screen_allocates_nothing(monkeypatch):
    """An empty survivor list is the signal that DRAM is tight, so do not allocate one more output.

    Screen 2 only ever iterates the survivors, so with none the answer is already DEFAULT and
    `ref` would be a full output built for an empty loop. That is not hypothetical at the sizes
    this file is about: at 1536 residues every candidate for the 2717908992 B shape refused.
    """
    fx = Fixture(times=[0.010]).install(monkeypatch)
    monkeypatch.setattr(M, "_mm_maxabs", lambda a, b: 1.0)   # nothing is bitwise equal
    assert calibrate(fx) is None
    assert [t.name for t in fx.live if t.name == "ref"] == []
    assert fx.leaked() == []


def test_a_timed_default_that_does_not_fit_PROPAGATES(monkeypatch):
    """The one allocation left unguarded, and it has to stay that way.

    Every other allocation here is calibration's own scratch, so refusing it costs speed and
    nothing else. The timed default is not scratch: it holds ONE output, which is exactly what
    the model's own `ttnn.linear` will hold a moment later, so an allocator refusal on it is the
    MODEL's ceiling. Swallowing it would turn a real wall into a silent skip and the run would
    fail later somewhere less informative -- which is the whole failure mode this file exists to
    stop, pointing the other way.
    """
    fx = Fixture(refuse="timed", times=[0.010]).install(monkeypatch)
    with pytest.raises(RuntimeError, match="Out of Memory"):
        calibrate(fx)
    assert fx.leaked() == []


@pytest.mark.parametrize("refuse", ["rref", "ref"])
def test_declining_never_raises_at_the_call_site(monkeypatch, refuse):
    """`_tuned_linear` treats None as "use the default", so a decline is invisible to the model."""
    fx = Fixture(refuse=refuse, times=[0.010]).install(monkeypatch)
    assert calibrate(fx) is None
