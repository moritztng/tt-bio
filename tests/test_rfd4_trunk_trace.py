"""A replayed trunk call is the eager call, and a capture never outlives what it reads — without a card.

Three ways a capture can be replayed against memory that is no longer what it was captured over, each
one found in this lever's own code before a chip ran it (state/rfd4-3x.md, 2026-11-11):

* a trunk that never resolved a lever set has no `_levers` at all;
* `masks_for` frees the mask tensors a capture reads at fixed addresses when a design's padding changes,
  and a trace key is depth + shapes + dtypes, which a different mask of the same shape matches;
* `cleanup()` closes the device, and a capture taken on the old one names a program that is gone.

What a replay must NOT do is change the arithmetic; that is measured on hardware by comparing the fair
harness's per-design sha256 between a traced and an untraced arm, not here.
"""

import pytest
import torch

T = pytest.importorskip("tt_bio.rfd4.trunk")


class _Tensor:
    def __init__(self, shape=(1, 8, 4), dtype="bf16"):
        self.shape = shape
        self.dtype = dtype
        self._alive = True

    def is_allocated(self):
        return self._alive


def _trunk(levers=("trunk_trace",)):
    tr = object.__new__(T.Trunk)
    tr._levers = frozenset(levers)
    return tr


def test_a_trunk_without_a_lever_set_does_not_replay():
    """`object.__new__(Trunk)` has no `_levers`; asking it to replay must not raise."""
    tr = object.__new__(T.Trunk)
    assert tr.traces_wanted() is False
    assert tr.replay(None, _Tensor(), _Tensor(), [], None) is None


def test_a_trunk_wants_traces_only_with_the_lever_and_a_region(monkeypatch):
    monkeypatch.setattr(T, "trace_region_size", lambda: 1 << 20)
    assert _trunk().traces_wanted() is True
    assert _trunk(levers=()).traces_wanted() is False
    monkeypatch.setattr(T, "trace_region_size", lambda: 0)
    assert _trunk().traces_wanted() is False, "no region reserved at open: the trunk stays eager"


def test_the_trace_key_changes_with_the_device_generation(monkeypatch):
    s, z, cs = _Tensor(), _Tensor((1, 8, 8, 2)), [_Tensor(), _Tensor()]
    monkeypatch.setattr(T, "device_generation", lambda: 1)
    first = T.Trunk.trace_key(s, z, cs)
    monkeypatch.setattr(T, "device_generation", lambda: 2)
    assert T.Trunk.trace_key(s, z, cs) != first, "a capture must not outlive the device it was taken on"


def test_the_trace_key_separates_depths_and_shapes(monkeypatch):
    monkeypatch.setattr(T, "device_generation", lambda: 1)
    s, z = _Tensor(), _Tensor((1, 8, 8, 2))
    assert T.Trunk.trace_key(s, z, [_Tensor()]) != T.Trunk.trace_key(s, z, [_Tensor(), _Tensor()])
    assert T.Trunk.trace_key(s, z, []) != T.Trunk.trace_key(_Tensor((1, 16, 4)), z, [])


def test_a_mask_change_releases_the_captures_taken_against_it(monkeypatch):
    """The masks a capture reads are freed when the padding changes, so the captures go first."""
    tr = _trunk()
    tr.dtype = tr.residual_dtype = None
    released, freed = [], []
    tr.release_traces = lambda device: released.append(device)
    monkeypatch.setattr(T.ttnn, "deallocate", lambda t: freed.append(t))
    monkeypatch.setattr(T, "trunk_masks", lambda mask, device, dtype, residual_dtype: (_Tensor(),))

    first = tr.masks_for(torch.ones(1, 8), device="dev")
    assert released == [] and freed == []
    assert tr.masks_for(torch.ones(1, 8), device="dev") is first, "same mask: nothing is rebuilt or freed"
    assert released == [] and freed == []

    second = tr.masks_for(torch.tensor([[1.0] * 6 + [0.0] * 2]), device="dev")
    assert second is not first
    assert released == ["dev"], "the captures were released before the masks they read were freed"
    assert freed == list(first), "and only then are the superseded masks freed"


def test_a_replay_without_a_capture_falls_through_to_the_eager_call(monkeypatch):
    monkeypatch.setattr(T, "trace_region_size", lambda: 1 << 20)
    monkeypatch.setattr(T, "device_generation", lambda: 1)
    tr = _trunk()
    assert tr.replay("dev", _Tensor(), _Tensor((1, 8, 8, 2)), [], None) is None


def test_a_replay_copies_the_step_into_the_captures_buffers_and_restores_the_count(monkeypatch):
    """A replay runs the captured program on new input-buffer contents, and counts that call's work."""
    monkeypatch.setattr(T, "trace_region_size", lambda: 1 << 20)
    monkeypatch.setattr(T, "device_generation", lambda: 1)
    tr = _trunk()
    tr._ran = [0, 0]
    s, z, cs = _Tensor(), _Tensor((1, 8, 8, 2)), [_Tensor()]
    bufs = _Tensor(), _Tensor((1, 8, 8, 2)), [_Tensor()]
    out = (_Tensor(), _Tensor((1, 8, 8, 2)))
    tr.traces[T.Trunk.trace_key(s, z, cs)] = {"tid": 7, "s": bufs[0], "z": bufs[1], "cs": bufs[2],
                                              "out": out, "ran": (3, 72)}
    copied, executed = [], []
    monkeypatch.setattr(T.ttnn, "copy", lambda src, dst: copied.append((src, dst)))
    monkeypatch.setattr(T.ttnn, "execute_trace",
                        lambda dev, tid, cq_id, blocking: executed.append((dev, tid, cq_id, blocking)))

    assert tr.replay("dev", s, z, cs, None) is out, "the capture's own outputs, which collect reads"
    assert copied == [(s, bufs[0]), (z, bufs[1]), (cs[0], bufs[2][0])]
    assert executed == [("dev", 7, 0, False)]
    assert tuple(tr._ran) == (3, 72), "the captured call's work, counted once for this call"
