"""A long curve leg keeps a small resume ring but must be able to replay a loss spike.

G3 leg 2 logged one signed_volume step 400x the median at step 823; its keep-2 policy had
already deleted every checkpoint before it, so the step could not be replayed (D67).
``Checkpointer.pin`` keeps the checkpoint the spike started from out of the ring for good.
"""
import pytest

pytest.importorskip("numpy")
from tt_bio.train import checkpoint as C  # noqa: E402


@pytest.fixture
def ck(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "save_adapter", lambda path, opt, meta=None: path.write_text(str(meta["step"])))
    return C.Checkpointer(tmp_path, every=10, keep=3)


def _on_disk(ck):
    return sorted(int(p.stem.split("-")[1]) for p in ck.dir.glob("adapter-*.safetensors"))


def _run(ck, steps, spike_at=None):
    for s in range(steps):          # step index s starts from checkpoint s, saves s + 1
        if s == spike_at:
            ck.pin(s)
        ck.save(s + 1, opt=None)    # no metric: a ring by step


def test_the_ring_alone_loses_the_checkpoint_a_spike_started_from(ck):
    _run(ck, 100)
    assert _on_disk(ck) == [80, 90, 100]
    assert 20 not in _on_disk(ck)   # step 23's start point is gone: the leg-2 failure


def test_pin_keeps_the_newest_checkpoint_at_or_before_the_spike(ck):
    _run(ck, 100, spike_at=23)
    assert _on_disk(ck) == [20, 80, 90, 100]
    assert [w["step"] for w in ck.pinned] == [20]


def test_a_spike_on_a_save_boundary_pins_the_checkpoint_it_started_from(ck):
    _run(ck, 60, spike_at=30)       # step 30 runs from checkpoint 30, not 20 or 40
    assert [w["step"] for w in ck.pinned] == [30]


def test_pin_before_any_checkpoint_returns_none(ck):
    assert ck.pin(5) is None
    _run(ck, 40)
    assert _on_disk(ck) == [20, 30, 40]
