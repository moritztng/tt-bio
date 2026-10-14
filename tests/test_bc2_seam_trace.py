"""`bindcraft2._SeamTraces` without a card: eager until an eager forward and backward ran under a
key, captured on the next forward, replayed after, and every trace dropped by `release`. A stub stands in for ttnn and
records what reached the device."""
import types

import pytest

torch = pytest.importorskip("torch")
from tt_bio import bindcraft2  # noqa: E402


class _Dev:
    pass


class _T:
    def __init__(self, shape, tag):
        self.shape, self.tag = tuple(shape), tag
        self.value = self


def _ttnn(log):
    tids = iter(range(100))

    def from_torch(t, layout=None, device=None, dtype=None):
        log.append(("up" if device is not None else "host", tuple(t.shape)))
        return _T(t.shape, "dev" if device is not None else "host")

    return types.SimpleNamespace(
        TILE_LAYOUT="tile", ROW_MAJOR_LAYOUT="rm", bfloat16="bf16",
        from_torch=from_torch,
        clone=lambda x: _T(x.shape, "clone"),
        deallocate=lambda x: None,
        to_layout=lambda x, layout: (log.append(("tilize", x.shape)), _T(x.shape, "tiled"))[1],
        begin_trace_capture=lambda dev, cq_id=0: (log.append(("begin",)), next(tids))[1],
        end_trace_capture=lambda dev, tid, cq_id=0: log.append(("end", tid)),
        execute_trace=lambda dev, tid, cq_id=0, blocking=False: log.append(("execute", tid)),
        release_trace=lambda dev, tid: log.append(("release", tid)),
        copy_host_to_device_tensor=lambda h, d: log.append(("copy", d.shape)),
    )


def _trunk(log):
    ttnn = _ttnn(log)
    trunk = types.SimpleNamespace(ttnn=ttnn, device=_Dev(), sync=lambda: None)
    trunk.up = lambda t, tile_on_card=False: ttnn.from_torch(t.unsqueeze(0), device=trunk.device)
    trunk.seed = lambda g, like: ttnn.from_torch(g, device=trunk.device)

    def backward(roots, seeds):
        log.append(("backward", len(roots)))
        for leaf in built["leaves"]:
            leaf.grad = _T(leaf.shape, "grad")

    built = {}
    trunk.ag = types.SimpleNamespace(backward=backward)
    return trunk, built


def test_eager_until_a_backward_ran_then_captures_and_replays():
    log = []
    trunk, built = _trunk(log)
    seams = bindcraft2._SeamTraces()
    host = [(torch.zeros(4, 4), False), (torch.ones(4, 1), True)]
    runs = []

    def build(ins):
        runs.append([i.tag for i in ins])
        leaf = types.SimpleNamespace(shape=ins[0].shape, grad=None)
        built["leaves"] = [leaf]
        return (_T(ins[0].shape, "root"),), (leaf,)

    assert seams.forward("k", trunk, host, build) is None          # first sighting: eager
    assert seams.take_pending() == "k" and seams.take_pending() is None
    assert seams.forward("k", trunk, host, build) is None          # a recycle, still no backward
    seams.warm(seams.take_pending())                                # the eager backward ran
    assert not log
    outs = seams.forward("k", trunk, host, build)                   # now: capture + execute
    assert [o.tag for o in outs] == ["clone"]
    assert runs == [["dev", "tiled"]]                               # the column mask tiles inside
    assert ("begin",) in log and ("execute", 0) in log
    log.clear()
    assert seams.forward("k", trunk, host, build) is outs           # replay: copy in, no rebuild
    assert runs == [["dev", "tiled"]]
    assert [e[0] for e in log] == ["host", "copy", "host", "copy", "execute"]

    log.clear()
    grads = seams.backward("k", trunk, [torch.zeros(4, 4)])        # backward capture
    assert [g.tag for g in grads] == ["grad"]
    assert ("backward", 1) in log and ("execute", 1) in log
    log.clear()
    assert seams.backward("k", trunk, [torch.zeros(4, 4)]) is grads
    assert ("backward", 1) not in log and log[-1] == ("execute", 1)

    log.clear()
    seams.release(trunk.device)
    assert log == [("release", 0), ("release", 1)]
    assert seams.forward("k", trunk, host, build) is None           # starts over at eager


def test_off_by_default_and_never_for_offload_or_interleave(monkeypatch):
    assert bindcraft2.SEAM_TRACE is False
    monkeypatch.setattr(bindcraft2, "SEAM_TRACE", True)
    assert bindcraft2._SeamTraces.wanted("", "fast")
    assert not bindcraft2._SeamTraces.wanted("", "offload")
    assert not bindcraft2._SeamTraces.wanted("t1", "fast")


def test_dropout_host_round_trips_upload_dropout():
    np = pytest.importorskip("numpy")
    draws = bindcraft2.DROPOUT_DRAWS[1:]
    keep = np.ones((2, len(draws), 30, 8), dtype=np.float32)
    host = bindcraft2.dropout_host(draws, keep, np.full(len(draws), 1.25), 32)
    assert len(host) == 2 and all(t.shape[-2] in (32, 1) or t.shape[0] in (32, 1)
                                  for b in host for t, _ in b.values())
    seen = []
    trunk = types.SimpleNamespace(up=lambda t, tile_on_card=False: seen.append((tuple(t.shape),
                                                                                tile_on_card)))
    bindcraft2.upload_dropout(trunk, draws, keep, np.full(len(draws), 1.25), 32)
    assert seen == [(tuple(t.shape), col) for b in host for t, col in b.values()]
    assert bindcraft2.dropout_host(draws, None, [], 32) is None
