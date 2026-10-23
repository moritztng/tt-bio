"""A process's first RFD4 design must equal its repeats (state/rfd4/bets/cells.md).

At 256 tokens on a p300c the first trunk call's first trimul ran from L1 and its second clashed,
which moved the shape to the DRAM path for good: that one call mixed two plans, and the first
design of every process landed 2.65 A from the warm ones at the same seed. The trunk now re-runs a
call during which a refusal taught the process a new plan. These tests pin that rule card-free.
"""
import pytest

pytest.importorskip("ttnn", reason="tenstorrent.py imports ttnn at module scope")

import torch  # noqa: E402

from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio.rfd4 import trunk as trunk_mod  # noqa: E402

CLASH = ("Statically allocated circular buffers in program 1060 clash with L1 buffers on core range "
         "[(x=0,y=0) - (x=7,y=8)]. L1 buffer allocated at 352256 and static circular buffer region "
         "ends at 382240")


def test_a_recorded_clash_counts_as_a_learned_plan():
    before = T.refusals_learned()
    T.note_l1_clash("test", CLASH)
    assert T.refusals_learned() == before + 1


class _Out:
    def __init__(self, tag):
        self.tag, self.freed = tag, False

    def is_allocated(self):
        return not self.freed

    def buffer_address(self):
        return id(self)


class _Trunk(trunk_mod.Trunk):
    """The real `launch`, with the device forward replaced by one that learns a plan on its first call."""

    def __init__(self, learn_on):
        self.dtype = self.residual_dtype = self.pair_residual_dtype = None
        self.learn_on, self.calls, self.outs = set(learn_on), 0, []

    def masks_for(self, mask, device):
        return [None] * 6

    def __call__(self, s, z, cs, masks):
        self.calls += 1
        if self.calls in self.learn_on:
            T.learned_from_refusal()
        out = (_Out(("s", self.calls)), _Out(("z", self.calls)))
        self.outs.append(out)
        return out


@pytest.fixture
def fake_device(monkeypatch):
    monkeypatch.setattr(trunk_mod, "upload", lambda t, dtype, device: _Out("input"))
    monkeypatch.setattr(trunk_mod.ttnn, "deallocate", lambda t: setattr(t, "freed", True))
    monkeypatch.setattr(trunk_mod, "download_later", lambda t: lambda: t)


def _launch(tr):
    n = 32
    return tr.launch(torch.zeros(1, n, 4), torch.zeros(1, n, n, 4), [torch.zeros(1, n, 4)],
                     torch.ones(1, n), device=None)


def test_a_call_that_learned_a_plan_is_run_again(fake_device):
    tr = _Trunk(learn_on={1})
    _, z_out, s_host = _launch(tr)
    assert tr.calls == 2
    assert s_host().tag == ("s", 2) and z_out.tag == ("z", 2)
    assert all(t.freed for t in tr.outs[0])          # the mixed-plan result is released


def test_a_settled_call_runs_once(fake_device):
    tr = _Trunk(learn_on=set())
    _launch(tr)
    assert tr.calls == 1


def test_a_count_that_never_settles_is_capped(fake_device):
    tr = _Trunk(learn_on=set(range(1, 100)))
    _, _, s_host = _launch(tr)
    assert tr.calls == 3 and s_host().tag == ("s", 3)


def test_launch_frees_all_but_the_pair_it_reads(fake_device):
    tr = _Trunk(learn_on=set())
    _, z_out, _ = _launch(tr)
    s_out, z = tr.outs[0]
    assert s_out.freed and z is z_out and not z.freed  # s is read back already; z waits for collect
    _, z_out, _ = tr.launch(torch.zeros(1, 32, 4), torch.zeros(1, 32, 32, 4), [torch.zeros(1, 32, 4)],
                            torch.ones(1, 32), device=None, pair=False)
    assert z_out is None and all(t.freed for t in tr.outs[1])
