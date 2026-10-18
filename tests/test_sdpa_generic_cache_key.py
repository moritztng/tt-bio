"""sdpa_generic's program cache: two calls that compile differently never share a program.

Opens no device: `build` and `generic_op` are stubbed, so what runs is the cache lookup.
"""
import types

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import sdpa_generic as SG                                       # noqa: E402


class _T:
    """A stand-in tensor with the attributes the cache key and the address check read."""
    padded_shape, dtype = (736, 4, 736, 32), "BFLOAT16"

    def buffer_address(self):
        return 0

    def memory_config(self):
        return types.SimpleNamespace(buffer_type="DRAM")


def test_a_different_scale_is_a_different_program(monkeypatch):
    built = []
    monkeypatch.setattr(SG, "_CACHE", {})
    monkeypatch.setattr(SG, "build", lambda *a, **k: built.append(a[-1]) or {"addrs": (0,) * 5, "pd": None})
    monkeypatch.setattr(SG.ttnn, "generic_op", lambda *a, **k: None)
    t = _T()
    for scale in (0.25, 32 ** -0.5, 0.25):
        SG.sdpa(None, t, t, t, t, t, 256, 736, (8, 8), (1, 0, 1, 0), scale)
    assert built == [0.25, 32 ** -0.5], "the head_dim-32 call reused the head_dim-16 program"


# ------------------------------------------------------------------ descriptors by address set

import collections                                                          # noqa: E402
import copy                                                                 # noqa: E402

from tt_bio import mm_generic as MG                                         # noqa: E402


class _At(_T):
    """A stand-in tensor at a chosen buffer address."""

    def __init__(self, addr):
        self.addr = addr

    def buffer_address(self):
        return self.addr


class _PD:
    """A descriptor that snapshots the runtime args it was built from, as the binding copies them."""

    def __init__(self, kernels, semaphores, cbs):
        self.args = [copy.deepcopy(k.runtime_args) for k in kernels]


def _fresh_bound(monkeypatch):
    monkeypatch.setattr(MG, "_BOUND", collections.OrderedDict())
    monkeypatch.setattr(MG, "BOUND_STATS", collections.Counter())


def test_sdpa_reuses_the_descriptor_it_built_for_an_address_set_it_has_seen(monkeypatch):
    """A, B, A, B: two rebuilds, and every launch carries exactly its own call's addresses."""
    _fresh_bound(monkeypatch)
    monkeypatch.setattr(SG, "_CACHE", {})
    k = [types.SimpleNamespace(runtime_args=None) for _ in range(3)]
    entry = {"addrs": (0, 1, 2, 3, 4), "pd": "built", "kernels": k, "semaphores": [], "cbs": [],
             "rt": ([(None, [0, 0, 0, 0, 7])], [(None, [0, 7])], []), "fuse": None, "gate": None}
    monkeypatch.setattr(SG, "build", lambda *a, **kw: entry)
    monkeypatch.setattr(SG.ttnn, "ProgramDescriptor", _PD)
    launched = []
    monkeypatch.setattr(SG.ttnn, "generic_op", lambda ops, pd: launched.append(pd))
    for base in (100, 200, 100, 200, 0):
        q, kk, v, m, o = (_At(base + i) for i in range(5))
        SG.sdpa(None, q, kk, v, m, o, 256, 736, (8, 8), (1, 0, 1, 0), 0.25)
    assert launched[0] is launched[2] and launched[1] is launched[3], "an address set rebuilt"
    assert launched[4] == "built", "the descriptor `build` made was not kept for its addresses"
    for base, pd in zip((100, 200, 100, 200), launched):
        reader, writer = pd.args[0][0][1], pd.args[1][0][1]
        assert reader == [base, base + 1, base + 2, base + 3, 7] and writer == [base + 4, 7]
    assert MG.BOUND_STATS == {"miss": 2, "hit": 3}


def test_matmul_reuses_its_descriptor_and_never_lends_it_to_another_program(monkeypatch):
    _fresh_bound(monkeypatch)
    monkeypatch.setattr(MG, "_CACHE", {})
    monkeypatch.setattr(MG.dest_guard, "descriptor", lambda c, ckc, guard: c)
    entries = []

    def build(device, in0, in1, outs, *a):
        e = {"addrs": (in0.addr, in1.addr, tuple(o.addr for o in outs)), "pd": ("built", in0.padded_shape)}
        entries.append(e)
        return e

    rebuilt = []

    def rebind(entry, a0, a1, outs):
        entry["pd"] = ("rebound", a0, a1, outs, len(rebuilt))
        entry["addrs"] = (a0, a1, outs)
        rebuilt.append(entry["pd"])

    monkeypatch.setattr(MG, "build", build)
    monkeypatch.setattr(MG, "rebind", rebind)
    launched = []
    monkeypatch.setattr(MG.ttnn, "generic_op", lambda ops, pd: launched.append(pd))

    class _Wide(_At):
        padded_shape = (1, 1, 64, 64)

    for cls, a in ((_At, 1), (_At, 2), (_At, 1), (_Wide, 2), (_At, 2)):
        MG.generic_minimal_matmul(None, cls(a), cls(a + 10), cls(a + 20), (1, 1), ())
    assert len(entries) == 2 and len(rebuilt) == 1
    assert launched[2] == ("built", _At.padded_shape), "the first address set was rebuilt"
    assert launched[3][0] == "built" and launched[3][1] == _Wide.padded_shape, \
        "a second program at the same addresses was handed the first program's descriptor"
    assert launched[4] is launched[1]
