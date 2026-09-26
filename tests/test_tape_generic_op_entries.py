"""A `generic_op` kernel's tape entry: the gate, the reentrancy, and the two registered VJPs.

`ttnn.generic_op` is opaque to the tape, so an entry has to be per-kernel and a kernel without
one has to keep declining. Three things can go wrong and none of them is loud:

  * an entry that never installs, so the lever reads as a shape the kernel does not cover;
  * a kernel that declines INSIDE its own tape entry, because `ops.taping()` is still True
    while the entry computes its value on raw operands -- the same silent inertness;
  * a registry that installs by default, which makes a release-gated route the shipped one.

Device-free throughout: the kernels are stubs and `ttnn.permute` is recorded, not run.
"""
import pytest

ops = pytest.importorskip("tt_bio.ops")
TT = pytest.importorskip("tt_bio.taped_ttnn")


@pytest.fixture
def taping(monkeypatch):
    """A tape that is open as far as `ops` can tell, with no entries installed."""
    monkeypatch.setattr(ops, "_KERNEL_ENTRIES", {})
    monkeypatch.setattr(ops, "_RAW_DEPTH", 0)
    prev = ops.set_grad_hook(lambda *a, **kw: None)
    yield
    ops.set_grad_hook(prev)


def _kernel(seen):
    @ops.fused_kernel("probe")
    def run(x, flag=None):
        if ops.declines_under_tape("probe"):
            seen.append("declined")
            return None
        seen.append(("served", x, flag))
        return "out"
    return run


def test_without_an_entry_the_kernel_declines_exactly_as_before(taping):
    seen = []
    assert _kernel(seen)(1) is None
    assert seen == ["declined"]


def test_with_no_tape_the_kernel_runs_untouched():
    seen = []
    assert _kernel(seen)(1, flag=2) == "out"
    assert seen == [("served", 1, 2)]


def test_the_entry_runs_the_kernel_with_its_own_gate_lifted(taping):
    seen, inside = [], []

    def entry(shipped, args, kwargs):
        inside.append(ops.taping())
        return shipped(*args, **kwargs)

    ops.set_kernel_entries({"probe": entry})
    assert _kernel(seen)(1, flag=2) == "out"
    # The kernel served rather than declining, and it saw no tape while it did.
    assert seen == [("served", 1, 2)]
    assert inside == [True], "the entry itself is on the tape; only the kernel body is not"


def test_the_gate_is_lifted_for_the_body_and_restored_after(taping):
    ops.set_kernel_entries({"probe": lambda shipped, a, kw: shipped(*a, **kw)})
    seen = []
    _kernel(seen)(1)
    assert ops.taping() is True, "the reentrancy term must not leak past the entry"


def test_declines_under_tape_answers_the_three_cases(taping):
    assert ops.declines_under_tape("probe") is True
    ops.set_kernel_entries({"probe": lambda *a: None})
    assert ops.declines_under_tape("probe") is False
    assert ops.declines_under_tape("other") is True


def test_nothing_installs_by_default(monkeypatch):
    """The release gate. Empty is byte-identical to the behaviour before the registry."""
    monkeypatch.delenv("TT_BIO_TAPED_KERNELS", raising=False)
    assert TT.enabled_kernels() == {}
    monkeypatch.setenv("TT_BIO_TAPED_KERNELS", "all")
    assert set(TT.enabled_kernels()) == set(TT.KERNELS)
    monkeypatch.setenv("TT_BIO_TAPED_KERNELS", "reblock_permute")
    assert set(TT.enabled_kernels()) == {"reblock_permute"}


def test_an_unknown_kernel_name_raises_instead_of_measuring_nothing(monkeypatch):
    monkeypatch.setenv("TT_BIO_TAPED_KERNELS", "reblock_permutte")
    with pytest.raises(ValueError, match="reblock_permutte"):
        TT.enabled_kernels()


def test_the_registered_entries_are_the_three_this_row_built():
    assert set(TT.KERNELS) == {"reblock_permute", "reblock_permute_back", "tri_att_sdpa_hifi"}


def test_the_channel_moves_vjp_is_the_inverse_permutation(monkeypatch, taping):
    """`[1, N, N, C] -> [1, C, N, N]` and back, so the two VJPs are each other's forward."""
    ag = pytest.importorskip("tt_bio.autograd")
    recorded = []
    monkeypatch.setattr(TT.ttnn, "permute", lambda g, dims: recorded.append(dims) or "g")
    monkeypatch.setattr(ag, "_tape", lambda out, parents, make, **kw: (out, make(), kw))

    for name, dims in (("reblock_permute", (0, 2, 3, 1)),
                       ("reblock_permute_back", (0, 3, 1, 2))):
        entry = TT.KERNELS[name]
        out, bw, kw = entry(lambda *a, **k: "moved", (_Parent(),), {})
        assert out == "moved"
        assert kw == {"reads": ()}, "the closure reads neither operand nor output"
        bw("cotangent")
    assert recorded == [(0, 2, 3, 1), (0, 3, 1, 2)]


class _Parent:
    """Something `_wrap` hands back unchanged and that records its cotangent."""
    requires_grad = False
    value = None

    def add_grad(self, g):
        self.got = g


def test_a_declining_kernel_returns_none_so_the_caller_falls_through(taping):
    """The fused-HiFi arm declines on a shape it does not cover; that is not the tape's business."""
    entry = TT.KERNELS["tri_att_sdpa_hifi"]
    before = list(TT.KERNEL_STATS.get("tri_att_sdpa_hifi", [0, 0]))
    assert entry(lambda *a, **kw: None, (None, None, None, None, 1.0), {}) is None
    after = TT.KERNEL_STATS["tri_att_sdpa_hifi"]
    assert after[1] == before[1] + 1, "a decline is counted as a decline, not as a serve"
