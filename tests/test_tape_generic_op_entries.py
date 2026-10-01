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


def test_the_registry_holds_exactly_the_kernels_that_declare_an_entry():
    """Pinned on purpose: a kernel that registers no entry declines under every tape, silently,
    and reads in an A/B as a lever that measured nothing."""
    assert set(TT.KERNELS) == {"reblock_permute", "reblock_permute_back", "tri_att_sdpa_hifi",
                               "rne_add", "reblock_permute_gated"}


def test_the_wide_adds_vjp_is_the_cotangent_to_both_operands(monkeypatch, taping):
    """`rne_add` replaces four taped nodes with one, and the four were each an identity on a
    cotangent: two typecasts pass it through and an add sends it to both parents unchanged."""
    ag = pytest.importorskip("tt_bio.autograd")
    monkeypatch.setattr(ag, "_tape", lambda out, parents, make, **kw: (out, parents, make(), kw))
    # `_wrap` would put a real autograd.Tensor in front of each operand and the recorded
    # cotangent would land there instead of here.
    monkeypatch.setattr(TT, "_wrap", lambda t: t)

    # The MSA track's own disagreement: an [1, S, N, C] activation and an [S, N, C] update.
    a, b = _Parent([1, 2, 288, 256]), _Parent([2, 288, 256])
    reduced = []
    monkeypatch.setattr(TT, "_reduce_to", lambda g, shape: reduced.append(shape) or g)
    out, parents, bw, kw = TT.KERNELS["rne_add"](lambda *x, **k: "summed", (a, b), {})
    assert out == "summed"
    assert parents == [a, b]
    assert kw == {"reads": ()}, "a sum's gradient is a function of the cotangent alone"
    bw("cotangent")
    assert (a.got, b.got) == ("cotangent", "cotangent")
    assert reduced == [[1, 2, 288, 256], [2, 288, 256]], \
        "each operand's cotangent is reshaped to that operand's own shape"


def test_the_wide_add_declines_under_a_tape_with_no_entry_installed(monkeypatch, taping):
    """Without the entry the kernel would cut the graph, so its own gate has to refuse."""
    K = pytest.importorskip("tt_bio.rne_add")
    ops = pytest.importorskip("tt_bio.ops")
    prev = ops.set_kernel_entries({})
    try:
        assert ops.declines_under_tape("rne_add") is True
        ops.set_kernel_entries({"rne_add": TT.KERNELS["rne_add"]})
        assert ops.declines_under_tape("rne_add") is False
    finally:
        ops.set_kernel_entries(prev)
    assert K.rne_add.tape_entry_name == "rne_add"


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

    def __init__(self, shape=None):
        self.shape = shape or [1, 1, 32, 32]

    def add_grad(self, g):
        self.got = g


def test_a_declining_kernel_returns_none_so_the_caller_falls_through(taping):
    """The fused-HiFi arm declines on a shape it does not cover; that is not the tape's business."""
    entry = TT.KERNELS["tri_att_sdpa_hifi"]
    before = list(TT.KERNEL_STATS.get("tri_att_sdpa_hifi", [0, 0]))
    assert entry(lambda *a, **kw: None, (None, None, None, None, 1.0), {}) is None
    after = TT.KERNEL_STATS["tri_att_sdpa_hifi"]
    assert after[1] == before[1] + 1, "a decline is counted as a decline, not as a serve"


class _Projection(_Parent):
    """The four-way in-projection: a value with a shape, recording its slice gradients."""
    requires_grad = True

    def __init__(self, shape):
        super().__init__(shape)
        self.value = self
        self.slices = []

    def add_grad_slice(self, g, starts, ends):
        self.slices.append((g, starts, ends))


@pytest.mark.parametrize("fused", [True, False])
def test_the_gated_moves_vjp_lands_both_gradients_on_their_own_slices(monkeypatch, taping, fused):
    """`reblock_permute_gated` reads two slices of the projection, so its VJP is two slice
    gradients of that one parent -- dp on the value slice, dg on the gate slice -- and the closure
    keeps the projection (`reads=(0,)`) because p and g are re-read from it. Both arms of the
    backward land in the same places; which one ran is the module switch's business."""
    ag = pytest.importorskip("tt_bio.autograd")
    R = pytest.importorskip("tt_bio.reblock_permute")
    monkeypatch.setattr(ag, "_tape", lambda out, parents, make, **kw: (out, parents, make(), kw))
    monkeypatch.setattr(TT, "_wrap", lambda t: t)
    monkeypatch.setattr(R, "GATED_BW_FUSED", fused)
    monkeypatch.setattr(R, "eligible_gated_bw", lambda g, x: True)
    monkeypatch.setattr(R, "reblock_permute_gated_bw", lambda g, x, po, go: ("dp*", "dg*"))
    # The composed arm on stock verbs, recorded rather than run.
    calls = []
    for verb in ("permute", "slice", "sigmoid", "multiply", "rsub"):
        monkeypatch.setattr(TT.ttnn, verb,
                            lambda *a, _v=verb, **k: calls.append(_v) or f"{_v}{len(calls)}")
    xw = _Projection([1, 64, 64, 512])
    out, parents, bw, kw = TT.KERNELS["reblock_permute_gated"](
        lambda *a, **k: "moved", (xw, 256, 384, 128), {"memory_config": None})
    assert (out, parents, kw) == ("moved", [xw], {"reads": (0,)})
    bw("da")
    assert [s[1:] for s in xw.slices] == [([0, 0, 0, 256], [1, 64, 64, 384]),
                                          ([0, 0, 0, 384], [1, 64, 64, 512])]
    if fused:
        assert [s[0] for s in xw.slices] == ["dp*", "dg*"] and calls == []
    else:
        assert calls.count("multiply") == 4 and calls.count("slice") == 2


def test_the_gated_move_refuses_a_row_block_under_a_tape(monkeypatch, taping):
    """A row block writes a caller-owned destination across calls and no one tape node owns it."""
    R = pytest.importorskip("tt_bio.reblock_permute")
    monkeypatch.setattr(ops, "_KERNEL_ENTRIES", {"reblock_permute_gated": object()})
    blk = type("Blk", (), {"shape": [1, 64, 288, 512]})()
    before = dict(R.REJECTS)
    assert R.eligible_gated(blk, 128, None) is False
    key = ("gated_rowblock_taped", (1, 64, 288, 512))
    assert R.REJECTS.get(key, 0) == before.get(key, 0) + 1


def test_nograd_is_inference_only_when_armed(monkeypatch):
    """`ops.recording()` under `no_grad`: True as before with the switch off, False with it on;
    `taping()` is True in both; a recording forward records either way. A `fused_kernel` in an
    unrecorded forward gets raw operands and runs with its tape gate lifted."""
    from tt_bio import autograd as ag, ops
    seen = []

    @ops.fused_kernel("test_probe_kernel")
    def probe(x):
        seen.append((type(x), ops.taping()))
        return x

    prev = ops.set_grad_hook(ag._hook)
    try:
        for armed in (False, True):
            monkeypatch.setattr(ops, "NOGRAD_IS_INFERENCE", armed)
            assert ops.recording()
            with ag.no_grad():
                assert ops.taping()
                assert ops.recording() is not armed
                seen.clear()
                class Raw:
                    pass
                probe(ag.Tensor(Raw()))
                assert seen == [(Raw if armed else ag.Tensor, not armed)]
            assert ops.recording()
    finally:
        ops.set_grad_hook(prev)
    assert not ops.taping() and not ops.recording()
