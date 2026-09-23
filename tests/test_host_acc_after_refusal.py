"""A chunked pair op joins its blocks on device first, and on the host only after DRAM refuses.

`concat_host_bytes()` sent every trimul, triangle attention and transition past 1.5 GiB to the
host, block by block. At OpenDDE 1536 x c_z=384 that caught the residue trunk's 1.81 GB pair on a
chip with 4.9 GiB in use, and the host round trips were 93 % of the fold's main-thread samples.

The control-flow tests are host-only: `run` is a plain callable reading `_host_concat` the way the
ops do. The device test folds one pairformer block both ways and compares the bytes.
"""
import pytest
import ttnn

from tt_bio import tenstorrent as T

# Verbatim refusal text, the format `is_alloc_refusal` parses.
REFUSAL = ("Not enough space to allocate 1811939328 B DRAM buffer across 12 banks, where each "
           "bank needs to store 150994944 B, but bank size is 1071644672 B (allocated: "
           "5268045824 B, free: 7591690240 B, largest free block: 69468160 B)")
KEY = ("trimul", (1, 1536, 1536, 384), False)


class _X:
    """Stands in for a bf16 pair tensor past the budget."""
    dtype = ttnn.bfloat16

    def __init__(self, nbytes=1811939328, allocated=True):
        self.nbytes, self.allocated = nbytes, allocated

    def logical_volume(self):
        return self.nbytes // 2

    def is_allocated(self):
        return self.allocated


class _Op:
    """Records which join each call took; refuses the first `refuse` device-joined calls."""

    def __init__(self, x, refuse=0, error=REFUSAL, join_fallback=False):
        self.x, self.refuse, self.error, self.join_fallback = x, refuse, error, join_fallback
        self.joins = []

    def __call__(self):
        host = T._host_concat(self.x)
        self.joins.append("host" if host else "device")
        if not host and self.refuse:
            self.refuse -= 1
            raise RuntimeError(self.error)
        if not host and self.join_fallback:
            T.ACC_CONCAT_HOST_FALLBACKS[0] += 1
        return self.joins[-1]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(T, "_HOST_ACC_KEYS", set())
    monkeypatch.setattr(T, "_CONCAT_HOST_BYTES", T.CONCAT_HOST_BYTES_BASE)
    yield
    assert T._DEVICE_ACC_TRIAL == [False], "the trial flag outlived its call"


def test_under_the_budget_nothing_changes():
    op = _Op(_X(nbytes=T.CONCAT_HOST_BYTES_BASE))
    assert T.host_acc_after_refusal(KEY, op.x, op) == "device"
    assert op.joins == ["device"] and T._HOST_ACC_KEYS == set()


def test_past_the_budget_the_device_join_is_tried_first():
    op = _Op(_X())
    assert T.host_acc_after_refusal(KEY, op.x, op) == "device"
    assert T._HOST_ACC_KEYS == set(), "nothing was refused, so nothing may be remembered"
    assert T._host_concat(op.x), "outside a trial the budget still reads as before"


def test_a_refusal_re_runs_the_call_with_the_host_join_and_remembers_it():
    op = _Op(_X(), refuse=1)
    assert T.host_acc_after_refusal(KEY, op.x, op) == "host"
    assert op.joins == ["device", "host"]
    assert KEY in T._HOST_ACC_KEYS
    op2 = _Op(op.x, refuse=1)
    assert T.host_acc_after_refusal(KEY, op2.x, op2) == "host"
    assert op2.joins == ["host"], "a refused key tried the device join again"


def test_a_host_fallback_at_the_join_is_remembered_without_a_re_run():
    """`_acc_concat` already rescues a refused device concat. The call succeeded, but the next
    one at this key would pay the device loop and the download again."""
    op = _Op(_X(), join_fallback=True)
    assert T.host_acc_after_refusal(KEY, op.x, op) == "device"
    assert op.joins == ["device"] and KEY in T._HOST_ACC_KEYS


def test_keys_are_separate():
    op = _Op(_X(), refuse=1)
    T.host_acc_after_refusal(KEY, op.x, op)
    other = ("trimul", (1, 1536, 1536, 384), True)
    op2 = _Op(op.x)
    assert T.host_acc_after_refusal(other, op2.x, op2) == "device"


def test_a_nested_call_inherits_the_trial():
    x = _X()
    inner = _Op(x)
    outer = lambda: (T._host_concat(x), T.host_acc_after_refusal(("transition", (1,)), x, inner))
    assert T.host_acc_after_refusal(KEY, x, outer) == (False, "device")


def test_anything_that_is_not_an_allocator_refusal_propagates():
    op = _Op(_X(), refuse=1, error="TT_THROW: circular buffer clash")
    with pytest.raises(RuntimeError, match="circular buffer clash"):
        T.host_acc_after_refusal(KEY, op.x, op)
    assert op.joins == ["device"] and T._HOST_ACC_KEYS == set()


def test_a_refusal_after_the_input_was_consumed_propagates():
    """A residual transition frees its input at the join; re-running it would read freed DRAM."""
    op = _Op(_X(allocated=False), refuse=1)
    with pytest.raises(RuntimeError, match="Not enough space"):
        T.host_acc_after_refusal(KEY, op.x, op)
    assert op.joins == ["device"]


@pytest.mark.device
def test_a_pairformer_block_is_the_same_with_either_join(monkeypatch):
    """Every chunked op of one block, the budget at 0 so all of them are past it: the device
    joins (the new default) against the host joins (what a refused key takes), bit for bit."""
    import torch
    from protenix_reference import make_pairformer_block, remap_pairformer_block

    dev = T.get_device()
    monkeypatch.setattr(T, "SEQ_LEN_MORE_CHUNKING", 64)
    monkeypatch.setattr(T, "_CONCAT_HOST_BYTES", 0)
    c_z, c_s, L = 128, 384, 160
    monkeypatch.setattr(T, "_TRIMUL_DRAM_SHAPES", {L})      # trimul's large (chunk-joined) path
    _, sd = make_pairformer_block(c_z=c_z, c_s=c_s, seed=0)
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    layer = T.PairformerLayer(32, 4, 24, 16, True, remap_pairformer_block(sd), ck)
    torch.manual_seed(0)
    s, z = torch.randn(1, L, c_s), torch.randn(1, L, L, c_z)
    ft = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    appended = []
    real_append = T._acc_append
    monkeypatch.setattr(T, "_acc_append",
                        lambda acc, t, host: (appended.append(host), real_append(acc, t, host)))

    blocks = {}
    real_helper = T.host_acc_after_refusal

    def helper(key, x, run):
        n = len(appended)
        out = real_helper(key, x, run)
        blocks[key[0]] = blocks.get(key[0], 0) + len(appended) - n
        return out

    monkeypatch.setattr(T, "host_acc_after_refusal", helper)

    def fold():
        appended.clear()
        blocks.clear()
        so, zo = layer(ft(s), ft(z))
        return ttnn.to_torch(so), ttnn.to_torch(zo)

    fallbacks = T.ACC_CONCAT_HOST_FALLBACKS[0]
    on_device = fold()
    assert appended and not any(appended), "a block went to the host on the device-join run"
    assert all(blocks.get(op) for op in ("trimul", "tri_att", "transition")), blocks
    assert T._HOST_ACC_KEYS == set() and T.ACC_CONCAT_HOST_FALLBACKS[0] == fallbacks

    class _Every(set):
        def __contains__(self, key):
            return True

    monkeypatch.setattr(T, "_HOST_ACC_KEYS", _Every())
    on_host = fold()
    assert appended and all(appended), "a block stayed on device on the host-join run"
    assert torch.equal(on_device[0], on_host[0]) and torch.equal(on_device[1], on_host[1])
