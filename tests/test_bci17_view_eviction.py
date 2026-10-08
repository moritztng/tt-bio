"""The tape's read-eviction must not move a parent that the op's own output is a view of.

`_evict_read_parents` moves each L1 parent an op has read down to DRAM, on the reasoning that
once a consumer has run, the forward is done with it. A shape op is not a consumer: its output
is the parent's own storage under another name, so evicting the parent moves the output too,
before anything has read it. On the AF2 triangle attention that put q/k/v in DRAM on the taped
arm and in L1 on the untaped one, and `batched_matmul` picks a different program by placement,
so the two arms of one design round computed different forwards (#17).

These run on CPU with stand-ins for the taped tensors; the device-side check is
`perf/bci_accept/op_trace_diff.py`.
"""

import weakref

import pytest

ttnn = pytest.importorskip("ttnn")
autograd = pytest.importorskip("tt_bio.autograd")


class _Memory:
    def __init__(self, buffer_type):
        self.buffer_type = buffer_type


class _Value:
    def __init__(self, buffer_type):
        self._memory = _Memory(buffer_type)

    def memory_config(self):
        return self._memory


class _Taped:
    """What `_evict_read_parents` reads off an `autograd.Tensor`, and nothing else."""

    def __init__(self, buffer_type=None):
        self.value = _Value(buffer_type if buffer_type is not None else ttnn.BufferType.L1)
        self.node = object()
        self.evictable = True
        self.shares = None
        self.evicted = 0

    def evict(self):
        self.evicted += 1


def _group(*members):
    shares = [weakref.ref(m) for m in members]
    for m in members:
        m.shares = shares
    return shares


def test_a_parent_the_output_is_a_view_of_stays_in_l1():
    parent, view = _Taped(), _Taped()
    _group(parent, view)
    autograd._evict_read_parents([parent], view)
    assert parent.evicted == 0


def test_a_parent_that_was_genuinely_consumed_is_still_evicted():
    parent, out = _Taped(), _Taped()
    autograd._evict_read_parents([parent], out)
    assert parent.evicted == 1


def test_only_the_view_parent_is_spared_when_an_op_reads_two():
    viewed, consumed, out = _Taped(), _Taped(), _Taped()
    _group(viewed, out)
    autograd._evict_read_parents([viewed, consumed], out)
    assert (viewed.evicted, consumed.evicted) == (0, 1)


def test_without_an_output_the_old_behaviour_holds():
    parent = _Taped()
    autograd._evict_read_parents([parent])
    assert parent.evicted == 1


def test_a_dram_parent_is_never_evicted():
    parent = _Taped(ttnn.BufferType.DRAM)
    autograd._evict_read_parents([parent], _Taped())
    assert parent.evicted == 0
