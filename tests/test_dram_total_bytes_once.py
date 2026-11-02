"""_dram_total_bytes reads the allocator once per device: a DRAM get_memory_view is 2.8 ms of host
time mid-fold, and the trimul T1 budget asked on every call."""
import types
from unittest import mock

from tt_bio import tenstorrent as T


def test_dram_total_reads_the_allocator_once_per_device():
    reads = []

    def view(device, _kind):
        reads.append(device)
        return types.SimpleNamespace(total_bytes_per_bank=3, num_banks=4 if device is a else 5)

    a, b = object(), object()
    with mock.patch.object(T.ttnn, "get_memory_view", view), mock.patch.dict(T._DRAM_TOTAL, clear=True):
        assert [T._dram_total_bytes(a) for _ in range(3)] == [12, 12, 12]
        assert T._dram_total_bytes(b) == 15
        assert reads == [a, b]


def test_a_failed_read_is_not_remembered():
    calls = []

    def view(_device, _kind):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("no allocator yet")
        return types.SimpleNamespace(total_bytes_per_bank=2, num_banks=2)

    dev = object()
    with mock.patch.object(T.ttnn, "get_memory_view", view), mock.patch.dict(T._DRAM_TOTAL, clear=True):
        assert T._dram_total_bytes(dev) == 0
        assert T._dram_total_bytes(dev) == 4
