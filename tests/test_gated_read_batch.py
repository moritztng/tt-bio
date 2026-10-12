"""The gated move's reader batch: its default, its admissible values and its CB invariant.

`TT_BIO_GATED_READ_BATCH` groups the gated reader's per-row barriers. It is OFF (1) by default and
the kernel's `read_batch == 1` arm is the shipped loop verbatim, so with this untouched the move is
byte-for-byte what it was. What this pins is the part a wrong value would corrupt silently rather
than loudly: the two input CBs must stay a whole multiple of both the reader's batch and the
compute kernel's granule, or a reserve wraps mid-batch and the pages land in a window that is not
contiguous. Host only, no card.
"""

import re
from pathlib import Path

import pytest

from tt_bio import reblock_permute as R

READER = (Path(R.__file__).resolve().parent / "kernels" / "reblock_permute_gated"
          / "reader_reblock_permute_gated.cpp")


def test_default_is_off():
    assert R.GATED_READ_BATCH == 1
    assert R._gated_in_depth() == 2 * R.GATE_GRANULARITY


@pytest.mark.parametrize("value", [0, 3, 5, 6, 12, 32, 64, -1])
def test_only_divisors_of_32_up_to_16(monkeypatch, value):
    monkeypatch.setenv("TT_BIO_GATED_READ_BATCH", str(value))
    assert R._read_batch_env() == 1, "an inadmissible batch must fall back to the shipped 1"
    with pytest.raises(AssertionError):
        R.set_gated_read_batch(value)
    assert R.GATED_READ_BATCH == 1


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("gran", [1, 2, 4])
def test_cb_depth_divides_both_consumers(monkeypatch, batch, gran):
    monkeypatch.setattr(R, "GATED_READ_BATCH", batch)
    monkeypatch.setattr(R, "GATE_GRANULARITY", gran)
    depth = R._gated_in_depth()
    assert depth % batch == 0, "a reserve of `batch` would wrap mid-batch"
    assert depth % gran == 0, "a pop of `gran` would wrap mid-group"
    assert depth >= 2 * batch and depth >= 2 * gran, "no double buffering left"
    # 32 rows a group, so every push is a whole batch and the last one is never ragged.
    assert 32 % batch == 0


@pytest.mark.parametrize("batch", [2, 4, 8, 16])
def test_batch_is_in_the_descriptor_cache_key(monkeypatch, batch):
    """An A/B that flips the batch must not be handed the first arm's compiled program."""
    seen = []

    class _Grid:
        x = y = 8

    class _Dev:
        def id(self):
            return 7

        def compute_with_storage_grid_size(self):
            return _Grid()

    class _T:
        shape = [1, 195, 195, 256]
        dtype = "bf16"
        layout = "TILE"

        def memory_config(self):
            return "L1"

    for b in (1, batch):
        monkeypatch.setattr(R, "GATED_READ_BATCH", b)
        seen.append(R._cache_key_gated(_T(), _T(), _Dev(), [0], [0]))
    assert seen[0] != seen[1]


def test_reader_kernel_reads_the_two_new_runtime_args():
    src = READER.read_text()
    assert "get_arg_val<uint32_t>(9)" in src and "get_arg_val<uint32_t>(10)" in src
    # The host writes 11 per-core reader args; the kernel must not read past them.
    assert not re.search(r"get_arg_val<uint32_t>\((1[1-9]|[2-9]\d)\)", src)
    host = Path(R.__file__).read_text()
    assert "GATED_READ_BATCH, in_bytes]" in host
    # The `== 1` arm has to stay the shipped loop, so the barrier is still per row there.
    assert src.count("noc_async_read_barrier();") >= 3
