"""`program_cache_for` clears ttnn's program cache when the input shape signature changes, and only then."""
import pytest
import torch

T = pytest.importorskip("tt_bio.tenstorrent")


class FakeDevice:
    def __init__(self):
        self.calls = []

    def disable_and_clear_program_cache(self):
        self.calls.append("clear")

    def enable_program_cache(self):
        self.calls.append("enable")


def test_clears_on_new_shapes_only(monkeypatch):
    monkeypatch.setattr(T.ttnn, "synchronize_device", lambda dev: dev.calls.append("sync"))
    monkeypatch.setattr(T, "_PROGRAM_CACHE_SHAPES", [None])
    monkeypatch.setattr(T, "DRAM_REFUSALS", [0])
    dev = FakeDevice()
    a = T.input_shapes({"ref_pos": torch.zeros(5941, 3), "restype": torch.zeros(772, 32), "seed": 105})
    b = T.input_shapes({"ref_pos": torch.zeros(6202, 3), "restype": torch.zeros(791, 32)})
    T.program_cache_for(a, dev)
    assert dev.calls == ["sync", "clear", "enable"]
    T.program_cache_for(a, dev)                      # warm repeat of the same input keeps its programs
    assert len(dev.calls) == 3
    T.program_cache_for(b, dev)
    assert dev.calls[3:] == ["sync", "clear", "enable"]
    T.program_cache_for(None, dev)                   # None clears every time
    T.program_cache_for(None, dev)
    assert dev.calls.count("clear") == 4


# The refusal ESMFold2-Fast hit on JapanFold (UF-EV-A4-GWH01 chip 10, 2026-10-29), as ttnn raised it.
REFUSAL = ("TT_FATAL @ /project/tt_metal/impl/allocator/bank_manager.cpp:439: false\ninfo:\nOut of Memory: "
           "Not enough space to allocate 277348352 B DRAM buffer across 12 banks, where each bank needs to "
           "store 23113728 B, but bank size is 1040187360 B (allocated: 958510848 B, free: 81676512 B, "
           "largest free block: 22630400 B)")


class CachingDevice(FakeDevice):
    """A chip whose cached programs' binaries are what a fold does not fit beside."""

    def __init__(self, programs):
        super().__init__()
        self.programs = programs

    def num_program_cache_entries(self):
        return self.programs

    def disable_and_clear_program_cache(self):
        super().disable_and_clear_program_cache()
        self.programs = 0


@pytest.fixture
def evicting(monkeypatch):
    monkeypatch.setattr(T.ttnn, "synchronize_device", lambda dev: dev.calls.append("sync"))
    monkeypatch.setattr(T, "_PROGRAM_CACHE_SHAPES", [None])
    monkeypatch.setattr(T, "dram_free", lambda: 0)
    monkeypatch.setattr(T, "_TRIMULS", T.weakref.WeakSet())
    monkeypatch.setattr(T, "_ZERO_MASKS", {})
    monkeypatch.setattr(T, "DRAM_REFUSALS", [0])


def test_a_refusal_with_programs_cached_evicts_them_and_runs_again(evicting):
    dev = CachingDevice(programs=2400)

    def fold():
        if dev.programs:
            raise RuntimeError(REFUSAL)
        return "folded"

    assert T.evicting_programs_on_refusal(fold, dev) == "folded"
    assert dev.calls.count("clear") == 1


@pytest.mark.parametrize("programs, error", [
    (0, REFUSAL),                                     # nothing to evict: the input does not fit
    (2400, "Statically allocated circular buffers on core range clash with L1 buffers"),
])
def test_anything_else_is_raised_as_it_was(evicting, programs, error):
    dev = CachingDevice(programs)
    runs = []

    def fold():
        runs.append(1)
        raise RuntimeError(error)

    with pytest.raises(RuntimeError, match=error[:20]):
        T.evicting_programs_on_refusal(fold, dev)
    assert len(runs) == 1 and "clear" not in dev.calls


def test_a_second_refusal_after_eviction_is_raised(evicting):
    dev = CachingDevice(programs=2400)
    runs = []

    def fold():
        runs.append(1)
        raise RuntimeError(REFUSAL)

    with pytest.raises(RuntimeError, match="Out of Memory"):
        T.evicting_programs_on_refusal(fold, dev)
    assert len(runs) == 2


class WidthVariants:
    """A trimul holding its in-projection at the chunk widths of two earlier lengths."""

    def __init__(self):
        self.gp = {(64, 1): "w64", (32, 2): "w32"}

    def width_variants(self):
        return (self.gp,)

    def drop_width_variants(self):
        self.gp.clear()


def test_a_refusal_drops_per_width_weight_copies_even_with_no_programs(evicting):
    # Program eviction alone left these, and JapanFold's ESMFold2 retries were refused beside them.
    dev, tm = CachingDevice(programs=0), WidthVariants()
    T._TRIMULS.add(tm)

    def fold():
        if tm.gp:
            raise RuntimeError(REFUSAL)
        return "folded"

    assert T.evicting_programs_on_refusal(fold, dev) == "folded"
    assert tm.gp == {}


def test_a_fold_after_a_refused_fold_starts_on_an_evicted_heap(evicting):
    # Mol-1 at 1024 on Wormhole: each warm repeat started with less contiguous DRAM than the last,
    # and the fourth fold was refused (state/mol1/evidence/wh1024leak/ROOT.md).
    dev, tm = CachingDevice(programs=0), WidthVariants()
    T._TRIMULS.add(tm)
    a = T.input_shapes({"ref_pos": torch.zeros(5941, 3)})
    T.program_cache_for(a, dev)
    dev.programs = 1546
    T.program_cache_for(a, dev)                      # a fold nobody refused stays warm
    assert dev.programs == 1546 and tm.gp
    T._dram_oom(RuntimeError("Out of Memory: Not enough space to allocate 4096 B L1 buffer"))
    assert T.DRAM_REFUSALS == [0]                    # an L1 refusal says nothing about the DRAM heap
    assert T._dram_oom(RuntimeError(REFUSAL))        # the fold handled a DRAM refusal
    T.program_cache_for(a, dev)
    assert dev.programs == 0 and tm.gp == {} and T.DRAM_REFUSALS == [0]
    dev.programs = 1546
    T.program_cache_for(a, dev)                      # and the one after a clean fold is warm again
    assert dev.programs == 1546
